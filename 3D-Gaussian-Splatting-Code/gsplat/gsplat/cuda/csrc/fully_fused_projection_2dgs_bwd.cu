#include "bindings.h"
#include "helpers.cuh"
#include "transform.cuh"
#include "2dgs.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Projection of Gaussians (Batched) Backward Pass
 ****************************************************************************/
template <typename T>
__global__ void fully_fused_projection_bwd_2dgs_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ quats,    // [N, 4]
    const T *__restrict__ scales,   // [N, 3]
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    // fwd outputs
    const int32_t *__restrict__ radii, // [C, N]
    const T *__restrict__ ray_transforms,      // [C, N, 3, 3]
    // grad outputs
    const T *__restrict__ v_means2d, // [C, N, 2]
    const T *__restrict__ v_depths,  // [C, N]
    const T *__restrict__ v_normals, // [C, N, 3]
    // grad inputs
    T *__restrict__ v_ray_transforms,  // [C, N, 3, 3]
    T *__restrict__ v_means,   // [N, 3]
    T *__restrict__ v_quats,   // [N, 4]
    T *__restrict__ v_scales,  // [N, 3]
    T *__restrict__ v_viewmats // [C, 4, 4]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx] <= 0) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    ray_transforms += idx * 9;

    v_means2d += idx * 2;
    v_depths += idx;
    v_normals += idx * 3;
    v_ray_transforms += idx * 9;

    // transform Gaussian to camera space
    mat3<T> R = mat3<T>(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);
    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);

    vec4<T> quat = glm::make_vec4(quats + gid * 4);
    vec2<T> scale = glm::make_vec2(scales + gid * 3);

    mat3<T> P = mat3<T>(Ks[0], 0.0, Ks[2], 0.0, Ks[4], Ks[5], 0.0, 0.0, 1.0);

    mat3<T> _v_ray_transforms = mat3<T>(
        v_ray_transforms[0],
        v_ray_transforms[1],
        v_ray_transforms[2],
        v_ray_transforms[3],
        v_ray_transforms[4],
        v_ray_transforms[5],
        v_ray_transforms[6],
        v_ray_transforms[7],
        v_ray_transforms[8]
    );

    _v_ray_transforms[2][2] += v_depths[0];

    vec3<T> v_normal = glm::make_vec3(v_normals);

    vec3<T> v_mean(0.f);
    vec2<T> v_scale(0.f);
    vec4<T> v_quat(0.f);
    compute_ray_transforms_aabb_vjp(
        ray_transforms,
        v_means2d,
        v_normal,
        R,
        P,
        t,
        mean_c,
        quat,
        scale,
        _v_ray_transforms,
        v_quat,
        v_scale,
        v_mean
    );

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }

    // Directly output gradients w.r.t. the quaternion and scale
    warpSum(v_quat, warp_group_g);
    warpSum(v_scale, warp_group_g);
    if (warp_group_g.thread_rank() == 0) {
        v_quats += gid * 4;
        v_scales += gid * 3;
        gpuAtomicAdd(v_quats, v_quat[0]);
        gpuAtomicAdd(v_quats + 1, v_quat[1]);
        gpuAtomicAdd(v_quats + 2, v_quat[2]);
        gpuAtomicAdd(v_quats + 3, v_quat[3]);
        gpuAtomicAdd(v_scales, v_scale[0]);
        gpuAtomicAdd(v_scales + 1, v_scale[1]);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_bwd_2dgs_tensor(
    // fwd inputs
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 4]
    const torch::Tensor &scales,   // [N, 2]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const torch::Tensor &radii,  // [C, N]
    const torch::Tensor &ray_transforms, // [C, N, 3, 3]
    // grad outputs
    const torch::Tensor &v_means2d, // [C, N, 2]
    const torch::Tensor &v_depths,  // [C, N]
    const torch::Tensor &v_normals, // [C, N, 3]
    const torch::Tensor &v_ray_transforms,  // [C, N, 3, 3]
    const bool viewmats_requires_grad
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(radii);
    GSPLAT_CHECK_INPUT(ray_transforms);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_depths);
    GSPLAT_CHECK_INPUT(v_normals);
    GSPLAT_CHECK_INPUT(v_ray_transforms);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_quats = torch::zeros_like(quats);
    torch::Tensor v_scales = torch::zeros_like(scales);
    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }
    if (C && N) {
        fully_fused_projection_bwd_2dgs_kernel<float>
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                radii.data_ptr<int32_t>(),
                ray_transforms.data_ptr<float>(),
                v_means2d.data_ptr<float>(),
                v_depths.data_ptr<float>(),
                v_normals.data_ptr<float>(),
                v_ray_transforms.data_ptr<float>(),
                v_means.data_ptr<float>(),
                v_quats.data_ptr<float>(),
                v_scales.data_ptr<float>(),
                viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
            );
    }
    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}

} // namespace gsplat