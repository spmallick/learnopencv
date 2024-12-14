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
 * Projection of Gaussians (Batched) Backward Pass 2DGS
 ****************************************************************************/

template <typename T>
__global__ void fully_fused_projection_packed_bwd_2dgs_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const uint32_t nnz,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ quats,    // [N, 4]
    const T *__restrict__ scales,   // [N, 3]
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    // fwd outputs
    const int64_t *__restrict__ camera_ids,   // [nnz]
    const int64_t *__restrict__ gaussian_ids, // [nnz]
    const T *__restrict__ ray_transforms,             // [nnz, 3]
    // grad outputs
    const T *__restrict__ v_means2d, // [nnz, 2]
    const T *__restrict__ v_depths,  // [nnz]
    const T *__restrict__ v_normals, // [nnz, 3]
    const bool sparse_grad, // whether the outputs are in COO format [nnz, ...]
    // grad inputs
    T *__restrict__ v_ray_transforms,
    T *__restrict__ v_means,   // [N, 3] or [nnz, 3]
    T *__restrict__ v_quats,   // [N, 4] or [nnz, 4] Optional
    T *__restrict__ v_scales,  // [N, 3] or [nnz, 3] Optional
    T *__restrict__ v_viewmats // [C, 4, 4] Optional
) {
    // parallelize over nnz.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= nnz) {
        return;
    }
    const int64_t cid = camera_ids[idx];   // camera id
    const int64_t gid = gaussian_ids[idx]; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    ray_transforms += idx * 9;

    v_means2d += idx * 2;
    v_normals += idx * 3;
    v_depths += idx;
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
    compute_ray_transforms_aabb_vjp<T>(
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

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    if (sparse_grad) {
        // write out results with sparse layout
        if (v_means != nullptr) {
            v_means += idx * 3;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                v_means[i] = v_mean[i];
            }
        }
        v_quats += idx * 4;
        v_scales += idx * 3;
        v_quats[0] = v_quat[0];
        v_quats[1] = v_quat[1];
        v_quats[2] = v_quat[2];
        v_quats[3] = v_quat[3];
        v_scales[0] = v_scale[0];
        v_scales[1] = v_scale[1];
    } else {
        // write out results with dense layout
        // #if __CUDA_ARCH__ >= 700
        // write out results with warp-level reduction
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
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_packed_bwd_2dgs_tensor(
    // fwd inputs
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 4]
    const torch::Tensor &scales,   // [N, 3]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const torch::Tensor &camera_ids,   // [nnz]
    const torch::Tensor &gaussian_ids, // [nnz]
    const torch::Tensor &ray_transforms,       // [nnz, 3, 3]
    // grad outputs
    const torch::Tensor &v_means2d, // [nnz, 2]
    const torch::Tensor &v_depths,  // [nnz]
    const torch::Tensor &v_ray_transforms,  // [nnz, 3, 3]
    const torch::Tensor &v_normals, // [nnz, 3]
    const bool viewmats_requires_grad,
    const bool sparse_grad
) {

    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(camera_ids);
    GSPLAT_CHECK_INPUT(gaussian_ids);
    GSPLAT_CHECK_INPUT(ray_transforms);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_depths);
    GSPLAT_CHECK_INPUT(v_normals);
    GSPLAT_CHECK_INPUT(v_ray_transforms);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    uint32_t nnz = camera_ids.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means, v_quats, v_scales, v_viewmats;
    if (sparse_grad) {
        v_means = torch::zeros({nnz, 3}, means.options());

        v_quats = torch::zeros({nnz, 4}, quats.options());
        v_scales = torch::zeros({nnz, 3}, scales.options());

        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros({C, 4, 4}, viewmats.options());
        }

    } else {
        v_means = torch::zeros_like(means);

        v_quats = torch::zeros_like(quats);
        v_scales = torch::zeros_like(scales);

        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros_like(viewmats);
        }
    }
    if (nnz) {

        fully_fused_projection_packed_bwd_2dgs_kernel<float>
            <<<(nnz + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
                C,
                N,
                nnz,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                camera_ids.data_ptr<int64_t>(),
                gaussian_ids.data_ptr<int64_t>(),
                ray_transforms.data_ptr<float>(),
                v_means2d.data_ptr<float>(),
                v_depths.data_ptr<float>(),
                v_normals.data_ptr<float>(),
                sparse_grad,
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