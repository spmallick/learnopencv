#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"
#include "quat.cuh"
#include "quat_scale_to_covar_preci.cuh"
#include "proj.cuh"
#include "transform.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Projection of Gaussians (Single Batch) Backward Pass
 ****************************************************************************/

template <typename T>
__global__ void fully_fused_projection_bwd_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ covars,   // [N, 6] optional
    const T *__restrict__ quats,    // [N, 4] optional
    const T *__restrict__ scales,   // [N, 3] optional
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const T eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const int32_t *__restrict__ radii,   // [C, N]
    const T *__restrict__ conics,        // [C, N, 3]
    const T *__restrict__ compensations, // [C, N] optional
    // grad outputs
    const T *__restrict__ v_means2d,       // [C, N, 2]
    const T *__restrict__ v_depths,        // [C, N]
    const T *__restrict__ v_conics,        // [C, N, 3]
    const T *__restrict__ v_compensations, // [C, N] optional
    // grad inputs
    T *__restrict__ v_means,   // [N, 3]
    T *__restrict__ v_covars,  // [N, 6] optional
    T *__restrict__ v_quats,   // [N, 4] optional
    T *__restrict__ v_scales,  // [N, 3] optional
    T *__restrict__ v_viewmats // [C, 4, 4] optional
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

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;

    // vjp: compute the inverse of the 2d covariance
    mat2<T> covar2d_inv = mat2<T>(conics[0], conics[1], conics[1], conics[2]);
    mat2<T> v_covar2d_inv =
        mat2<T>(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    mat2<T> v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const T compensation = compensations[idx];
        const T v_compensation = v_compensations[idx];
        add_blur_vjp(
            eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
        );
    }

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

    mat3<T> covar;
    vec4<T> quat;
    vec3<T> scale;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = mat3<T>(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quat = glm::make_vec4(quats + gid * 4);
        scale = glm::make_vec3(scales + gid * 3);
        quat_scale_to_covar_preci<T>(quat, scale, &covar, nullptr);
    }
    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    mat3<T> covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // vjp: perspective projection
    T fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3<T> v_covar_c(0.f);
    vec3<T> v_mean_c(0.f);

    switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj_vjp<T>(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj_vjp<T>(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj_vjp<T>(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
    }

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    // vjp: transform Gaussian covariance to camera space
    vec3<T> v_mean(0.f);
    mat3<T> v_covar(0.f);
    mat3<T> v_R(0.f);
    vec3<T> v_t(0.f);
    pos_world_to_cam_vjp(
        R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean
    );
    covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

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
    if (v_covars != nullptr) {
        // Output gradients w.r.t. the covariance matrix
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += gid * 6;
            gpuAtomicAdd(v_covars, v_covar[0][0]);
            gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
            gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
            gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
            gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
            gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
        }
    } else {
        // Directly output gradients w.r.t. the quaternion and scale
        mat3<T> rotmat = quat_to_rotmat<T>(quat);
        vec4<T> v_quat(0.f);
        vec3<T> v_scale(0.f);
        quat_scale_to_covar_vjp<T>(
            quat, scale, rotmat, v_covar, v_quat, v_scale
        );
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
            gpuAtomicAdd(v_scales + 2, v_scale[2]);
        }
    }
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const torch::Tensor &radii,                       // [C, N]
    const torch::Tensor &conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &compensations, // [C, N] optional
    // grad outputs
    const torch::Tensor &v_means2d,                     // [C, N, 2]
    const torch::Tensor &v_depths,                      // [C, N]
    const torch::Tensor &v_conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &v_compensations, // [C, N] optional
    const bool viewmats_requires_grad
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    if (covars.has_value()) {
        GSPLAT_CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        GSPLAT_CHECK_INPUT(quats.value());
        GSPLAT_CHECK_INPUT(scales.value());
    }
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(radii);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_depths);
    GSPLAT_CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        GSPLAT_CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        GSPLAT_CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_covars, v_quats, v_scales; // optional
    if (covars.has_value()) {
        v_covars = torch::zeros_like(covars.value());
    } else {
        v_quats = torch::zeros_like(quats.value());
        v_scales = torch::zeros_like(scales.value());
    }
    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }
    if (C && N) {
        fully_fused_projection_bwd_kernel<float>
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                covars.has_value() ? nullptr : quats.value().data_ptr<float>(),
                covars.has_value() ? nullptr : scales.value().data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                eps2d,
                camera_model,
                radii.data_ptr<int32_t>(),
                conics.data_ptr<float>(),
                compensations.has_value()
                    ? compensations.value().data_ptr<float>()
                    : nullptr,
                v_means2d.data_ptr<float>(),
                v_depths.data_ptr<float>(),
                v_conics.data_ptr<float>(),
                v_compensations.has_value()
                    ? v_compensations.value().data_ptr<float>()
                    : nullptr,
                v_means.data_ptr<float>(),
                covars.has_value() ? v_covars.data_ptr<float>() : nullptr,
                covars.has_value() ? nullptr : v_quats.data_ptr<float>(),
                covars.has_value() ? nullptr : v_scales.data_ptr<float>(),
                viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
            );
    }
    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

} // namespace gsplat