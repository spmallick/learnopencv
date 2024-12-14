#include "bindings.h"
#include "helpers.cuh"
#include "transform.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * World to Camera Transformation Backward Pass
 ****************************************************************************/

template <typename T>
__global__ void world_to_cam_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,      // [N, 3]
    const T *__restrict__ covars,     // [N, 3, 3]
    const T *__restrict__ viewmats,   // [C, 4, 4]
    const T *__restrict__ v_means_c,  // [C, N, 3]
    const T *__restrict__ v_covars_c, // [C, N, 3, 3]
    T *__restrict__ v_means,          // [N, 3]
    T *__restrict__ v_covars,         // [N, 3, 3]
    T *__restrict__ v_viewmats        // [C, 4, 4]
) {

    // For now we'll upcast float16 and bfloat16 to float32
    using OpT = typename OpType<T>::type;

    // parallelize over C * N.
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    covars += gid * 9;
    viewmats += cid * 16;

    // glm is column-major but input is row-major
    const mat3<OpT> R = mat3<OpT>(
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
    const vec3<OpT> t = vec3<OpT>(viewmats[3], viewmats[7], viewmats[11]);

    vec3<OpT> v_mean(0.f);
    mat3<OpT> v_covar(0.f);
    mat3<OpT> v_R(0.f);
    vec3<OpT> v_t(0.f);

    if (v_means_c != nullptr) {
        const vec3<OpT> v_mean_c = glm::make_vec3(v_means_c + idx * 3);
        const vec3<OpT> mean = glm::make_vec3(means);
        pos_world_to_cam_vjp<OpT>(R, t, mean, v_mean_c, v_R, v_t, v_mean);
    }
    if (v_covars_c != nullptr) {
        const mat3<OpT> v_covar_c_t = glm::make_mat3(v_covars_c + idx * 9);
        const mat3<OpT> v_covar_c = glm::transpose(v_covar_c_t);
        const mat3<OpT> covar = glm::make_mat3(covars);
        covar_world_to_cam_vjp<OpT>(R, covar, v_covar_c, v_R, v_covar);
    }

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
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += gid * 9;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_covars + i * 3 + j, T(v_covar[j][i]));
                }
            }
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
                    gpuAtomicAdd(v_viewmats + i * 4 + j, T(v_R[j][i]));
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, T(v_t[i]));
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> world_to_cam_bwd_tensor(
    const torch::Tensor &means,                    // [N, 3]
    const torch::Tensor &covars,                   // [N, 3, 3]
    const torch::Tensor &viewmats,                 // [C, 4, 4]
    const at::optional<torch::Tensor> &v_means_c,  // [C, N, 3]
    const at::optional<torch::Tensor> &v_covars_c, // [C, N, 3, 3]
    const bool means_requires_grad,
    const bool covars_requires_grad,
    const bool viewmats_requires_grad
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(covars);
    GSPLAT_CHECK_INPUT(viewmats);
    if (v_means_c.has_value()) {
        GSPLAT_CHECK_INPUT(v_means_c.value());
    }
    if (v_covars_c.has_value()) {
        GSPLAT_CHECK_INPUT(v_covars_c.value());
    }
    uint32_t N = means.size(0);
    uint32_t C = viewmats.size(0);

    torch::Tensor v_means, v_covars, v_viewmats;
    if (means_requires_grad) {
        v_means = torch::zeros({N, 3}, means.options());
    }
    if (covars_requires_grad) {
        v_covars = torch::zeros({N, 3, 3}, means.options());
    }
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros({C, 4, 4}, means.options());
    }

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means.scalar_type(),
            "world_to_cam_bwd",
            [&]() {
                world_to_cam_bwd_kernel<scalar_t>
                    <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                       GSPLAT_N_THREADS,
                       0,
                       stream>>>(
                        C,
                        N,
                        means.data_ptr<scalar_t>(),
                        covars.data_ptr<scalar_t>(),
                        viewmats.data_ptr<scalar_t>(),
                        v_means_c.has_value()
                            ? v_means_c.value().data_ptr<scalar_t>()
                            : nullptr,
                        v_covars_c.has_value()
                            ? v_covars_c.value().data_ptr<scalar_t>()
                            : nullptr,
                        means_requires_grad ? v_means.data_ptr<scalar_t>()
                                            : nullptr,
                        covars_requires_grad ? v_covars.data_ptr<scalar_t>()
                                             : nullptr,
                        viewmats_requires_grad ? v_viewmats.data_ptr<scalar_t>()
                                               : nullptr
                    );
            }
        );
    }
    return std::make_tuple(v_means, v_covars, v_viewmats);
}

} // namespace gsplat