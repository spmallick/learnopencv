#include "bindings.h"
#include "quat_scale_to_covar_preci.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Quat-Scale to Covariance and Precision Forward Pass
 ****************************************************************************/

template <typename T>
__global__ void quat_scale_to_covar_preci_fwd_kernel(
    const uint32_t N,
    const T *__restrict__ quats,  // [N, 4]
    const T *__restrict__ scales, // [N, 3]
    const bool triu,
    // outputs
    T *__restrict__ covars, // [N, 3, 3] or [N, 6]
    T *__restrict__ precis  // [N, 3, 3] or [N, 6]
) {

    // For now we'll upcast float16 and bfloat16 to float32
    using OpT = typename OpType<T>::type;

    // parallelize over N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    // shift pointers to the current gaussian
    quats += idx * 4;
    scales += idx * 3;

    // compute the matrices
    mat3<OpT> covar, preci;
    const vec4<OpT> quat = glm::make_vec4(quats);
    const vec3<OpT> scale = glm::make_vec3(scales);
    quat_scale_to_covar_preci(
        quat, scale, covars ? &covar : nullptr, precis ? &preci : nullptr
    );

    // write to outputs: glm is column-major but we want row-major
    if (covars != nullptr) {
        if (triu) {
            covars += idx * 6;
            covars[0] = T(covar[0][0]);
            covars[1] = T(covar[0][1]);
            covars[2] = T(covar[0][2]);
            covars[3] = T(covar[1][1]);
            covars[4] = T(covar[1][2]);
            covars[5] = T(covar[2][2]);
        } else {
            covars += idx * 9;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    covars[i * 3 + j] = T(covar[j][i]);
                }
            }
        }
    }
    if (precis != nullptr) {
        if (triu) {
            precis += idx * 6;
            precis[0] = T(preci[0][0]);
            precis[1] = T(preci[0][1]);
            precis[2] = T(preci[0][2]);
            precis[3] = T(preci[1][1]);
            precis[4] = T(preci[1][2]);
            precis[5] = T(preci[2][2]);
        } else {
            precis += idx * 9;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    precis[i * 3 + j] = T(preci[j][i]);
                }
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci_fwd_tensor(
    const torch::Tensor &quats,  // [N, 4]
    const torch::Tensor &scales, // [N, 3]
    const bool compute_covar,
    const bool compute_preci,
    const bool triu
) {
    GSPLAT_DEVICE_GUARD(quats);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);

    uint32_t N = quats.size(0);

    torch::Tensor covars, precis;
    if (compute_covar) {
        if (triu) {
            covars = torch::empty({N, 6}, quats.options());
        } else {
            covars = torch::empty({N, 3, 3}, quats.options());
        }
    }
    if (compute_preci) {
        if (triu) {
            precis = torch::empty({N, 6}, quats.options());
        } else {
            precis = torch::empty({N, 3, 3}, quats.options());
        }
    }

    if (N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            quats.scalar_type(),
            "quat_scale_to_covar_preci_fwd",
            [&]() {
                quat_scale_to_covar_preci_fwd_kernel<<<
                    (N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    N,
                    quats.data_ptr<float>(),
                    scales.data_ptr<float>(),
                    triu,
                    compute_covar ? covars.data_ptr<float>() : nullptr,
                    compute_preci ? precis.data_ptr<float>() : nullptr
                );
            }
        );
    }
    return std::make_tuple(covars, precis);
}

} // namespace gsplat