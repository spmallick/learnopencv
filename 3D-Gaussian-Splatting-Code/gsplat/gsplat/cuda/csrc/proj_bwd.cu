#include "bindings.h"
#include "proj.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Perspective Projection Backward Pass
 ****************************************************************************/

template <typename T>
__global__ void proj_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,  // [C, N, 3]
    const T *__restrict__ covars, // [C, N, 3, 3]
    const T *__restrict__ Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const T *__restrict__ v_means2d,  // [C, N, 2]
    const T *__restrict__ v_covars2d, // [C, N, 2, 2]
    T *__restrict__ v_means,          // [C, N, 3]
    T *__restrict__ v_covars          // [C, N, 3, 3]
) {

    // For now we'll upcast float16 and bfloat16 to float32
    using OpT = typename OpType<T>::type;

    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    // const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += idx * 3;
    covars += idx * 9;
    v_means += idx * 3;
    v_covars += idx * 9;
    Ks += cid * 9;
    v_means2d += idx * 2;
    v_covars2d += idx * 4;

    OpT fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3<OpT> v_covar(0.f);
    vec3<OpT> v_mean(0.f);
    const vec3<OpT> mean = glm::make_vec3(means);
    const mat3<OpT> covar = glm::make_mat3(covars);
    const vec2<OpT> v_mean2d = glm::make_vec2(v_means2d);
    const mat2<OpT> v_covar2d = glm::make_mat2(v_covars2d);

    switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj_vjp<OpT>(
                mean,
                covar,
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                glm::transpose(v_covar2d),
                v_mean2d,
                v_mean,
                v_covar
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj_vjp<OpT>(
                mean,
                covar,
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                glm::transpose(v_covar2d),
                v_mean2d,
                v_mean,
                v_covar
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj_vjp<OpT>(
                mean,
                covar,
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                glm::transpose(v_covar2d),
                v_mean2d,
                v_mean,
                v_covar
            );
            break;
    }

    // write to outputs: glm is column-major but we want row-major
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) { // rows
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t j = 0; j < 3; j++) { // cols
            v_covars[i * 3 + j] = T(v_covar[j][i]);
        }
    }

    GSPLAT_PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) {
        v_means[i] = T(v_mean[i]);
    }
}

std::tuple<torch::Tensor, torch::Tensor> proj_bwd_tensor(
    const torch::Tensor &means,  // [C, N, 3]
    const torch::Tensor &covars, // [C, N, 3, 3]
    const torch::Tensor &Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const torch::Tensor &v_means2d, // [C, N, 2]
    const torch::Tensor &v_covars2d // [C, N, 2, 2]
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(covars);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_covars2d);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    torch::Tensor v_means = torch::empty({C, N, 3}, means.options());
    torch::Tensor v_covars = torch::empty({C, N, 3, 3}, means.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            v_means.scalar_type(),
            "proj_bwd",
            [&]() {
                proj_bwd_kernel<scalar_t>
                    <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                       GSPLAT_N_THREADS,
                       0,
                       stream>>>(
                        C,
                        N,
                        means.data_ptr<scalar_t>(),
                        covars.data_ptr<scalar_t>(),
                        Ks.data_ptr<scalar_t>(),
                        width,
                        height,
                        camera_model,
                        v_means2d.data_ptr<scalar_t>(),
                        v_covars2d.data_ptr<scalar_t>(),
                        v_means.data_ptr<scalar_t>(),
                        v_covars.data_ptr<scalar_t>()
                    );
            }
        );
    }
    return std::make_tuple(v_means, v_covars);
}

} // namespace gsplat