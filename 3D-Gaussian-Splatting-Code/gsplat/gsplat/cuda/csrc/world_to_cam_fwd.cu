#include "bindings.h"
#include "transform.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * World to Camera Transformation Forward Pass
 ****************************************************************************/

template <typename T>
__global__ void world_to_cam_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ covars,   // [N, 3, 3]
    const T *__restrict__ viewmats, // [C, 4, 4]
    T *__restrict__ means_c,        // [C, N, 3]
    T *__restrict__ covars_c        // [C, N, 3, 3]
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

    if (means_c != nullptr) {
        vec3<OpT> mean_c;
        const vec3<OpT> mean = glm::make_vec3(means);
        pos_world_to_cam(R, t, mean, mean_c);
        means_c += idx * 3;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t i = 0; i < 3; i++) { // rows
            means_c[i] = T(mean_c[i]);
        }
    }

    // write to outputs: glm is column-major but we want row-major
    if (covars_c != nullptr) {
        mat3<OpT> covar_c;
        const mat3<OpT> covar = glm::make_mat3(covars);
        covar_world_to_cam<OpT>(R, covar, covar_c);
        covars_c += idx * 9;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t i = 0; i < 3; i++) { // rows
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t j = 0; j < 3; j++) { // cols
                covars_c[i * 3 + j] = T(covar_c[j][i]);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> world_to_cam_fwd_tensor(
    const torch::Tensor &means,   // [N, 3]
    const torch::Tensor &covars,  // [N, 3, 3]
    const torch::Tensor &viewmats // [C, 4, 4]
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(covars);
    GSPLAT_CHECK_INPUT(viewmats);

    uint32_t N = means.size(0);
    uint32_t C = viewmats.size(0);

    torch::Tensor means_c = torch::empty({C, N, 3}, means.options());
    torch::Tensor covars_c = torch::empty({C, N, 3, 3}, means.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means.scalar_type(),
            "world_to_cam_bwd",
            [&]() {
                world_to_cam_fwd_kernel<scalar_t>
                    <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                       GSPLAT_N_THREADS,
                       0,
                       stream>>>(
                        C,
                        N,
                        means.data_ptr<scalar_t>(),
                        covars.data_ptr<scalar_t>(),
                        viewmats.data_ptr<scalar_t>(),
                        means_c.data_ptr<scalar_t>(),
                        covars_c.data_ptr<scalar_t>()
                    );
            }
        );
    }
    return std::make_tuple(means_c, covars_c);
}

} // namespace gsplat