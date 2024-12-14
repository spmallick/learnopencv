#include "bindings.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

template<typename T>
__global__ void selective_adam_update_kernel(
    T* __restrict__ param,
    const T* __restrict__ param_grad,
    T* __restrict__ exp_avg,
    T* __restrict__ exp_avg_sq,
    const bool* tiles_touched,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M
) {
    auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / M;
    if (g_idx >= N) return;
    if (tiles_touched[g_idx]) {
        T Register_param_grad = param_grad[p_idx];
        T Register_exp_avg = exp_avg[p_idx];
        T Register_exp_avg_sq = exp_avg_sq[p_idx];
        Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
        Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;
        T step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);

        param[p_idx] += step;
        exp_avg[p_idx] = Register_exp_avg;
        exp_avg_sq[p_idx] = Register_exp_avg_sq;
    }
}

void selective_adam_update(
    torch::Tensor &param,
    torch::Tensor &param_grad,
    torch::Tensor &exp_avg,
    torch::Tensor &exp_avg_sq,
    torch::Tensor &tiles_touched,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M
) {
    GSPLAT_DEVICE_GUARD(param);
    GSPLAT_CHECK_INPUT(param);
    GSPLAT_CHECK_INPUT(param_grad);
    GSPLAT_CHECK_INPUT(exp_avg);
    GSPLAT_CHECK_INPUT(exp_avg_sq);
    GSPLAT_CHECK_INPUT(tiles_touched);

    const uint32_t cnt = N * M;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    selective_adam_update_kernel<float><<<(cnt + 255) / 256, 256, 0, stream>>>(
        param.data_ptr<float>(),
        param_grad.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        tiles_touched.data_ptr<bool>(),
        lr,
        b1,
        b2,
        eps,
        N,
        M
    );
}

} // namespace gsplat