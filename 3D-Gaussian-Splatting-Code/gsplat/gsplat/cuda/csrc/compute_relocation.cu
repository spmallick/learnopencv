#include "bindings.h"

namespace gsplat {

// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
__global__ void compute_relocation_kernel(
    int N,
    float *opacities,
    float *scales,
    int *ratios,
    float *binoms,
    int n_max,
    float *new_opacities,
    float *new_scales
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;

    int n_idx = ratios[idx];
    float denom_sum = 0.0f;

    // compute new opacity
    new_opacities[idx] = 1.0f - powf(1.0f - opacities[idx], 1.0f / n_idx);

    // compute new scale
    for (int i = 1; i <= n_idx; ++i) {
        for (int k = 0; k <= (i - 1); ++k) {
            float bin_coeff = binoms[(i - 1) * n_max + k];
            float term = (pow(-1.0f, k) / sqrt(static_cast<float>(k + 1))) *
                         pow(new_opacities[idx], k + 1);
            denom_sum += (bin_coeff * term);
        }
    }
    float coeff = (opacities[idx] / denom_sum);
    for (int i = 0; i < 3; ++i)
        new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i];
}

std::tuple<torch::Tensor, torch::Tensor> compute_relocation_tensor(
    torch::Tensor &opacities,
    torch::Tensor &scales,
    torch::Tensor &ratios,
    torch::Tensor &binoms,
    const int n_max
) {
    GSPLAT_DEVICE_GUARD(opacities);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(scales);
    GSPLAT_CHECK_INPUT(ratios);
    GSPLAT_CHECK_INPUT(binoms);
    torch::Tensor new_opacities = torch::empty_like(opacities);
    torch::Tensor new_scales = torch::empty_like(scales);

    uint32_t N = opacities.size(0);
    if (N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        compute_relocation_kernel<<<
            (N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
            GSPLAT_N_THREADS,
            0,
            stream>>>(
            N,
            opacities.data_ptr<float>(),
            scales.data_ptr<float>(),
            ratios.data_ptr<int>(),
            binoms.data_ptr<float>(),
            n_max,
            new_opacities.data_ptr<float>(),
            new_scales.data_ptr<float>()
        );
    }
    return std::make_tuple(new_opacities, new_scales);
}

} // namespace gsplat