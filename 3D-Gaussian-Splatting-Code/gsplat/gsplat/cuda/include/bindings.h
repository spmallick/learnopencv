#ifndef GSPLAT_CUDA_BINDINGS_H
#define GSPLAT_CUDA_BINDINGS_H

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

#define GSPLAT_N_THREADS 256

#define GSPLAT_CHECK_CUDA(x)                                                   \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define GSPLAT_CHECK_CONTIGUOUS(x)                                             \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define GSPLAT_CHECK_INPUT(x)                                                  \
    GSPLAT_CHECK_CUDA(x);                                                      \
    GSPLAT_CHECK_CONTIGUOUS(x)
#define GSPLAT_DEVICE_GUARD(_ten)                                              \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define GSPLAT_PRAGMA_UNROLL _Pragma("unroll")

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
#define GSPLAT_CUB_WRAPPER(func, ...)                                          \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)

namespace gsplat {

enum CameraModelType
{
    PINHOLE = 0,
    ORTHO = 1,
    FISHEYE = 2,
};

std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci_fwd_tensor(
    const torch::Tensor &quats,  // [N, 4]
    const torch::Tensor &scales, // [N, 3]
    const bool compute_covar,
    const bool compute_preci,
    const bool triu
);

std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci_bwd_tensor(
    const torch::Tensor &quats,                  // [N, 4]
    const torch::Tensor &scales,                 // [N, 3]
    const at::optional<torch::Tensor> &v_covars, // [N, 3, 3]
    const at::optional<torch::Tensor> &v_precis, // [N, 3, 3]
    const bool triu
);

std::tuple<torch::Tensor, torch::Tensor> proj_fwd_tensor(
    const torch::Tensor &means,  // [C, N, 3]
    const torch::Tensor &covars, // [C, N, 3, 3]
    const torch::Tensor &Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model
);

std::tuple<torch::Tensor, torch::Tensor> proj_bwd_tensor(
    const torch::Tensor &means,  // [C, N, 3]
    const torch::Tensor &covars, // [C, N, 3, 3]
    const torch::Tensor &Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const torch::Tensor &v_means2d, // [C, N, 2]
    const torch::Tensor &v_covars2d // [C, N, 2, 2]
);

std::tuple<torch::Tensor, torch::Tensor> world_to_cam_fwd_tensor(
    const torch::Tensor &means,   // [N, 3]
    const torch::Tensor &covars,  // [N, 3, 3]
    const torch::Tensor &viewmats // [C, 4, 4]
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> world_to_cam_bwd_tensor(
    const torch::Tensor &means,                    // [N, 3]
    const torch::Tensor &covars,                   // [N, 3, 3]
    const torch::Tensor &viewmats,                 // [C, 4, 4]
    const at::optional<torch::Tensor> &v_means_c,  // [C, N, 3]
    const at::optional<torch::Tensor> &v_covars_c, // [C, N, 3, 3]
    const bool means_requires_grad,
    const bool covars_requires_grad,
    const bool viewmats_requires_grad
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
);

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
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles_tensor(
    const torch::Tensor &means2d,                    // [C, N, 2] or [nnz, 2]
    const torch::Tensor &radii,                      // [C, N] or [nnz]
    const torch::Tensor &depths,                     // [C, N] or [nnz]
    const at::optional<torch::Tensor> &camera_ids,   // [nnz]
    const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool double_buffer
);

torch::Tensor isect_offset_encode_tensor(
    const torch::Tensor &isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2]
    const torch::Tensor &conics,                    // [C, N, 3]
    const torch::Tensor &colors,                    // [C, N, D]
    const torch::Tensor &opacities,                 // [N]
    const at::optional<torch::Tensor> &backgrounds, // [C, D]
    const at::optional<torch::Tensor> &mask, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2]
    const torch::Tensor &conics,                    // [C, N, 3]
    const torch::Tensor &colors,                    // [C, N, 3]
    const torch::Tensor &opacities,                 // [N]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &mask, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad
);

std::tuple<torch::Tensor, torch::Tensor> rasterize_to_indices_in_range_tensor(
    const uint32_t range_start,
    const uint32_t range_end,           // iteration steps
    const torch::Tensor transmittances, // [C, image_height, image_width]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
);

torch::Tensor compute_sh_fwd_tensor(
    const uint32_t degrees_to_use,
    const torch::Tensor &dirs,              // [..., 3]
    const torch::Tensor &coeffs,            // [..., K, 3]
    const at::optional<torch::Tensor> masks // [...]
);
std::tuple<torch::Tensor, torch::Tensor> compute_sh_bwd_tensor(
    const uint32_t K,
    const uint32_t degrees_to_use,
    const torch::Tensor &dirs,               // [..., 3]
    const torch::Tensor &coeffs,             // [..., K, 3]
    const at::optional<torch::Tensor> masks, // [...]
    const torch::Tensor &v_colors,           // [..., 3]
    bool compute_v_dirs
);

/****************************************************************************************
 * Packed Version
 ****************************************************************************************/
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_packed_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6]
    const at::optional<torch::Tensor> &quats,  // [N, 3]
    const at::optional<torch::Tensor> &scales, // [N, 3]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_packed_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6]
    const at::optional<torch::Tensor> &quats,  // [N, 4]
    const at::optional<torch::Tensor> &scales, // [N, 3]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const torch::Tensor &camera_ids,                  // [nnz]
    const torch::Tensor &gaussian_ids,                // [nnz]
    const torch::Tensor &conics,                      // [nnz, 3]
    const at::optional<torch::Tensor> &compensations, // [nnz] optional
    // grad outputs
    const torch::Tensor &v_means2d,                     // [nnz, 2]
    const torch::Tensor &v_depths,                      // [nnz]
    const torch::Tensor &v_conics,                      // [nnz, 3]
    const at::optional<torch::Tensor> &v_compensations, // [nnz] optional
    const bool viewmats_requires_grad,
    const bool sparse_grad
);

std::tuple<torch::Tensor, torch::Tensor> compute_relocation_tensor(
    torch::Tensor &opacities,
    torch::Tensor &scales,
    torch::Tensor &ratios,
    torch::Tensor &binoms,
    const int n_max
);

//====== 2DGS ======//
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_fwd_2dgs_tensor(
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 4]
    const torch::Tensor &scales,   // [N, 3]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_bwd_2dgs_tensor(
    // fwd inputs
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 4]
    const torch::Tensor &scales,   // [N, 3]
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
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_to_pixels_fwd_2dgs_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &ray_transforms,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const torch::Tensor &normals,   // [C, N, 3] or [nnz, 3]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_to_pixels_bwd_2dgs_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &ray_transforms,    // [C, N, 3, 3] or [nnz, 3, 3]
    const torch::Tensor &colors,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities, // [C, N] or [nnz]
    const torch::Tensor &normals,   // [C, N, 3] or [nnz, 3],
    const torch::Tensor &densify,
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // ray_crossions
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor
        &render_colors, // [C, image_height, image_width, COLOR_DIM]
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    const torch::Tensor &median_ids,    // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
    const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
    const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
    // options
    bool absgrad
);

std::tuple<torch::Tensor, torch::Tensor>
rasterize_to_indices_in_range_2dgs_tensor(
    const uint32_t range_start,
    const uint32_t range_end,           // iteration steps
    const torch::Tensor transmittances, // [C, image_height, image_width]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &ray_transforms,    // [C, N, 3, 3]
    const torch::Tensor &opacities, // [C, N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_packed_fwd_2dgs_tensor(
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &quats,    // [N, 3]
    const torch::Tensor &scales,   // [N, 3]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float radius_clip
);

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
    const torch::Tensor &v_normals, // [nnz, 3]
    const torch::Tensor &v_ray_transforms,  // [nnz, 3, 3]
    const bool viewmats_requires_grad,
    const bool sparse_grad
);

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
    const uint32_t M);

} // namespace gsplat

#endif // GSPLAT_CUDA_BINDINGS_H
