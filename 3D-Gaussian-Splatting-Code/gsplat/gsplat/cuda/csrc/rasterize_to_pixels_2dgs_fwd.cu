#include "bindings.h"
#include "types.cuh"
#include "2dgs.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Forward Pass 2DGS
 ****************************************************************************/

 /**
  *
  */
template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_fwd_2dgs_kernel(
    const uint32_t C,    // number of cameras
    const uint32_t N,    // number of gaussians
    const uint32_t n_isects,  // number of ray-primitive intersections.
    const bool packed,       // whether the input tensors are packed
    const vec2<S> *__restrict__ means2d,   // Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
    const S *__restrict__ ray_transforms,  // transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [C, N, 3, 3] if packed is False, [nnz, channels] if packed is True.
                                           // This is (KWH)^{-1} in the paper (takes screen [x,y] and map to [u,v])
    const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]  // Gaussian colors or ND features.
    const S *__restrict__ opacities,   // [C, N] or [nnz]                        // Gaussian opacities that support per-view values.
    const S *__restrict__ normals,     // [C, N, 3] or [nnz, 3]                  // The normals in camera space.
    const S *__restrict__ backgrounds, // [C, COLOR_DIM]                         // Background colors on camera basis
    const bool *__restrict__ masks,    // [C, tile_height, tile_width]            // Optional tile mask to skip rendering GS to masked tiles.
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]    // Intersection offsets outputs from `isect_offset_encode()`, this is the result of a prefix sum, and
                                                                                 // gives the interval that our gaussians are gonna use.
    const int32_t *__restrict__ flatten_ids,  // [n_isects]                      // The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`.
    
    
    // outputs
    S *__restrict__ render_colors,  // [C, image_height, image_width, COLOR_DIM]
    S *__restrict__ render_alphas,  // [C, image_height, image_width, 1]
    S *__restrict__ render_normals, // [C, image_height, image_width, 3]
    S *__restrict__ render_distort, // [C, image_height, image_width, 1]  // Stores the per-pixel distortion error proposed in Mip-NeRF 360.
    S *__restrict__ render_median,  // [C, image_height, image_width, 1]  // Stores the median depth contribution for each pixel "set to the depth of the Gaussian that brings the accumulated opacity over 0.5."
    int32_t *__restrict__ last_ids, // [C, image_height, image_width]     // Stores the index of the last Gaussian that contributed to each pixel.
    int32_t *__restrict__ median_ids // [C, image_height, image_width]    // Stores the index of the Gaussian that contributes to the median depth for each pixel (bring over 0.5).
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    /**
     * ==============================
     * Thread and block setup:
     * This sets up the thread and block indices, determining which camera, tile, and pixel each thread will process.
     * The grid structure is assigend as:
     * C * tile_height * tile_width blocks (3d grid), each block is a tile.
     * Each thread is responsible for one pixel. (blockSize = tile_size * tile_size)
     * ==============================
     */
    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;  // get the global offset of the tile w.r.t the camera
    render_colors += camera_id * image_height * image_width * COLOR_DIM;  // get the global offset of the pixel w.r.t the camera
    render_alphas += camera_id * image_height * image_width;  // get the global offset of the pixel w.r.t the camera
    last_ids += camera_id * image_height * image_width;  // get the global offset of the pixel w.r.t the camera
    render_normals += camera_id * image_height * image_width * 3;
    render_distort += camera_id * image_height * image_width;
    render_median += camera_id * image_height * image_width;
    median_ids += camera_id * image_height * image_width;

    // get the global offset of the background and mask
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    // find the center of the pixel
    S px = (S)j + 0.5f;
    S py = (S)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile

    // print 
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        // see if this is the last tile in the camera
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;


    /**
     * ==============================
     * Register computing variables:
     * For each pixel, we need to find its uv intersection with the gaussian primitives.
     * then we retrieve the kernel's parameters and kernel weights 
     * do the splatting rendering equation.
     * ==============================
     */
    // Shared memory layout:
    // This memory is laid out as follows:
    // | gaussian indices | x : y : alpha | u | v | w |
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]

    // stores the concatination for projected primitive source (x, y) and opacity alpha
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]

    // these are row vectors of the ray transformation matrices for the current batch of gaussians
    vec3<S> *u_Ms_batch = reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]); // [block_size]
    vec3<S> *v_Ms_batch = reinterpret_cast<vec3<float> *>(&u_Ms_batch[block_size]); // [block_size]
    vec3<S> *w_Ms_batch = reinterpret_cast<vec3<float> *>(&v_Ms_batch[block_size]); // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    // The coefficient for volumetric rendering for our responsible pixel.
    S T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();  

    // Per-pixel distortion error proposed in Mip-NeRF 360.
    // Implemented reference:
    // https://github.com/nerfstudio-project/nerfacc/blob/master/nerfacc/losses.py#L7
    S distort = 0.f;
    S accum_vis_depth = 0.f; // accumulate vis * depth

    // keep track of median depth contribution
    S median_depth = 0.f;
    uint32_t median_idx = 0.f;

    /**
     * ==============================
     * Per-pixel rendering: (2DGS Differntiable Rasterizer Forward Pass)
     * This section is responsible for rendering a single pixel.
     * It processes batches of gaussians and accumulates the pixel color and normal.
     * ==============================
     */

    // TODO (WZ): merge pix_out and normal_out to
    //  S pix_out[COLOR_DIM + 3] = {0.f}
    S pix_out[COLOR_DIM] = {0.f};
    S normal_out[3] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;

        // only threads within the range of the tile will fetch gaussians
        /**
         * Launch this block with each thread responsible for one gaussian.
         */
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            u_Ms_batch[tr] = {
                ray_transforms[g * 9], ray_transforms[g * 9 + 1], ray_transforms[g * 9 + 2]
            };
            v_Ms_batch[tr] = {
                ray_transforms[g * 9 + 3], ray_transforms[g * 9 + 4], ray_transforms[g * 9 + 5]
            };
            w_Ms_batch[tr] = {
                ray_transforms[g * 9 + 6], ray_transforms[g * 9 + 7], ray_transforms[g * 9 + 8]
            };
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        /**
         * ==================================================
         * Forward rasterization pass:
         * ==================================================
         * 
         * GSplat computes rasterization point of intersection as:
         * 1. Generate 2 homogeneous plane parameter vectors as sets of points in UV space
         * 2. Find the set of points that satisfy both conditions with the cross product
         * 3. Find where this solution set intersects with UV plane using projective flattening
         * 
         * For each gaussian G_i and pixel q_xy:
         * 
         * 1. Compute homogeneous plane parameters:
         *    h_u = p_x * M_w - M_u
         *    h_v = p_y * M_w - M_v
         *    where M_u, M_v, M_w are rows of the KWH transform
         *
         * Note: this works because:
         *    for any vector q_uv [u, v, 1], applying co-vector h_u will yield the following expression: 
         *    h_u * [u, v, 1]^T = P_x * (M_w * q_uv) - M_u * q_uv
         *                      = P_x * q_ray.z - q_ray.x * q_ray.z
         *    - where P_x is the x-coordinate of the ray origin
         *    Thus: h_u  defines a set of q_uv where q_uv's projected x coordinate in ray space is P_x
         *    which aligns with the homogeneous plane definition in original 2DGS paper (similar for h_v)
         *
         * 2. Compute intersection:
         *    zeta = h_u × h_v
         *    This cross product is the only solution that satisfies both homogeneous plane equations (dot product == 0)
         * 
         * 3. Project to UV space:
         *    s_uv = [zeta_1/zeta_3, zeta_2/zeta_3]
         *    - since UV space is essentially another ray space, and arbitrary scale of q_uv will not change the result of dot product over orthogonality
         *    - thus, the result is the point of intersection in UV space
         * 
         * 4. Evaluate gaussian kernel:
         *    G_i = exp(-(s_u^2 + s_v^2)/2)
         * 
         * 5. Accumulate color:
         *    p_xy += alpha_i * c_i * G_i * prod(1 - alpha_j * G_j)
         * 
         * This method efficiently computes the point of intersection and 
         * evaluates the gaussian kernel in UV space.
         * Note: in some cases, we use the minimum of ray-intersection kernels and 2D projected gaussian kernels
         */
        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {

            const vec3<S> xy_opac = xy_opacity_batch[t];
            const S opac = xy_opac.z;

            const vec3<S> u_M = u_Ms_batch[t];
            const vec3<S> v_M = v_Ms_batch[t];
            const vec3<S> w_M = w_Ms_batch[t];

            // h_u and h_v are the homogeneous plane representations (they are contravariant to the points on the primitive plane)
            const vec3<S> h_u = px * w_M - u_M;
            const vec3<S> h_v = py * w_M - v_M;

            const vec3<S> ray_cross = glm::cross(h_u, h_v);
            if (ray_cross.z == 0.0)
                continue;

            const vec2<S> s = vec2<S>(ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z);

            // IMPORTANT: This is where the gaussian kernel is evaluated!!!!!

            // point of interseciton in uv space
            const S gauss_weight_3d = s.x * s.x + s.y * s.y;
            
            // projected gaussian kernel
            const vec2<S> d = {xy_opac.x - px, xy_opac.y - py};
            // #define FILTER_INV_SQUARE 2.0f
            const S gauss_weight_2d = FILTER_INV_SQUARE * (d.x * d.x + d.y * d.y);
            
            // merge ray-intersection kernel and 2d gaussian kernel
            const S gauss_weight = min(gauss_weight_3d, gauss_weight_2d);


            const S sigma = 0.5f * gauss_weight;
            // evaluation of the gaussian exponential term
            S alpha = min(0.999f, opac * __expf(-sigma));

            // ignore transparent gaussians
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const S next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            // run volumetric rendering..
            int32_t g = id_batch[t];
            const S vis = alpha * T;
            const S *c_ptr = colors + g * COLOR_DIM;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }

            const S *n_ptr = normals + g * 3;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < 3; ++k) {
                normal_out[k] += n_ptr[k] * vis;
            }

            if (render_distort != nullptr) {
                // the last channel of colors is depth
                const S depth = c_ptr[COLOR_DIM - 1];
                // in nerfacc, loss_bi_0 = weights * t_mids *
                // exclusive_sum(weights)
                const S distort_bi_0 = vis * depth * (1.0f - T);
                // in nerfacc, loss_bi_1 = weights * exclusive_sum(weights *
                // t_mids)
                const S distort_bi_1 = vis * accum_vis_depth;
                distort += 2.0f * (distort_bi_0 - distort_bi_1);
                accum_vis_depth += vis * depth;
            }

            // compute median depth
            if (T > 0.5) {
                median_depth = c_ptr[COLOR_DIM - 1];
                median_idx = batch_start + t;
            }

            cur_idx = batch_start + t;

            T = next_T;
        }
    }
    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        render_alphas[pix_id] = 1.0f - T;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < 3; ++k) {
            render_normals[pix_id * 3 + k] = normal_out[k];
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);

        if (render_distort != nullptr) {
            render_distort[pix_id] = distort;
        }

        render_median[pix_id] = median_depth;
        // index in bin of gaussian that contributes to median depth
        median_ids[pix_id] = static_cast<int32_t>(median_idx);
    }
}

template <uint32_t CDIM>
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &ray_transforms,    // [C, N, 3, 3] or [nnz, 3, 3]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const torch::Tensor &normals,   // [C, N, 3]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(ray_transforms);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(normals);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t channels = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    // we assign one pixel to one thread.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor renders = torch::empty(
        {C, image_height, image_width, channels},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor alphas = torch::empty(
        {C, image_height, image_width, 1},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor last_ids = torch::empty(
        {C, image_height, image_width}, means2d.options().dtype(torch::kInt32)
    );
    torch::Tensor median_ids = torch::empty(
        {C, image_height, image_width}, means2d.options().dtype(torch::kInt32)
    );

    torch::Tensor render_normals = torch::empty(
        {C, image_height, image_width, 3},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor render_distort = torch::empty(
        {C, image_height, image_width, 1},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor render_median = torch::empty(
        {C, image_height, image_width, 1},
        means2d.options().dtype(torch::kFloat32)
    );

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>) +
         sizeof(vec3<float>) + sizeof(vec3<float>));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_fwd_2dgs_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }
    rasterize_to_pixels_fwd_2dgs_kernel<CDIM, float>
        <<<blocks, threads, shared_mem, stream>>>(
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            ray_transforms.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            normals.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_normals.data_ptr<float>(),
            render_distort.data_ptr<float>(),
            render_median.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            median_ids.data_ptr<int32_t>()
        );

    return std::make_tuple(
        renders,
        alphas,
        render_normals,
        render_distort,
        render_median,
        last_ids,
        median_ids
    );
}

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
) {
    GSPLAT_CHECK_INPUT(colors);
    uint32_t channels = colors.size(-1);

#define __GS__CALL_(N)                                                         \
    case N:                                                                    \
        return call_kernel_with_dim<N>(                                        \
            means2d,                                                           \
            ray_transforms,                                                            \
            colors,                                                            \
            opacities,                                                         \
            normals,                                                           \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids                                                        \
        );
    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    switch (channels) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
}

} // namespace gsplat