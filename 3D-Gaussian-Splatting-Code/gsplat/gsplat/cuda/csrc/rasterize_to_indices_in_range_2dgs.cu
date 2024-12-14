#include "bindings.h"
#include "types.cuh"
#include "2dgs.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Indices in Range 2DGS
 ****************************************************************************/

template <typename T>
__global__ void rasterize_to_indices_in_range_kernel(
    const uint32_t range_start,
    const uint32_t range_end,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec2<T> *__restrict__ means2d, // [C, N, 2]
    const T *__restrict__ ray_transforms,        // [C, N, 3, 3]
    const T *__restrict__ opacities,     // [C, N]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const T *__restrict__ transmittances,     // [C, image_height, image_width]
    const int32_t *__restrict__ chunk_starts, // [C, image_height, image_width]
    int32_t *__restrict__ chunk_cnts,         // [C, image_height, image_width]
    int64_t *__restrict__ gaussian_ids,       // [n_elems]
    int64_t *__restrict__ pixel_ids           // [n_elems]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    // move pointers to the current camera
    tile_offsets += camera_id * tile_height * tile_width;
    transmittances += camera_id * image_height * image_width;

    T px = (T)j + 0.5f;
    T py = (T)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    bool first_pass = chunk_starts == nullptr;
    int32_t base;
    if (!first_pass && inside) {
        chunk_starts += camera_id * image_height * image_width;
        base = chunk_starts[pix_id];
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t isect_range_start = tile_offsets[tile_id];
    int32_t isect_range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (isect_range_end - isect_range_start + block_size - 1) / block_size;

    if (range_start >= num_batches) {
        // this entire tile has been processed in the previous iterations
        // so we don't need to do anything.
        return;
    }

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3<T> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<T> *u_Ms_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]
        ); // [block_size]
    vec3<T> *v_Ms_batch =
        reinterpret_cast<vec3<float> *>(&u_Ms_batch[block_size]
        ); // [block_size]
    vec3<T> *w_Ms_batch =
        reinterpret_cast<vec3<float> *>(&v_Ms_batch[block_size]
        ); // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we (should) use double for it. However double make
    // bwd 1.5x slower so we stick with float for now.
    T trans, next_trans;
    if (inside) {
        trans = transmittances[pix_id];
        next_trans = trans;
    }

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    int32_t cnt = 0;
    for (uint32_t b = range_start; b < min(range_end, num_batches); ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = isect_range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < isect_range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2<T> xy = means2d[g];
            const T opac = opacities[g];
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

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, isect_range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<T> u_M = u_Ms_batch[t];
            const vec3<T> v_M = v_Ms_batch[t];
            const vec3<T> w_M = w_Ms_batch[t];
            const vec3<T> xy_opac = xy_opacity_batch[t];
            const T opac = xy_opac.z;

            const vec3<T> h_u = px * w_M - u_M;
            const vec3<T> h_v = py * w_M - v_M;

            const vec3<T> ray_cross = glm::cross(h_u, h_v);

            if (ray_cross.z == 0.0)
                continue;

            const vec2<T> s = {
                ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z
            };
            const T gauss_weight_3d = s.x * s.x + s.y * s.y;

            // Low pass filter
            const vec2<T> d = {xy_opac.x - px, xy_opac.y - py};
            // 2D screen distance
            const T gauss_weight_2d =
                FILTER_INV_SQUARE * (d.x * d.x + d.y * d.y);
            const T gauss_weight = min(gauss_weight_3d, gauss_weight_2d);

            const T sigma = 0.5f * gauss_weight;
            T alpha = min(0.999f, opac * __expf(-sigma));

            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            next_trans = trans * (1.0f - alpha);
            if (next_trans <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            if (first_pass) {
                // First pass of this function we count the number of gaussians
                // that contribute to each pixel
                cnt += 1;
            } else {
                // Second pass we write out the gaussian ids and pixel ids
                int32_t g = id_batch[t]; // flatten index in [C * N]
                gaussian_ids[base + cnt] = g % N;
                pixel_ids[base + cnt] =
                    pix_id + camera_id * image_height * image_width;
                cnt += 1;
            }

            trans = next_trans;
        }
    }

    if (inside && first_pass) {
        chunk_cnts += camera_id * image_height * image_width;
        chunk_cnts[pix_id] = cnt;
    }
}

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
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(ray_transforms);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);

    uint32_t C = means2d.size(0); // number of cameras
    uint32_t N = means2d.size(1); // number of gaussians
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>) +
         sizeof(vec3<float>) + sizeof(vec3<float>));
    if (cudaFuncSetAttribute(
            rasterize_to_indices_in_range_kernel<float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }

    // First pass: count the number of gaussians that contribute to each pixel
    int64_t n_elems;
    torch::Tensor chunk_starts;
    if (n_isects) {
        torch::Tensor chunk_cnts = torch::zeros(
            {C * image_height * image_width},
            means2d.options().dtype(torch::kInt32)
        );
        rasterize_to_indices_in_range_kernel<float>
            <<<blocks, threads, shared_mem, stream>>>(
                range_start,
                range_end,
                C,
                N,
                n_isects,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                ray_transforms.data_ptr<float>(),
                opacities.data_ptr<float>(),
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                transmittances.data_ptr<float>(),
                nullptr,
                chunk_cnts.data_ptr<int32_t>(),
                nullptr,
                nullptr
            );

        torch::Tensor cumsum =
            torch::cumsum(chunk_cnts, 0, chunk_cnts.scalar_type());
        n_elems = cumsum[-1].item<int64_t>();
        chunk_starts = cumsum - chunk_cnts;
    } else {
        n_elems = 0;
    }

    // Second pass: allocate memory and write out the gaussian and pixel ids.
    torch::Tensor gaussian_ids =
        torch::empty({n_elems}, means2d.options().dtype(torch::kInt64));
    torch::Tensor pixel_ids =
        torch::empty({n_elems}, means2d.options().dtype(torch::kInt64));
    if (n_elems) {
        rasterize_to_indices_in_range_kernel<float>
            <<<blocks, threads, shared_mem, stream>>>(
                range_start,
                range_end,
                C,
                N,
                n_isects,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                ray_transforms.data_ptr<float>(),
                opacities.data_ptr<float>(),
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                transmittances.data_ptr<float>(),
                chunk_starts.data_ptr<int32_t>(),
                nullptr,
                gaussian_ids.data_ptr<int64_t>(),
                pixel_ids.data_ptr<int64_t>()
            );
    }
    return std::make_tuple(gaussian_ids, pixel_ids);
}

} // namespace gsplat