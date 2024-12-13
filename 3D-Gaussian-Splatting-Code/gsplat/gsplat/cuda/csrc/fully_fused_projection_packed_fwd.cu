#include "bindings.h"
#include "utils.cuh"
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
 * Projection of Gaussians (Batched) Forward Pass
 ****************************************************************************/

template <typename T>
__global__ void fully_fused_projection_packed_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ covars,   // [N, 6] Optional
    const T *__restrict__ quats,    // [N, 4] Optional
    const T *__restrict__ scales,   // [N, 3] Optional
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const T eps2d,
    const T near_plane,
    const T far_plane,
    const T radius_clip,
    const int32_t
        *__restrict__ block_accum,    // [C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ block_cnts, // [C * blocks_per_row] packing helper
    int32_t *__restrict__ indptr,       // [C + 1]
    int64_t *__restrict__ camera_ids,   // [nnz]
    int64_t *__restrict__ gaussian_ids, // [nnz]
    int32_t *__restrict__ radii,        // [nnz]
    T *__restrict__ means2d,            // [nnz, 2]
    T *__restrict__ depths,             // [nnz]
    T *__restrict__ conics,             // [nnz, 3]
    T *__restrict__ compensations       // [nnz] optional
) {
    int32_t blocks_per_row = gridDim.x;

    int32_t row_idx = blockIdx.y; // cid
    int32_t block_col_idx = blockIdx.x;
    int32_t block_idx = row_idx * blocks_per_row + block_col_idx;

    int32_t col_idx = block_col_idx * blockDim.x + threadIdx.x; // gid

    bool valid = (row_idx < C) && (col_idx < N);

    // check if points are with camera near and far plane
    vec3<T> mean_c;
    mat3<T> R;
    if (valid) {
        // shift pointers to the current camera and gaussian
        means += col_idx * 3;
        viewmats += row_idx * 16;

        // glm is column-major but input is row-major
        R = mat3<T>(
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

        // transform Gaussian center to camera space
        pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
        if (mean_c.z < near_plane || mean_c.z > far_plane) {
            valid = false;
        }
    }

    // check if the perspective projection is valid.
    mat2<T> covar2d;
    vec2<T> mean2d;
    mat2<T> covar2d_inv;
    T compensation;
    T det;
    if (valid) {
        // transform Gaussian covariance to camera space
        mat3<T> covar;
        if (covars != nullptr) {
            // if a precomputed covariance is provided
            covars += col_idx * 6;
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
            // if not then compute it from quaternions and scales
            quats += col_idx * 4;
            scales += col_idx * 3;
            quat_scale_to_covar_preci<T>(
                glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
            );
        }
        mat3<T> covar_c;
        covar_world_to_cam(R, covar, covar_c);
        
        Ks += row_idx * 9;
        switch (camera_model) {
            case CameraModelType::PINHOLE: // perspective projection
                persp_proj<T>(
                    mean_c,
                    covar_c,
                    Ks[0],
                    Ks[4],
                    Ks[2],
                    Ks[5],
                    image_width,
                    image_height,
                    covar2d,
                    mean2d
                );
                break;
            case CameraModelType::ORTHO: // orthographic projection
                ortho_proj<T>(
                    mean_c,
                    covar_c,
                    Ks[0],
                    Ks[4],
                    Ks[2],
                    Ks[5],
                    image_width,
                    image_height,
                    covar2d,
                    mean2d
                );
                break;
            case CameraModelType::FISHEYE: // fisheye projection
                fisheye_proj<T>(
                    mean_c,
                    covar_c,
                    Ks[0],
                    Ks[4],
                    Ks[2],
                    Ks[5],
                    image_width,
                    image_height,
                    covar2d,
                    mean2d
                );
                break;
        }

        det = add_blur(eps2d, covar2d, compensation);
        if (det <= 0.f) {
            valid = false;
        } else {
            // compute the inverse of the 2d covariance
            inverse(covar2d, covar2d_inv);
        }
    }

    // check if the points are in the image region
    T radius;
    if (valid) {
        // take 3 sigma as the radius (non differentiable)
        T b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
        T v1 = b + sqrt(max(0.1f, b * b - det));
        T v2 = b - sqrt(max(0.1f, b * b - det));
        radius = ceil(3.f * sqrt(max(v1, v2)));

        if (radius <= radius_clip) {
            valid = false;
        }

        // mask out gaussians outside the image region
        if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
            mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
            valid = false;
        }
    }

    int32_t thread_data = static_cast<int32_t>(valid);
    if (block_cnts != nullptr) {
        // First pass: compute the block-wide sum
        int32_t aggregate;
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockReduce<int32_t, GSPLAT_N_THREADS> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            aggregate = BlockReduce(temp_storage).Sum(thread_data);
        } else {
            aggregate = 0;
        }
        if (threadIdx.x == 0) {
            block_cnts[block_idx] = aggregate;
        }
    } else {
        // Second pass: write out the indices of the non zero elements
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockScan<int32_t, GSPLAT_N_THREADS> BlockScan;
            __shared__ typename BlockScan::TempStorage temp_storage;
            BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
        }
        if (valid) {
            if (block_idx > 0) {
                int32_t offset = block_accum[block_idx - 1];
                thread_data += offset;
            }
            // write to outputs
            camera_ids[thread_data] = row_idx;   // cid
            gaussian_ids[thread_data] = col_idx; // gid
            radii[thread_data] = (int32_t)radius;
            means2d[thread_data * 2] = mean2d.x;
            means2d[thread_data * 2 + 1] = mean2d.y;
            depths[thread_data] = mean_c.z;
            conics[thread_data * 3] = covar2d_inv[0][0];
            conics[thread_data * 3 + 1] = covar2d_inv[0][1];
            conics[thread_data * 3 + 2] = covar2d_inv[1][1];
            if (compensations != nullptr) {
                compensations[thread_data] = compensation;
            }
        }
        // lane 0 of the first block in each row writes the indptr
        if (threadIdx.x == 0 && block_col_idx == 0) {
            if (row_idx == 0) {
                indptr[0] = 0;
                indptr[C] = block_accum[C * blocks_per_row - 1];
            } else {
                indptr[row_idx] = block_accum[block_idx - 1];
            }
        }
    }
}

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

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    auto opt = means.options().dtype(torch::kInt32);

    uint32_t nrows = C;
    uint32_t ncols = N;
    uint32_t blocks_per_row = (ncols + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;

    dim3 threads = {GSPLAT_N_THREADS, 1, 1};
    // limit on the number of blocks: [2**31 - 1, 65535, 65535]
    dim3 blocks = {blocks_per_row, nrows, 1};

    // first pass
    int32_t nnz;
    torch::Tensor block_accum;
    if (C && N) {
        torch::Tensor block_cnts = torch::empty({nrows * blocks_per_row}, opt);
        fully_fused_projection_packed_fwd_kernel<float>
            <<<blocks, threads, 0, stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
                scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                nullptr,
                camera_model,
                block_cnts.data_ptr<int32_t>(),
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr
            );
        block_accum = torch::cumsum(block_cnts, 0, torch::kInt32);
        nnz = block_accum[-1].item<int32_t>();
    } else {
        nnz = 0;
    }

    // second pass
    torch::Tensor indptr = torch::empty({C + 1}, opt);
    torch::Tensor camera_ids = torch::empty({nnz}, opt.dtype(torch::kInt64));
    torch::Tensor gaussian_ids = torch::empty({nnz}, opt.dtype(torch::kInt64));
    torch::Tensor radii =
        torch::empty({nnz}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({nnz, 2}, means.options());
    torch::Tensor depths = torch::empty({nnz}, means.options());
    torch::Tensor conics = torch::empty({nnz, 3}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({nnz}, means.options());
    }

    if (nnz) {
        fully_fused_projection_packed_fwd_kernel<float>
            <<<blocks, threads, 0, stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
                scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                block_accum.data_ptr<int32_t>(),
                camera_model,
                nullptr,
                indptr.data_ptr<int32_t>(),
                camera_ids.data_ptr<int64_t>(),
                gaussian_ids.data_ptr<int64_t>(),
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                conics.data_ptr<float>(),
                calc_compensations ? compensations.data_ptr<float>() : nullptr
            );
    } else {
        indptr.fill_(0);
    }

    return std::make_tuple(
        indptr,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations
    );
}

} // namespace gsplat