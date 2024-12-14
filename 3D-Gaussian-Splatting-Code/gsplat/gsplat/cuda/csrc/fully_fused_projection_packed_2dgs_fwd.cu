
#include "bindings.h"
#include "helpers.cuh"
#include "quat.cuh"
#include "transform.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Projection of Gaussians (Batched) Forward Pass 2DGS
 ****************************************************************************/

template <typename T>
__global__ void fully_fused_projection_packed_fwd_2dgs_kernel(
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ quats,    // [N, 4]
    const T *__restrict__ scales,   // [N, 3]
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const T near_plane,
    const T far_plane,
    const T radius_clip,
    const int32_t
        *__restrict__ block_accum,    // [C * blocks_per_row] packing helper
    int32_t *__restrict__ block_cnts, // [C * blocks_per_row] packing helper
    // outputs
    int32_t *__restrict__ indptr,       // [C + 1]
    int64_t *__restrict__ camera_ids,   // [nnz]
    int64_t *__restrict__ gaussian_ids, // [nnz]
    int32_t *__restrict__ radii,        // [nnz]
    T *__restrict__ means2d,            // [nnz, 2]
    T *__restrict__ depths,             // [nnz]
    T *__restrict__ ray_transforms,             // [nnz, 3, 3]
    T *__restrict__ normals             // [nnz, 3]
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

    vec2<T> mean2d;
    mat3<T> M;
    T radius;
    vec3<T> normal;
    if (valid) {
        // build ray transformation matrix and transform from world space to
        // camera space
        quats += col_idx * 4;
        scales += col_idx * 3;

        mat3<T> RS_camera =
            R * quat_to_rotmat<T>(glm::make_vec4(quats)) *
            mat3<T>(scales[0], 0.0, 0.0, 0.0, scales[1], 0.0, 0.0, 0.0, 1.0);
        ;
        mat3<T> WH = mat3<T>(RS_camera[0], RS_camera[1], mean_c);

        mat3<T> world_2_pix =
            mat3<T>(Ks[0], 0.0, Ks[2], 0.0, Ks[4], Ks[5], 0.0, 0.0, 1.0);
        M = glm::transpose(WH) * world_2_pix;

        // compute AABB
        const vec3<T> M0 = vec3<T>(M[0][0], M[0][1], M[0][2]);
        const vec3<T> M1 = vec3<T>(M[1][0], M[1][1], M[1][2]);
        const vec3<T> M2 = vec3<T>(M[2][0], M[2][1], M[2][2]);

        const vec3<T> temp_point = vec3<T>(1.0f, 1.0f, -1.0f);
        const T distance = sum(temp_point * M2 * M2);

        if (distance == 0.0f)
            valid = false;

        const vec3<T> f = (1 / distance) * temp_point;
        mean2d = vec2<T>(sum(f * M0 * M2), sum(f * M1 * M2));

        const vec2<T> temp = {sum(f * M0 * M0), sum(f * M1 * M1)};
        const vec2<T> half_extend = mean2d * mean2d - temp;
        radius = ceil(3.f * sqrt(max(1e-4, max(half_extend.x, half_extend.y))));

        if (radius <= radius_clip) {
            valid = false;
        }

        // mask out gaussians outside the image region
        if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
            mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
            valid = false;
        }

        // normal dual visible
        normal = RS_camera[2];
        T multipler = glm::dot(-normal, mean_c) > 0 ? 1 : -1;
        normal *= multipler;
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
            ray_transforms[thread_data * 9] = M[0][0];
            ray_transforms[thread_data * 9 + 1] = M[0][1];
            ray_transforms[thread_data * 9 + 2] = M[0][2];
            ray_transforms[thread_data * 9 + 3] = M[1][0];
            ray_transforms[thread_data * 9 + 4] = M[1][1];
            ray_transforms[thread_data * 9 + 5] = M[1][2];
            ray_transforms[thread_data * 9 + 6] = M[2][0];
            ray_transforms[thread_data * 9 + 7] = M[2][1];
            ray_transforms[thread_data * 9 + 8] = M[2][2];
            normals[thread_data * 3] = normal.x;
            normals[thread_data * 3 + 1] = normal.y;
            normals[thread_data * 3 + 2] = normal.z;
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
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);
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
        fully_fused_projection_packed_fwd_2dgs_kernel<float>
            <<<blocks, threads, 0, stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                near_plane,
                far_plane,
                radius_clip,
                nullptr,
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
    torch::Tensor ray_transforms = torch::empty({nnz, 3, 3}, means.options());
    torch::Tensor normals = torch::empty({nnz, 3}, means.options());

    if (nnz) {
        fully_fused_projection_packed_fwd_2dgs_kernel<float>
            <<<blocks, threads, 0, stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                near_plane,
                far_plane,
                radius_clip,
                block_accum.data_ptr<int32_t>(),
                nullptr,
                indptr.data_ptr<int32_t>(),
                camera_ids.data_ptr<int64_t>(),
                gaussian_ids.data_ptr<int64_t>(),
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                ray_transforms.data_ptr<float>(),
                normals.data_ptr<float>()
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
        ray_transforms,
        normals
    );
}

} // namespace gsplat