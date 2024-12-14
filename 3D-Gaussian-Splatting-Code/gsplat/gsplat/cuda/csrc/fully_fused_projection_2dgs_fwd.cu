#include "bindings.h"
#include "helpers.cuh"
#include "transform.cuh"
#include "2dgs.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Projection of Gaussians (Single Batch) Forward Pass 2DGS
 ****************************************************************************/

template <typename T>
__global__ void fully_fused_projection_fwd_2dgs_kernel(
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]:  Gaussian means. (i.e. source points)
    const T *__restrict__ quats,    // [N, 4]:  Quaternions (No need to be normalized): This is the rotation component (for 2D)
    const T *__restrict__ scales,   // [N, 3]:  Scales. [N, 3] scales for x, y, z
    const T *__restrict__ viewmats, // [C, 4, 4]:  Camera-to-World coordinate mat
                                    // [R t]
                                    // [0 1]
    const T *__restrict__ Ks,       // [C, 3, 3]:  Projective transformation matrix
                                    // [f_x 0  c_x]
                                    // [0  f_y c_y]
                                    // [0   0   1]  : f_x, f_y are focal lengths, c_x, c_y is coords for camera center on screen space
    const int32_t image_width,       // Image width  pixels
    const int32_t image_height,      // Image height pixels
    const T near_plane,              // Near clipping plane (for finite range used in z sorting)
    const T far_plane,               // Far clipping plane (for finite range used in z sorting)
    const T radius_clip,             // Radius clipping threshold (through away small primitives)
    // outputs
    int32_t *__restrict__ radii, // [C, N]   The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [C, N].
    T *__restrict__ means2d,     // [C, N, 2] 2D means of the projected Gaussians.
    T *__restrict__ depths,      // [C, N] The z-depth of the projected Gaussians.
    T *__restrict__ ray_transforms,      // [C, N, 3, 3] Transformation matrices that transform xy-planes in pixel spaces into splat coordinates (WH)^T in equation (9) in paper
    T *__restrict__ normals      // [C, N, 3] The normals in camera spaces.
) {

    /**
     * ===============================================
     * Initialize execution and threading variables:
     * idx: global thread index
     * cid: camera id (N is the total number of primitives, C is the number of cameras)
     * gid: gaussian id (N is the total number of primitives, C is the number of cameras)

     * THIS KERNEL LAUNCHES PER PRIMITIVE PER CAMERA i.e. C*N THREADS IN TOTAL
     * ===============================================
    */

    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();  // get the thread index from grid
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    /**
     * ===============================================
     * Load data and put together camera rotation / translation
     * ===============================================
    */

    // shift pointers to the current camera and gaussian
    means += gid * 3;      // find the mean of the primitive this thread is responsible for
    viewmats += cid * 16;  // step 4x4 camera matrix
    Ks += cid * 9;         // step 3x3 intrinsic matrix

    // glm is column-major but input is row-major
    // rotation component of the camera. Explicit Transpose
    mat3<T> R = mat3<T>(
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
    // translation component of the camera
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    /**
     * ===============================================
     * Build ray transformation matrix from Primitive to Camera
     * in the original paper, q_ray [xz, yz, z, 1] = WH * q_uv : [u,v,1,1]
     * 
     * Thus: RS_camera = R * H(P->W)

     * Since H matrix (4x4) is defined as:
     * [v_x v_y 0_vec3  t]
     * [0   0   0       1]
     * 
     * thus RS_Camera defined as R * [v_x v_y 0], which gives
     * [R⋅v_x R⋅v_y 0]
     * Thus the only non zero terms will be the first two columns of R
     *
     * This gives the "affine rotation component" from uv to camera space as RS_camera
     *
     * the final addition component will be mean_c, which is the center of primitive in camera space, as
     * q_cam = RS_camera * q_uv + mean_c
     *
     * Like with homogeneous coordinates. if we encode incoming 2d points as [u,v,1], we can have:
     * q_cam = [RS_camera[0,1] | mean_c] * [u,v,1] 
     * ===============================================
    */

    // transform Gaussian center to camera space
    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);

    // return this thread for overly small primitives
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx] = 0;
        return;
    }

    quats += gid * 4;
    scales += gid * 3;

    mat3<T> RS_camera =
        R * quat_to_rotmat<T>(glm::make_vec4(quats)) *
        mat3<T>(scales[0], 0.0      , 0.0,
                0.0      , scales[1], 0.0,
                0.0      , 0.0      , 1.0);

    mat3<T> WH = mat3<T>(RS_camera[0], RS_camera[1], mean_c);

    // projective transformation matrix: Camera -> Screen
    // when write in this order, the matrix is actually K^T as glm will read it in column major order
    // [Ks[0],  0,  0]
    // [0,   Ks[4],  0]
    // [Ks[2], Ks[5],  1]
    mat3<T> world_2_pix =
        mat3<T>(Ks[0], 0.0  , Ks[2],
                0.0  , Ks[4], Ks[5],
                0.0  , 0.0  , 1.0);

    // WH is defined as [R⋅v_x, R⋅v_y, mean_c]: q_uv = [u,v,-1] -> q_cam = [c1,c2,c3]
    // here is the issue, world_2_pix is actually K^T
    // M is thus (KWH)^T = (WH)^T * K^T = (WH)^T * world_2_pix
    // thus M stores the "row majored" version of KWH, or column major version of (KWH)^T
    mat3<T> M = glm::transpose(WH) * world_2_pix;
    /**
     * ===============================================
     * Compute AABB
     * ===============================================
     */

    // compute AABB
    const vec3<T> M0 = vec3<T>(M[0][0], M[0][1], M[0][2]);  // the first column of KWH^T, thus first row of KWH
    const vec3<T> M1 = vec3<T>(M[1][0], M[1][1], M[1][2]);  // the second column of KWH^T, thus second row of KWH
    const vec3<T> M2 = vec3<T>(M[2][0], M[2][1], M[2][2]);  // the third column of KWH^T, thus third row of KWH

    // we know that KWH brings [u,v,-1] to ray1, ray2, ray3] = [xz, yz, z]
    // temp_point is [1,1,-1], which is a "corner" of the UV space.
    const vec3<T> temp_point = vec3<T>(1.0f, 1.0f, -1.0f);

    // ==============================================
    // trivial implementation to find mean and 1 sigma radius
    // ==============================================
    // const vec3<T> mean_ray = glm::transpose(M) * vec3<T>(0.0f, 0.0f, -1.0f);
    // const vec3<T> temp_point_ray = glm::transpose(M) * temp_point;

    // const vec2<T> mean2d = vec2<T>(mean_ray.x / mean_ray.z, mean_ray.y / mean_ray.z);
    // const vec2<T> half_extend_p = vec2<T>(temp_point_ray.x / temp_point_ray.z, temp_point_ray.y / temp_point_ray.z) - mean2d;
    // const vec2<T> half_extend = vec2<T>(half_extend_p.x * half_extend_p.x, half_extend_p.y * half_extend_p.y);

    // ==============================================
    // pro implementation
    // ==============================================
    // this is purely resulted from algebraic manipulation
    // check here for details: https://github.com/hbb1/diff-surfel-rasterization/issues/8#issuecomment-2138069016
    const T distance = sum(temp_point * M2 * M2);

    // ill-conditioned primitives will have distance = 0.0f, we ignore them
    if (distance == 0.0f)
        return;

    const vec3<T> f = (1 / distance) * temp_point;
    const vec2<T> mean2d = vec2<T>(sum(f * M0 * M2), sum(f * M1 * M2));

    const vec2<T> temp = {sum(f * M0 * M0), sum(f * M1 * M1)};
    const vec2<T> half_extend = mean2d * mean2d - temp;

    // ==============================================
    const T radius =
        ceil(3.f * sqrt(max(1e-4, max(half_extend.x, half_extend.y))));

    if (radius <= radius_clip) {
        radii[idx] = 0;
        return;
    }

    // CULLING STEP:
    // mask out gaussians outside the image region
    if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
        mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
        radii[idx] = 0;  
        return;
    }

    // normals dual visible
    vec3<T> normal = RS_camera[2];
    // flip normal if it is pointing away from the camera
    T multipler = glm::dot(-normal, mean_c) > 0 ? 1 : -1;
    normal *= multipler;

    // write to outputs
    radii[idx] = (int32_t)radius;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;

    // row major storing (KWH)
    ray_transforms[idx * 9] = M0.x;
    ray_transforms[idx * 9 + 1] = M0.y;
    ray_transforms[idx * 9 + 2] = M0.z;
    ray_transforms[idx * 9 + 3] = M1.x;
    ray_transforms[idx * 9 + 4] = M1.y;
    ray_transforms[idx * 9 + 5] = M1.z;
    ray_transforms[idx * 9 + 6] = M2.x;
    ray_transforms[idx * 9 + 7] = M2.y;
    ray_transforms[idx * 9 + 8] = M2.z;

    // primitive normals
    normals[idx * 3] = normal.x;
    normals[idx * 3 + 1] = normal.y;
    normals[idx * 3 + 2] = normal.z;
}

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

    torch::Tensor radii =
        torch::empty({C, N}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor ray_transforms = torch::empty({C, N, 3, 3}, means.options());
    torch::Tensor normals = torch::empty({C, N, 3}, means.options());

    if (C && N) {
        fully_fused_projection_fwd_2dgs_kernel<float>
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
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
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                ray_transforms.data_ptr<float>(),
                normals.data_ptr<float>()
            );
    }
    return std::make_tuple(radii, means2d, depths, ray_transforms, normals);
}

} // namespace gsplat