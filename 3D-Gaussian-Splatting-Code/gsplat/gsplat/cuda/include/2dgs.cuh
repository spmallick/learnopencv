#ifndef GSPLAT_CUDA_2DGS_CUH
#define GSPLAT_CUDA_2DGS_CUH

#include "types.cuh"
#include "quat.cuh"

#define FILTER_INV_SQUARE 2.0f

namespace gsplat {

template <typename T>
inline __device__ void compute_ray_transforms_aabb_vjp(
    const T *ray_transforms,
    const T *v_means2d,
    const vec3<T> v_normals,
    const mat3<T> W,
    const mat3<T> P,
    const vec3<T> cam_pos,
    const vec3<T> mean_c,
    const vec4<T> quat,
    const vec2<T> scale,
    mat3<T> &_v_ray_transforms,
    vec4<T> &v_quat,
    vec2<T> &v_scale,
    vec3<T> &v_mean
) {
    if (v_means2d[0] != 0 || v_means2d[1] != 0) {
        const T distance = ray_transforms[6] * ray_transforms[6] + ray_transforms[7] * ray_transforms[7] -
                           ray_transforms[8] * ray_transforms[8];
        const T f = 1 / (distance);
        const T dpx_dT00 = f * ray_transforms[6];
        const T dpx_dT01 = f * ray_transforms[7];
        const T dpx_dT02 = -f * ray_transforms[8];
        const T dpy_dT10 = f * ray_transforms[6];
        const T dpy_dT11 = f * ray_transforms[7];
        const T dpy_dT12 = -f * ray_transforms[8];
        const T dpx_dT30 = ray_transforms[0] * (f - 2 * f * f * ray_transforms[6] * ray_transforms[6]);
        const T dpx_dT31 = ray_transforms[1] * (f - 2 * f * f * ray_transforms[7] * ray_transforms[7]);
        const T dpx_dT32 = -ray_transforms[2] * (f + 2 * f * f * ray_transforms[8] * ray_transforms[8]);
        const T dpy_dT30 = ray_transforms[3] * (f - 2 * f * f * ray_transforms[6] * ray_transforms[6]);
        const T dpy_dT31 = ray_transforms[4] * (f - 2 * f * f * ray_transforms[7] * ray_transforms[7]);
        const T dpy_dT32 = -ray_transforms[5] * (f + 2 * f * f * ray_transforms[8] * ray_transforms[8]);

        _v_ray_transforms[0][0] += v_means2d[0] * dpx_dT00;
        _v_ray_transforms[0][1] += v_means2d[0] * dpx_dT01;
        _v_ray_transforms[0][2] += v_means2d[0] * dpx_dT02;
        _v_ray_transforms[1][0] += v_means2d[1] * dpy_dT10;
        _v_ray_transforms[1][1] += v_means2d[1] * dpy_dT11;
        _v_ray_transforms[1][2] += v_means2d[1] * dpy_dT12;
        _v_ray_transforms[2][0] += v_means2d[0] * dpx_dT30 + v_means2d[1] * dpy_dT30;
        _v_ray_transforms[2][1] += v_means2d[0] * dpx_dT31 + v_means2d[1] * dpy_dT31;
        _v_ray_transforms[2][2] += v_means2d[0] * dpx_dT32 + v_means2d[1] * dpy_dT32;
    }

    mat3<T> R = quat_to_rotmat(quat);
    mat3<T> v_M = P * glm::transpose(_v_ray_transforms);
    mat3<T> W_t = glm::transpose(W);
    mat3<T> v_RS = W_t * v_M;
    vec3<T> v_tn = W_t * v_normals;

    // dual visible
    vec3<T> tn = W * R[2];
    T cos = glm::dot(-tn, mean_c);
    T multiplier = cos > 0 ? 1 : -1;
    v_tn *= multiplier;

    mat3<T> v_R = mat3<T>(v_RS[0] * scale[0], v_RS[1] * scale[1], v_tn);

    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);
    v_scale[0] += (T)glm::dot(v_RS[0], R[0]);
    v_scale[1] += (T)glm::dot(v_RS[1], R[1]);

    v_mean += v_RS[2];
}

} // namespace gsplat

#endif // GSPLAT_CUDA_2DGS_CUH