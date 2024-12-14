#ifndef GSPLAT_CUDA_UTILS_H
#define GSPLAT_CUDA_UTILS_H

#include "types.cuh"

namespace gsplat {

template <typename T>
inline __device__ void ortho_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * x + cx, fy * y + cy});
}

template <typename T>
inline __device__ void ortho_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * df/dpixx
    // df/dy = fy * df/dpixy
    // df/dz = 0
    v_mean3d += vec3<T>(fx * v_mean2d[0], fy * v_mean2d[1], 0.f);
}

template <typename T>
inline __device__ void persp_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * x * rz + cx, fy * y * rz + cy});
}

template <typename T>
inline __device__ void persp_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d += vec3<T>(
        fx * rz * v_mean2d[0],
        fy * rz * v_mean2d[1],
        -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2
    );

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    T rz3 = rz2 * rz;
    mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d.x += -fx * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += -fy * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
    }
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] +
                  2.f * fx * tx * rz3 * v_J[2][0] +
                  2.f * fy * ty * rz3 * v_J[2][1];
}

template <typename T>
inline __device__ void fisheye_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T eps = 0.0000001f;
    T xy_len = glm::length(glm::vec2({x, y})) + eps;
    T theta = glm::atan(xy_len, z + eps);
    mean2d =
        vec2<T>({x * fx * theta / xy_len + cx, y * fy * theta / xy_len + cy});

    T x2 = x * x + eps;
    T y2 = y * y;
    T xy = x * y;
    T x2y2 = x2 + y2;
    T x2y2z2_inv = 1.f / (x2y2 + z * z);

    T b = glm::atan(xy_len, z) / xy_len / x2y2;
    T a = z * x2y2z2_inv / (x2y2);
    mat3x2<T> J = mat3x2<T>(
        fx * (x2 * a + y2 * b),
        fy * xy * (a - b),
        fx * xy * (a - b),
        fy * (y2 * a + x2 * b),
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv
    );
    cov2d = J * cov3d * glm::transpose(J);
}

template <typename T>
inline __device__ void fisheye_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    const T eps = 0.0000001f;
    T x2 = x * x + eps;
    T y2 = y * y;
    T xy = x * y;
    T x2y2 = x2 + y2;
    T len_xy = length(glm::vec2({x, y})) + eps;
    const T x2y2z2 = x2y2 + z * z;
    T x2y2z2_inv = 1.f / x2y2z2;
    T b = glm::atan(len_xy, z) / len_xy / x2y2;
    T a = z * x2y2z2_inv / (x2y2);
    v_mean3d += vec3<T>(
        fx * (x2 * a + y2 * b) * v_mean2d[0] + fy * xy * (a - b) * v_mean2d[1],
        fx * xy * (a - b) * v_mean2d[0] + fy * (y2 * a + x2 * b) * v_mean2d[1],
        -fx * x * x2y2z2_inv * v_mean2d[0] - fy * y * x2y2z2_inv * v_mean2d[1]
    );

    const T theta = glm::atan(len_xy, z);
    const T J_b = theta / len_xy / x2y2;
    const T J_a = z * x2y2z2_inv / (x2y2);
    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * (x2 * J_a + y2 * J_b),
        fy * xy * (J_a - J_b), // 1st column
        fx * xy * (J_a - J_b),
        fy * (y2 * J_a + x2 * J_b), // 2nd column
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv // 3rd column
    );
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;
    T l4 = x2y2z2 * x2y2z2;

    T E = -l4 * x2y2 * theta + x2y2z2 * x2y2 * len_xy * z;
    T F = 3 * l4 * theta - 3 * x2y2z2 * len_xy * z - 2 * x2y2 * len_xy * z;

    T A = x * (3 * E + x2 * F);
    T B = y * (E + x2 * F);
    T C = x * (E + y2 * F);
    T D = y * (3 * E + y2 * F);

    T S1 = x2 - y2 - z * z;
    T S2 = y2 - x2 - z * z;
    T inv1 = x2y2z2_inv * x2y2z2_inv;
    T inv2 = inv1 / (x2y2 * x2y2 * len_xy);

    T dJ_dx00 = fx * A * inv2;
    T dJ_dx01 = fx * B * inv2;
    T dJ_dx02 = fx * S1 * inv1;
    T dJ_dx10 = fy * B * inv2;
    T dJ_dx11 = fy * C * inv2;
    T dJ_dx12 = 2.f * fy * xy * inv1;

    T dJ_dy00 = dJ_dx01;
    T dJ_dy01 = fx * C * inv2;
    T dJ_dy02 = 2.f * fx * xy * inv1;
    T dJ_dy10 = dJ_dx11;
    T dJ_dy11 = fy * D * inv2;
    T dJ_dy12 = fy * S2 * inv1;

    T dJ_dz00 = dJ_dx02;
    T dJ_dz01 = dJ_dy02;
    T dJ_dz02 = 2.f * fx * x * z * inv1;
    T dJ_dz10 = dJ_dx12;
    T dJ_dz11 = dJ_dy12;
    T dJ_dz12 = 2.f * fy * y * z * inv1;

    T dL_dtx_raw = dJ_dx00 * v_J[0][0] + dJ_dx01 * v_J[1][0] +
                   dJ_dx02 * v_J[2][0] + dJ_dx10 * v_J[0][1] +
                   dJ_dx11 * v_J[1][1] + dJ_dx12 * v_J[2][1];
    T dL_dty_raw = dJ_dy00 * v_J[0][0] + dJ_dy01 * v_J[1][0] +
                   dJ_dy02 * v_J[2][0] + dJ_dy10 * v_J[0][1] +
                   dJ_dy11 * v_J[1][1] + dJ_dy12 * v_J[2][1];
    T dL_dtz_raw = dJ_dz00 * v_J[0][0] + dJ_dz01 * v_J[1][0] +
                   dJ_dz02 * v_J[2][0] + dJ_dz10 * v_J[0][1] +
                   dJ_dz11 * v_J[1][1] + dJ_dz12 * v_J[2][1];
    v_mean3d.x += dL_dtx_raw;
    v_mean3d.y += dL_dty_raw;
    v_mean3d.z += dL_dtz_raw;
}

} // namespace gsplat

#endif // GSPLAT_CUDA_UTILS_H
