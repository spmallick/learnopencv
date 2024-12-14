#ifndef GSPLAT_CUDA_QUAT_SCALE_TO_COVAR_PRECI_CUH
#define GSPLAT_CUDA_QUAT_SCALE_TO_COVAR_PRECI_CUH

#include "types.cuh"
#include "quat.cuh"

namespace gsplat {

template <typename T>
inline __device__ void quat_scale_to_covar_preci(
    const vec4<T> quat,
    const vec3<T> scale,
    // optional outputs
    mat3<T> *covar,
    mat3<T> *preci
) {
    mat3<T> R = quat_to_rotmat<T>(quat);
    if (covar != nullptr) {
        // C = R * S * S * Rt
        mat3<T> S =
            mat3<T>(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
        mat3<T> M = R * S;
        *covar = M * glm::transpose(M);
    }
    if (preci != nullptr) {
        // P = R * S^-1 * S^-1 * Rt
        mat3<T> S = mat3<T>(
            1.0f / scale[0],
            0.f,
            0.f,
            0.f,
            1.0f / scale[1],
            0.f,
            0.f,
            0.f,
            1.0f / scale[2]
        );
        mat3<T> M = R * S;
        *preci = M * glm::transpose(M);
    }
}

template <typename T>
inline __device__ void quat_scale_to_covar_vjp(
    // fwd inputs
    const vec4<T> quat,
    const vec3<T> scale,
    // precompute
    const mat3<T> R,
    // grad outputs
    const mat3<T> v_covar,
    // grad inputs
    vec4<T> &v_quat,
    vec3<T> &v_scale
) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T sx = scale[0], sy = scale[1], sz = scale[2];

    // M = R * S
    mat3<T> S = mat3<T>(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3<T> M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3<T> v_M = (v_covar + glm::transpose(v_covar)) * M;
    mat3<T> v_R = v_M * S;

    // grad for (quat, scale) from covar
    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);

    v_scale[0] +=
        R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2];
    v_scale[1] +=
        R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2];
    v_scale[2] +=
        R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2];
}

template <typename T>
inline __device__ void quat_scale_to_preci_vjp(
    // fwd inputs
    const vec4<T> quat,
    const vec3<T> scale,
    // precompute
    const mat3<T> R,
    // grad outputs
    const mat3<T> v_preci,
    // grad inputs
    vec4<T> &v_quat,
    vec3<T> &v_scale
) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T sx = 1.0f / scale[0], sy = 1.0f / scale[1], sz = 1.0f / scale[2];

    // M = R * S
    mat3<T> S = mat3<T>(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3<T> M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3<T> v_M = (v_preci + glm::transpose(v_preci)) * M;
    mat3<T> v_R = v_M * S;

    // grad for (quat, scale) from preci
    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);

    v_scale[0] +=
        -sx * sx *
        (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
    v_scale[1] +=
        -sy * sy *
        (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
    v_scale[2] +=
        -sz * sz *
        (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);
}

} // namespace gsplat

#endif // GSPLAT_CUDA_QUAT_SCALE_TO_COVAR_PRECI_CUH
