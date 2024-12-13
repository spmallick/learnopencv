#ifndef GSPLAT_CUDA_HELPERS_CUH
#define GSPLAT_CUDA_HELPERS_CUH

#include "types.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>

namespace gsplat {

namespace cg = cooperative_groups;

template <uint32_t DIM, class T, class WarpT>
inline __device__ void warpSum(T *val, WarpT &warp) {
#pragma unroll
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<T>());
    }
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(ScalarT &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec4<ScalarT> &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<ScalarT>());
    val.y = cg::reduce(warp, val.y, cg::plus<ScalarT>());
    val.z = cg::reduce(warp, val.z, cg::plus<ScalarT>());
    val.w = cg::reduce(warp, val.w, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec3<ScalarT> &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<ScalarT>());
    val.y = cg::reduce(warp, val.y, cg::plus<ScalarT>());
    val.z = cg::reduce(warp, val.z, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec2<ScalarT> &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<ScalarT>());
    val.y = cg::reduce(warp, val.y, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat4<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
    warpSum(val[3], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat3<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat2<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpMax(ScalarT &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::greater<ScalarT>());
}

template <typename T> __forceinline__ __device__ T sum(vec3<T> a) {
    return a.x + a.y + a.z;
}

} // namespace gsplat

#endif // GSPLAT_CUDA_HELPERS_CUH
