#ifndef GSPLAT_CUDA_TYPES_CUH
#define GSPLAT_CUDA_TYPES_CUH

#include <glm/gtc/type_ptr.hpp>
#include <ATen/Dispatch.h>

namespace gsplat {

template <typename T> using vec2 = glm::vec<2, T>;

template <typename T> using vec3 = glm::vec<3, T>;

template <typename T> using vec4 = glm::vec<4, T>;

template <typename T> using mat2 = glm::mat<2, 2, T>;

template <typename T> using mat3 = glm::mat<3, 3, T>;

template <typename T> using mat4 = glm::mat<4, 4, T>;

template <typename T> using mat3x2 = glm::mat<3, 2, T>;

template <typename T> struct OpType {
    typedef T type;
};

template <> struct OpType<__nv_bfloat16> {
    typedef float type;
};

template <> struct OpType<__half> {
    typedef float type;
};

template <> struct OpType<c10::Half> {
    typedef float type;
};

template <> struct OpType<c10::BFloat16> {
    typedef float type;
};

} // namespace gsplat

#endif // GSPLAT_CUDA_TYPES_CUH