/// @ref core
/// @file glm/detail/func_common_simd.inl

#if GLM_ARCH & GLM_ARCH_SSE2_BIT

#include "../simd/common.h"

#include <immintrin.h>

namespace glm{
namespace detail
{
	template<qualifier Q>
	struct compute_abs_vector<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& v)
		{
			vec<4, float, Q> result;
			result.data = glm_vec4_abs(v.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_abs_vector<4, int, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, int, Q> call(vec<4, int, Q> const& v)
		{
			vec<4, int, Q> result;
			result.data = glm_ivec4_abs(v.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_floor<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& v)
		{
			vec<4, float, Q> result;
			result.data = glm_vec4_floor(v.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_ceil<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& v)
		{
			vec<4, float, Q> result;
			result.data = glm_vec4_ceil(v.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_fract<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& v)
		{
			vec<4, float, Q> result;
			result.data = glm_vec4_fract(v.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_round<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& v)
		{
			vec<4, float, Q> result;
			result.data = glm_vec4_round(v.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_mod<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& x, vec<4, float, Q> const& y)
		{
			vec<4, float, Q> result;
			result.data = glm_vec4_mod(x.data, y.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_min_vector<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& v1, vec<4, float, Q> const& v2)
		{
			vec<4, float, Q> result;
			result.data = _mm_min_ps(v1.data, v2.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_min_vector<4, int, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, int, Q> call(vec<4, int, Q> const& v1, vec<4, int, Q> const& v2)
		{
			vec<4, int, Q> result;
			result.data = _mm_min_epi32(v1.data, v2.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_min_vector<4, uint, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, uint, Q> call(vec<4, uint, Q> const& v1, vec<4, uint, Q> const& v2)
		{
			vec<4, uint, Q> result;
			result.data = _mm_min_epu32(v1.data, v2.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_max_vector<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& v1, vec<4, float, Q> const& v2)
		{
			vec<4, float, Q> result;
			result.data = _mm_max_ps(v1.data, v2.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_max_vector<4, int, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, int, Q> call(vec<4, int, Q> const& v1, vec<4, int, Q> const& v2)
		{
			vec<4, int, Q> result;
			result.data = _mm_max_epi32(v1.data, v2.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_max_vector<4, uint, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, uint, Q> call(vec<4, uint, Q> const& v1, vec<4, uint, Q> const& v2)
		{
			vec<4, uint, Q> result;
			result.data = _mm_max_epu32(v1.data, v2.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_clamp_vector<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& x, vec<4, float, Q> const& minVal, vec<4, float, Q> const& maxVal)
		{
			vec<4, float, Q> result;
			result.data = _mm_min_ps(_mm_max_ps(x.data, minVal.data), maxVal.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_clamp_vector<4, int, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, int, Q> call(vec<4, int, Q> const& x, vec<4, int, Q> const& minVal, vec<4, int, Q> const& maxVal)
		{
			vec<4, int, Q> result;
			result.data = _mm_min_epi32(_mm_max_epi32(x.data, minVal.data), maxVal.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_clamp_vector<4, uint, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, uint, Q> call(vec<4, uint, Q> const& x, vec<4, uint, Q> const& minVal, vec<4, uint, Q> const& maxVal)
		{
			vec<4, uint, Q> result;
			result.data = _mm_min_epu32(_mm_max_epu32(x.data, minVal.data), maxVal.data);
			return result;
		}
	};

	template<qualifier Q>
	struct compute_mix_vector<4, float, bool, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& x, vec<4, float, Q> const& y, vec<4, bool, Q> const& a)
		{
			__m128i const Load = _mm_set_epi32(-static_cast<int>(a.w), -static_cast<int>(a.z), -static_cast<int>(a.y), -static_cast<int>(a.x));
			__m128 const Mask = _mm_castsi128_ps(Load);

			vec<4, float, Q> Result;
#			if 0 && GLM_ARCH & GLM_ARCH_AVX
				Result.data = _mm_blendv_ps(x.data, y.data, Mask);
#			else
				Result.data = _mm_or_ps(_mm_and_ps(Mask, y.data), _mm_andnot_ps(Mask, x.data));
#			endif
			return Result;
		}
	};
/* FIXME
	template<qualifier Q>
	struct compute_step_vector<float, Q, tvec4>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& edge, vec<4, float, Q> const& x)
		{
			vec<4, float, Q> Result;
			result.data = glm_vec4_step(edge.data, x.data);
			return result;
		}
	};
*/
	template<qualifier Q>
	struct compute_smoothstep_vector<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& edge0, vec<4, float, Q> const& edge1, vec<4, float, Q> const& x)
		{
			vec<4, float, Q> Result;
			Result.data = glm_vec4_smoothstep(edge0.data, edge1.data, x.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_fma<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<4, float, Q> const& a, vec<4, float, Q> const& b, vec<4, float, Q> const& c)
		{
			vec<4, float, Q> Result;
			Result.data = glm_vec4_fma(a.data, b.data, c.data);
			return Result;
		}
	};

	template<qualifier Q>
	struct compute_fma<3, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<3, float, Q> call(vec<3, float, Q> const& a, vec<3, float, Q> const& b, vec<3, float, Q> const& c)
		{
			vec<3, float, Q> Result;
			Result.data = glm_vec4_fma(a.data, b.data, c.data);
			return Result;
		}
	};


	template<qualifier Q>
	struct compute_fma<4, double, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, double, Q> call(vec<4, double, Q> const& a, vec<4, double, Q> const& b, vec<4, double, Q> const& c)
		{
			vec<4, double, Q> Result;
#	if (GLM_ARCH & GLM_ARCH_AVX2_BIT) && !(GLM_COMPILER & GLM_COMPILER_CLANG)
			Result.data = _mm256_fmadd_pd(a.data, b.data, c.data);
#	elif (GLM_ARCH & GLM_ARCH_AVX_BIT)
			Result.data = _mm256_add_pd(_mm256_mul_pd(a.data, b.data), c.data);
#	else
			Result.data.setv(0, _mm_add_pd(_mm_mul_pd(a.data.getv(0), b.data.getv(0)), c.data.getv(0)));
			Result.data.setv(1, _mm_add_pd(_mm_mul_pd(a.data.getv(1), b.data.getv(1)), c.data.getv(1)));
#	endif
			return Result;
		}
	};

	// copy vec3 to vec4 and set w to 0
	template<qualifier Q>
	struct convert_vec3_to_vec4W0<float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<3, float, Q> const& a)
		{
			vec<4, float, Q> v;
#if (GLM_ARCH & GLM_ARCH_SSE41_BIT)
			v.data = _mm_blend_ps(a.data, _mm_setzero_ps(), 8);
#else
			__m128i mask = _mm_set_epi32(0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
			__m128 v0 = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(a.data), mask));
			v.data = v0;
#endif
			return v;
		}
	};

	// copy vec3 to vec4 and set w to 1
	template<qualifier Q>
	struct convert_vec3_to_vec4W1<float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<3, float, Q> const& a)
		{
			vec<4, float, Q> v;
#if (GLM_ARCH & GLM_ARCH_SSE41_BIT)
			v.data = _mm_blend_ps(a.data, _mm_set1_ps(1.0f), 8);
#else
			__m128 t1 = _mm_shuffle_ps(a.data, a.data, _MM_SHUFFLE(0, 2, 1, 3)); //permute x, w
			__m128 t2 = _mm_move_ss(t1, _mm_set_ss(1.0f)); // set x to 1.0f
			v.data = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(0, 2, 1, 3)); //permute x, w
#endif
			return v;
		}
	};

	// copy vec3 to vec4 and set w to vec3.z
	template<qualifier Q>
	struct convert_vec3_to_vec4WZ<float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<3, float, Q> const& a)
		{
			vec<4, float, Q> v;
			v.data = _mm_shuffle_ps(a.data, a.data, _MM_SHUFFLE(2, 2, 1, 0));
			return v;
		}
	};

	// copy vec3 to vec4 and set w to 0
	template<qualifier Q>
	struct convert_vec3_to_vec4W0<double, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, double, Q> call(vec<3, double, Q> const& a)
		{
			vec<4, double, Q> v;
#if (GLM_ARCH & GLM_ARCH_AVX_BIT)
			v.data = _mm256_blend_pd(a.data, _mm256_setzero_pd(), 8);
#else
			v.data.setv(0, a.data.getv(0));
			glm_dvec2 av2 = a.data.getv(1);
			av2 = _mm_shuffle_pd(av2, _mm_setzero_pd(), 2);
			v.data.setv(1, av2);
#endif
			return v;
		}
	};

	// copy vec3 to vec4 and set w to vec3.z
	template<qualifier Q>
	struct convert_vec3_to_vec4WZ<double, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, double, Q> call(vec<3, double, Q> const& a)
		{
			vec<4, double, Q> v;
#if (GLM_ARCH & GLM_ARCH_AVX_BIT)
			v.data = _mm256_permute_pd(a.data, 2);
#else
			v.data.setv(0, a.data.getv(0));
			glm_dvec2 av2 = a.data.getv(1);
			__m128d t1 = _mm_shuffle_pd(av2, av2, 0);
			v.data.setv(1, t1);
#endif
			return v;
		}
	};

	// copy vec3 to vec4 and set w to 1
	template<qualifier Q>
	struct convert_vec3_to_vec4W1<double, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<4, double, Q> call(vec<3, double, Q> const& a)
		{
			vec<4, double, Q> v;
#if (GLM_ARCH & GLM_ARCH_AVX_BIT)
			v.data = _mm256_blend_pd(a.data, _mm256_set1_pd(1.0), 8);
#else
			v.data.setv(0, a.data.getv(0));
			glm_dvec2 av2 = a.data.getv(1);
			av2 = _mm_shuffle_pd(av2, _mm_set1_pd(1.), 2);
			v.data.setv(1, av2);
#endif
			return v;
		}
	};

	template<qualifier Q>
	struct convert_vec4_to_vec3<float, Q, true> {
		GLM_FUNC_QUALIFIER static vec<3, float, Q> call(vec<4, float, Q> const& a)
		{
			vec<3, float, Q> v;
			v.data = a.data;
			return v;
		}
	};

	template<qualifier Q>
	struct convert_vec4_to_vec3<double, Q, true> {
		GLM_FUNC_QUALIFIER static vec<3, double, Q> call(vec<4, double, Q> const& a)
		{
			vec<3, double, Q> v;
#if GLM_ARCH & GLM_ARCH_AVX_BIT
			v.data = a.data;
#else
			v.data.setv(0, a.data.getv(0));
			v.data.setv(1, a.data.getv(1));
#endif
			return v;
		}
	};


	// set all coordinates to same value vec[c]
	template<length_t L, qualifier Q>
	struct convert_splat<L, float, Q, true> {
		template<int c>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, float, Q> call(vec<L, float, Q> const& a)
		{
			vec<L, float, Q> Result;
			const int s = _MM_SHUFFLE(c, c, c, c);
			glm_f32vec4 va = static_cast<glm_f32vec4>(a.data);
#			if GLM_ARCH & GLM_ARCH_AVX_BIT
			Result.data = _mm_permute_ps(va, s);
#			else
			Result.data = _mm_shuffle_ps(va, va, s);
#			endif
			return Result;
		}
	};

	// set all coordinates to same value vec[c]
	template<length_t L, qualifier Q>
	struct convert_splat<L, double, Q, true> {

		template<bool, int c>
		struct detailSSE
		{};

		template<int c>
		struct detailSSE<true, c>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, double, Q> call(vec<L, double, Q> const& a)
			{
				vec<L, double, Q> Result;
				glm_f64vec2 r0 = _mm_shuffle_pd(a.data.getv(0), a.data.getv(0), c | c << 1);
				Result.data.setv(0, r0);
				Result.data.setv(1, r0);
				return Result;
			}
		};

		template<int c>
		struct detailSSE<false, c>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, double, Q> call(vec<L, double, Q> const& a)
			{
				vec<L, double, Q> Result;
				const unsigned int d = static_cast<unsigned int>(c - 2);
				glm_f64vec2 r0 = _mm_shuffle_pd(a.data.getv(1), a.data.getv(1), d | d << 1);
				Result.data.setv(0, r0);
				Result.data.setv(1, r0);
				return Result;
			}
		};

#if GLM_ARCH & GLM_ARCH_AVX_BIT
		template<bool, int c> //note: bool is useless but needed to compil on linux (gcc)
		struct detailAVX
		{};

		template<bool b>
		struct detailAVX<b, 0>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, double, Q> call(vec<L, double, Q> const& a)
			{
				vec<L, double, Q> Result;
				__m256d t1 = _mm256_permute2f128_pd(a.data, a.data, 0x0);
				Result.data = _mm256_permute_pd(t1, 0);
				return Result;
			}
		};

		template<bool b>
		struct detailAVX<b, 1>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, double, Q> call(vec<L, double, Q> const& a)
			{
				vec<L, double, Q> Result;
				__m256d t1 = _mm256_permute2f128_pd(a.data, a.data, 0x0);
				Result.data = _mm256_permute_pd(t1, 0xf);
				return Result;
			}
		};

		template<bool b>
		struct detailAVX<b, 2>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, double, Q> call(vec<L, double, Q> const& a)
			{
				vec<L, double, Q> Result;
				__m256d t2 = _mm256_permute2f128_pd(a.data, a.data, 0x11);
				Result.data = _mm256_permute_pd(t2, 0x0);
				return Result;
			}
		};

		template<bool b>
		struct detailAVX<b, 3>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, double, Q> call(vec<L, double, Q> const& a)
			{
				vec<L, double, Q> Result;
				__m256d t2 = _mm256_permute2f128_pd(a.data, a.data, 0x11);
				Result.data = _mm256_permute_pd(t2, 0xf);
				return Result;
			}
		};
#endif //GLM_ARCH & GLM_ARCH_AVX_BIT

		template<int c>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, double, Q> call(vec<L, double, Q> const& a)
		{
			//return compute_splat<L, double, Q, false>::call<c>(a);
			vec<L, double, Q> Result;
#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
			Result.data = _mm256_permute4x64_pd(a.data, _MM_SHUFFLE(c, c, c, c));
#	elif GLM_ARCH & GLM_ARCH_AVX_BIT
			Result = detailAVX<true, c>::call(a);
#	else
#if 1 //detail<(c <= 1), c>::call2(a) is equivalent to following code but without if constexpr usage
			Result = detailSSE<(c <= 1), c>::call(a);
#else
			if constexpr (c <= 1)
			{
				glm_f64vec2 r0 = _mm_shuffle_pd(a.data.getv(0), a.data.getv(0), c | c << 1);
				Result.data.setv(0, r0);
				Result.data.setv(1, r0);
			}
			else
			{
				const unsigned int d = (unsigned int)(c - 2);
				glm_f64vec2 r0 = _mm_shuffle_pd(a.data.getv(1), a.data.getv(1), d | d << 1);
				Result.data.setv(0, r0);
				Result.data.setv(1, r0);
			}
#endif
#			endif
			return Result;
		}
	};


}//namespace detail
}//namespace glm

#endif//GLM_ARCH & GLM_ARCH_SSE2_BIT

#if GLM_ARCH & GLM_ARCH_NEON_BIT
namespace glm {
	namespace detail {

		template<qualifier Q>
		struct convert_vec3_to_vec4W0<float, Q, true>
		{
			GLM_FUNC_QUALIFIER static vec<4, float, Q> call(vec<3, float, Q> const& a)
			{
				vec<4, float, Q> v;
				static const uint32x4_t mask = { 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };
				v.data = vbslq_f32(mask, a.data, vdupq_n_f32(0));
				return v;
			}
		};

		template<qualifier Q>
		struct convert_vec4_to_vec3<float, Q, true> {
			GLM_FUNC_QUALIFIER static vec<3, float, Q> call(vec<4, float, Q> const& a)
			{
				vec<3, float, Q> v;
				v.data = a.data;
				return v;
			}
		};

		template<length_t L, qualifier Q>
		struct compute_splat<L, float, Q, true> {
			template<int c>
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, float, Q> call(vec<L, float, Q> const& a)
			{}

			template<>
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, float, Q> call<0>(vec<L, float, Q> const& a)
			{
				vec<L, float, Q> Result;
				Result.data = vdupq_lane_f32(vget_low_f32(a.data), 0);
				return Result;
			}

			template<>
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, float, Q> call<1>(vec<L, float, Q> const& a)
			{
				vec<L, float, Q> Result;
				Result.data = vdupq_lane_f32(vget_low_f32(a.data), 1);
				return Result;
			}

			template<>
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, float, Q> call<2>(vec<L, float, Q> const& a)
			{
				vec<L, float, Q> Result;
				Result.data = vdupq_lane_f32(vget_high_f32(a.data), 0);
				return Result;
			}

			template<>
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, float, Q> call<3>(vec<L, float, Q> const& a)
			{
				vec<L, float, Q> Result;
				Result.data = vdupq_lane_f32(vget_high_f32(a.data), 1);
				return Result;
			}
	};

}//namespace detail
}//namespace glm
#endif //GLM_ARCH & GLM_ARCH_NEON_BIT
