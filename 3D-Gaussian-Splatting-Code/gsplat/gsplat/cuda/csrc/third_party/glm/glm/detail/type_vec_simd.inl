#pragma once

#define CTORSL(L, CTOR)\
CTOR(L, aligned_lowp)\
CTOR(L, aligned_mediump)\
CTOR(L, aligned_highp)\

namespace glm {
	namespace detail
	{

template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
struct compute_vec_and<L, T, Q, IsInt, Size, true> : public compute_vec_and<L, T, Q, IsInt, Size, false>
{};

template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
struct compute_vec_or<L, T, Q, IsInt, Size, true>: public compute_vec_or<L, T, Q, IsInt, Size, false>
{};

template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
struct compute_vec_xor<L, T, Q, IsInt, Size, true> : public compute_vec_xor<L, T, Q, IsInt, Size, false>
{};

template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
struct compute_vec_shift_left<L, T, Q, IsInt, Size, true> : public compute_vec_shift_left<L, T, Q, IsInt, Size, false>
{};

template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
struct compute_vec_shift_right<L, T, Q, IsInt, Size, true> : public compute_vec_shift_right<L, T, Q, IsInt, Size, false>
{};

template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
struct compute_vec_bitwise_not<L, T, Q, IsInt, Size, true>:public compute_vec_bitwise_not<L, T, Q, IsInt, Size, false>
{};

template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
struct compute_vec_equal<L, T, Q, IsInt, Size, true> : public compute_vec_equal<L, T, Q, IsInt, Size, false>
{};

template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
struct compute_vec_nequal<L, T, Q, IsInt, Size, true> : public compute_vec_nequal<L, T, Q, IsInt, Size, false>
{};

template<length_t L, typename T, qualifier Q>
struct compute_vec_mod<L, T, Q, true> : public compute_vec_mod<L, T, Q, false>
{};


template<typename T, length_t L, qualifier Q>
struct compute_vec_add<L, T, Q, true> : public compute_vec_add<L, T, Q, false>
{};

template< length_t L, typename T,  qualifier Q>
struct compute_vec_sub<L, T, Q, true> : public compute_vec_sub<L, T, Q, false>
{};

template< length_t L, typename T, qualifier Q>
struct compute_vec_mul<L, T, Q, true> : public compute_vec_mul<L, T, Q, false>
{};

template< length_t L, typename T, qualifier Q>
struct compute_vec_div<L, T, Q, true> : public compute_vec_div<L, T, Q, false>
{};

#if GLM_ARCH & GLM_ARCH_SSE2_BIT

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	template<length_t L, qualifier Q, int E0, int E1, int E2, int E3>
	struct _swizzle_base1<L, float, Q, E0,E1,E2,E3, true> : public _swizzle_base0<float, L>
	{
		GLM_FUNC_QUALIFIER vec<L, float, Q> operator ()()  const
		{
			__m128 data = *reinterpret_cast<__m128 const*>(&this->_buffer);

			vec<L, float, Q> Result;
#			if GLM_ARCH & GLM_ARCH_AVX_BIT
				Result.data = _mm_permute_ps(data, _MM_SHUFFLE(E3, E2, E1, E0));
#			else
				Result.data = _mm_shuffle_ps(data, data, _MM_SHUFFLE(E3, E2, E1, E0));
#			endif
			return Result;
		}
	};

	template<qualifier Q, int E0, int E1, int E2, int E3>
	struct _swizzle_base1<2, float, Q, E0, E1, E2, E3, true> : public _swizzle_base1<2, float, Q, E0, E1, E2, E3, false> {};

	template<qualifier Q, int E0, int E1, int E2, int E3>
	struct _swizzle_base1<2, int, Q, E0, E1, E2, E3, true> : public _swizzle_base1<2, int, Q, E0, E1, E2, E3, false> {};

	template<length_t L, qualifier Q, int E0, int E1, int E2, int E3>
	struct _swizzle_base1<L, int, Q, E0,E1,E2,E3, true> : public _swizzle_base0<int, L>
	{
		GLM_FUNC_QUALIFIER vec<L, int, Q> operator ()()  const
		{
			__m128i data = *reinterpret_cast<__m128i const*>(&this->_buffer);

			vec<L, int, Q> Result;
			Result.data = _mm_shuffle_epi32(data, _MM_SHUFFLE(E3, E2, E1, E0));
			return Result;
		}
	};

	template<length_t L, qualifier Q, int E0, int E1, int E2, int E3>
	struct _swizzle_base1<L, uint, Q, E0,E1,E2,E3, true> : public _swizzle_base0<uint, L>
	{
		GLM_FUNC_QUALIFIER vec<L, uint, Q> operator ()()  const
		{
			__m128i data = *reinterpret_cast<__m128i const*>(&this->_buffer);

			vec<L, uint, Q> Result;
			Result.data = _mm_shuffle_epi32(data, _MM_SHUFFLE(E3, E2, E1, E0));
			return Result;
		}
	};
#	endif// GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR


	template<length_t L, qualifier Q>
	struct compute_vec_add<L, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, float, Q> call(vec<L, float, Q> const& a, vec<L, float, Q> const& b)
		{
			vec<L, float, Q> Result;
			Result.data = _mm_add_ps(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_add<L, int, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, int, Q> call(vec<L, int, Q> const& a, vec<L, int, Q> const& b)
		{
			vec<L, int, Q> Result;
			Result.data = _mm_add_epi32(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_add<L, double, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, double, Q> call(vec<L, double, Q> const& a, vec<L, double, Q> const& b)
		{
			vec<L, double, Q> Result;
#	if (GLM_ARCH & GLM_ARCH_AVX_BIT)
			Result.data = _mm256_add_pd(a.data, b.data);
#else
			Result.data.setv(0, _mm_add_pd(a.data.getv(0), b.data.getv(0)));
			Result.data.setv(1, _mm_add_pd(a.data.getv(1), b.data.getv(1)));
#endif
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_sub<L, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, float, Q> call(vec<L, float, Q> const& a, vec<L, float, Q> const& b)
		{
			vec<L, float, Q> Result;
			Result.data = _mm_sub_ps(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_sub<L, int, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, int, Q> call(vec<L, int, Q> const& a, vec<L, int, Q> const& b)
		{
			vec<L, int, Q> Result;
			Result.data = _mm_sub_epi32(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_sub<L, double, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, double, Q> call(vec<L, double, Q> const& a, vec<L, double, Q> const& b)
		{
			vec<L, double, Q> Result;
#if (GLM_ARCH & GLM_ARCH_AVX_BIT)
			Result.data = _mm256_sub_pd(a.data, b.data);
#else
			Result.data.setv(0,  _mm_sub_pd(a.data.getv(0), b.data.getv(0)));
			Result.data.setv(1,  _mm_sub_pd(a.data.getv(1), b.data.getv(1)));
#endif
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_mul<L, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, float, Q> call(vec<L, float, Q> const& a, vec<L, float, Q> const& b)
		{
			vec<L, float, Q> Result;
			Result.data = _mm_mul_ps(a.data, b.data);
			return Result;
		}
	};


	template<length_t L, qualifier Q>
	struct compute_vec_mul<L, double, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, double, Q> call(vec<L, double, Q> const& a, vec<L, double, Q> const& b)
		{
			vec<L, double, Q> Result;
#if (GLM_ARCH & GLM_ARCH_AVX_BIT)
			Result.data = _mm256_mul_pd(a.data, b.data);
#else
			Result.data.setv(0,  _mm_mul_pd(a.data.getv(0), b.data.getv(0)));
			Result.data.setv(1,  _mm_mul_pd(a.data.getv(1), b.data.getv(1)));
#endif
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_mul<L, int, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, int, Q> call(vec<L, int, Q> const& a, vec<L, int, Q> const& b)
		{
			vec<L, int, Q> Result;
			glm_i32vec4 ia = a.data;
			glm_i32vec4 ib = b.data;
#ifdef __SSE4_1__  // modern CPU - use SSE 4.1
			Result.data = _mm_mullo_epi32(ia, ib);
#else               // old CPU - use SSE 2
			__m128i tmp1 = _mm_mul_epu32(ia, ib); /* mul 2,0*/
			__m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(ia, 4), _mm_srli_si128(ib, 4)); /* mul 3,1 */
			Result.data = _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0))); /* shuffle results to [63..0] and pack */
#endif
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_div<L, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, float, Q> call(vec<L, float, Q> const& a, vec<L, float, Q> const& b)
		{
			vec<L, float, Q> Result;
			Result.data = _mm_div_ps(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_div<L, int, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, int, Q> call(vec<L, int, Q> const& a, vec<L, int, Q> const& b)
		{
#if defined(_MSC_VER) && _MSC_VER >= 1920 //_mm_div_epi32 only defined with VS >= 2019
			vec<L, int, Q> Result;
			Result.data = _mm_div_epi32(a.data, b.data);
			return Result;
#else
			return compute_vec_div<L, int, Q, false>::call(a, b);
#endif
		}
	};


	// note: div on uninitialized w can generate div by 0 exception
	template<qualifier Q>
	struct compute_vec_div<3, int, Q, true>
	{

		GLM_FUNC_QUALIFIER static vec<3, int, Q> call(vec<3, int, Q> const& a, vec<3, int, Q> const& b)
		{
#if defined(_MSC_VER) && _MSC_VER >= 1920 //_mm_div_epi32 only defined with VS >= 2019
			vec<3, int, Q> Result;
			glm_i32vec4 bv = b.data;
			bv = _mm_shuffle_epi32(bv, _MM_SHUFFLE(0, 2, 1, 0));
			Result.data = _mm_div_epi32(a.data, bv);
			return Result;
#else
			return compute_vec_div<3, int, Q, false>::call(a, b);
#endif
		}
	};


	template<length_t L, qualifier Q>
	struct compute_vec_div<L, double, Q, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, double, Q> call(vec<L, double, Q> const& a, vec<L, double, Q> const& b)
		{
			vec<L, double, Q> Result;
#	if GLM_ARCH & GLM_ARCH_AVX_BIT
			Result.data = _mm256_div_pd(a.data, b.data);
#	else
			Result.data.setv(0,  _mm_div_pd(a.data.getv(0), b.data.getv(0)));
			Result.data.setv(1,  _mm_div_pd(a.data.getv(1), b.data.getv(1)));
#	endif
			return Result;
		}
	};

	template<length_t L>
	struct compute_vec_div<L, float, aligned_lowp, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, float, aligned_lowp> call(vec<L, float, aligned_lowp> const& a, vec<L, float, aligned_lowp> const& b)
		{
			vec<L, float, aligned_lowp> Result;
			Result.data = _mm_mul_ps(a.data, _mm_rcp_ps(b.data));
			return Result;
		}
	};

	template<length_t L, typename T, qualifier Q>
	struct compute_vec_and<L, T, Q, -1, 32, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
		{
			vec<L, T, Q> Result;
			Result.data = _mm_and_si128(a.data, b.data);
			return Result;
		}
	};


	template<length_t L, typename T, qualifier Q>
	struct compute_vec_and<L, T, Q, true, 64, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
		{
			vec<L, T, Q> Result;
#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
			Result.data = _mm256_and_si256(a.data, b.data);
#	elif GLM_ARCH & GLM_ARCH_AVX_BIT
			Result.data = _mm256_and_pd(_mm256_castpd256_pd128(a.data), _mm256_castpd256_pd128(b.data));
#	else
			Result.data.setv(0,  _mm_and_si128(a.data.getv(0), b.data.getv(0)));
			Result.data.setv(1,  _mm_and_si128(a.data.getv(1), b.data.getv(1)));
#	endif
			return Result;
		}
	};



	template<length_t L, typename T, qualifier Q>
	struct compute_vec_or<L, T, Q, -1, 32, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
		{
			vec<L, T, Q> Result;
			Result.data = _mm_or_si128(a.data, b.data);
			return Result;
		}
	};


	template<length_t L, typename T, qualifier Q>
	struct compute_vec_or<L, T, Q, true, 64, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
		{
			vec<L, T, Q> Result;
#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
			Result.data = _mm256_or_si256(a.data, b.data);
#	else
			Result.data.setv(0,  _mm_or_si128(a.data.getv(0), b.data.getv(0)));
			Result.data.setv(1,  _mm_or_si128(a.data.getv(1), b.data.getv(1)));
#	endif
			return Result;
		}
	};

	template<length_t L, typename T, qualifier Q>
	struct compute_vec_xor<L, T, Q, true, 32, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
		{
			vec<L, T, Q> Result;
			Result.data = _mm_xor_si128(a.data, b.data);
			return Result;
		}
	};


	template<length_t L, typename T, qualifier Q>
	struct compute_vec_xor<L, T, Q, true, 64, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
		{
			vec<L, T, Q> Result;
#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
			Result.data = _mm256_xor_si256(a.data, b.data);
#	else
			Result.data.setv(0,  _mm_xor_si128(a.data.getv(0), b.data.getv(0)));
			Result.data.setv(1,  _mm_xor_si128(a.data.getv(1), b.data.getv(1)));
#	endif
			return Result;
		}
	};


	//template<typename T, qualifier Q>
	//struct compute_vec_shift_left<3, T, Q, -1, 32, true>
	//{
	//	GLM_FUNC_QUALIFIER static vec<3, T, Q> call(vec<3, T, Q> const& a, vec<3, T, Q> const& b)
	//	{
	//		vec<3, T, Q> Result;
	//		__m128 v2 = _mm_castsi128_ps(b.data);
	//		v2 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 0)); // note: shift is done with xmm[3] that correspond vec w that doesn't exist on vect3
	//		_mm_set1_epi64x(_w, _z, _y, _x);
	//		__m128i vr = _mm_sll_epi32(a.data, _mm_castps_si128(v2));
	//		Result.data = vr;
	//		return Result;
	//	}
	//};

	//template<length_t L, typename T, qualifier Q>
	//struct compute_vec_shift_left<L, T, Q, -1, 32, true>
	//{
	//	GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
	//	{
	//		vec<L, T, Q> Result;
	//		Result.data = _mm_sll_epi32(a.data, b.data);
	//		return Result;
	//	}
	//};


//	template<length_t L, typename T, qualifier Q>
//	struct compute_vec_shift_left<L, T, Q, true, 64, true>
//	{
//		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
//		{
//			vec<L, T, Q> Result;
//#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
//			Result.data = _mm256_sll_epi64(a.data, b.data);
//#	else
//			Result.data.setv(0,  _mm_sll_epi64(a.data.getv(0), b.data.getv(0)));
//			Result.data.setv(1,  _mm_sll_epi64(a.data.getv(1), b.data.getv(1)));
//#	endif
//			return Result;
//		}
//	};


//	template<length_t L, typename T, qualifier Q>
//	struct compute_vec_shift_right<L, T, Q, -1, 32, true>
//	{
//		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
//		{
//			vec<L, T, Q> Result;
//			Result.data = _mm_srl_epi32(a.data, b.data);
//			return Result;
//		}
//	};
//
//	template<length_t L, typename T, qualifier Q>
//	struct compute_vec_shift_right<L, T, Q, true, 64, true>
//	{
//		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
//		{
//			vec<L, T, Q> Result;
//#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
//			Result.data = _mm256_srl_epi64(a.data, b.data);
//#	else
//			Result.data.setv(0,  _mm_srl_epi64(a.data.getv(0), b.data.getv(0)));
//			Result.data.setv(1,  _mm_srl_epi64(a.data.getv(1), b.data.getv(1)));
//#	endif
//			return Result;
//		}
//	};

	template<length_t L, typename T, qualifier Q>
	struct compute_vec_bitwise_not<L, T, Q, true, 32, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& v)
		{
			vec<L, T, Q> Result;
			Result.data = _mm_xor_si128(v.data, _mm_set1_epi32(-1));
			return Result;
		}
	};


	template<length_t L, typename T, qualifier Q>
	struct compute_vec_bitwise_not<L, T, Q, true, 64, true>
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& v)
		{
			vec<L, T, Q> Result;
#	if GLM_ARCH & GLM_ARCH_AVX2_BIT
			Result.data = _mm256_xor_si256(v.data, _mm256_set1_epi32(-1));
#	else
			Result.data.setv(0,  _mm_xor_si128(v.data.getv(0), _mm_set1_epi32(-1)));
			Result.data.setv(1,  _mm_xor_si128(v.data.getv(1), _mm_set1_epi32(-1)));
#	endif
			return Result;
		}
	};


	template<length_t L, qualifier Q>
	struct compute_vec_equal<L, float, Q, false, 32, true>
	{
		GLM_FUNC_QUALIFIER static bool call(vec<L, float, Q> const& v1, vec<L, float, Q> const& v2)
		{
			return _mm_movemask_ps(_mm_cmpneq_ps(v1.data, v2.data)) == 0;
		}
	};

#	if GLM_ARCH & GLM_ARCH_SSE41_BIT
	template<length_t L, qualifier Q>
	struct compute_vec_equal<L, int, Q, true, 32, true>
	{
		GLM_FUNC_QUALIFIER static bool call(vec<L, int, Q> const& v1, vec<L, int, Q> const& v2)
		{
			//return _mm_movemask_epi8(_mm_cmpeq_epi32(v1.data, v2.data)) != 0;
			__m128i neq = _mm_xor_si128(v1.data, v2.data);
			return _mm_test_all_zeros(neq, neq) == 0;
		}
	};
#	endif



	template<length_t L, qualifier Q>
	struct compute_vec_nequal<L, float, Q, false, 32, true>
	{
		GLM_FUNC_QUALIFIER static bool call(vec<L, float, Q> const& v1, vec<L, float, Q> const& v2)
		{
			return _mm_movemask_ps(_mm_cmpneq_ps(v1.data, v2.data)) != 0;
		}
	};

#	if GLM_ARCH & GLM_ARCH_SSE41_BIT
	template<length_t L, qualifier Q>
	struct compute_vec_nequal<L, int, Q, -1, 32, true>
	{
		GLM_FUNC_QUALIFIER static bool call(vec<L, int, Q> const& v1, vec<L, int, Q> const& v2)
		{
			//return _mm_movemask_epi8(_mm_cmpneq_epi32(v1.data, v2.data)) != 0;
			__m128i neq = _mm_xor_si128(v1.data, v2.data);
			int v = _mm_test_all_zeros(neq, neq);
			return  v != 1;
		}
	};
	template<length_t L, qualifier Q>
	struct compute_vec_nequal<L, unsigned int, Q, -1, 32, true>
	{
		GLM_FUNC_QUALIFIER static bool call(vec<L, unsigned int, Q> const& v1, vec<L, unsigned int, Q> const& v2)
		{
			//return _mm_movemask_epi8(_mm_cmpneq_epi32(v1.data, v2.data)) != 0;
			__m128i neq = _mm_xor_si128(v1.data, v2.data);
			return _mm_test_all_zeros(neq, neq) != 1;
		}
	};
#	else


	template<length_t L, qualifier Q>
	struct compute_vec_nequal<L, unsigned int, Q, -1, 32, true>
	{
		GLM_FUNC_QUALIFIER static bool call(vec<L, unsigned int, Q> const& v1, vec<L, unsigned int, Q> const& v2)
		{
			return compute_vec_nequal<L, unsigned int, Q, true, 32, false>::call(v1, v2);
		}
	};

#	endif




}//namespace detail


#define CTOR_FLOAT(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(float _s) :\
	data(_mm_set1_ps(_s))\
	{}

#if GLM_ARCH & GLM_ARCH_AVX_BIT
#	define CTOR_DOUBLE(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, double, Q>::vec(double _s) :\
		data(_mm256_set1_pd(_s)){}

#define CTOR_DOUBLE4(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, double, Q>::vec(double _x, double _y, double _z, double _w):\
		data(_mm256_set_pd(_w, _z, _y, _x)) {}

#define CTOR_DOUBLE3(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, double, Q>::vec(double _x, double _y, double _z):\
		data(_mm256_set_pd(_z, _z, _y, _x)) {}

#	define CTOR_INT64(L, Q)\
	template<>\
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, detail::int64, Q>::vec(detail::int64 _s) :\
		data(_mm256_set1_epi64x(_s)){}

#define CTOR_DOUBLE_COPY3(L, Q)\
	template<>\
	template<qualifier P>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, double, Q>::vec(vec<3, double, P> const& v) :\
		data(_mm256_setr_pd(v.x, v.y, v.z, v.z)){}

#else
#	define CTOR_DOUBLE(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, double, Q>::vec(double _v) \
	{\
		data.setv(0, _mm_set1_pd(_v)); \
		data.setv(1, _mm_set1_pd(_v)); \
	}

#define CTOR_DOUBLE4(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, double, Q>::vec(double _x, double _y, double _z, double _w)\
	{\
		data.setv(0, _mm_setr_pd(_x, _y)); \
		data.setv(1, _mm_setr_pd(_z, _w)); \
	}

#define CTOR_DOUBLE3(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, double, Q>::vec(double _x, double _y, double _z)\
	{\
		data.setv(0, _mm_setr_pd(_x, _y)); \
		data.setv(1, _mm_setr_pd(_z, _z)); \
	}

#	define CTOR_INT64(L, Q)\
	template<>\
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, detail::int64, Q>::vec(detail::int64 _s) :\
		data(_mm256_set1_epi64x(_s)){}

#define CTOR_DOUBLE_COPY3(L, Q)\
	template<>\
	template<qualifier P>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, double, Q>::vec(vec<3, double, P> const& v)\
	{\
		data.setv(0, _mm_setr_pd(v.x, v.y));\
		data.setv(1, _mm_setr_pd(v.z, 1.0));\
	}

#endif //GLM_ARCH & GLM_ARCH_AVX_BIT

#define CTOR_INT(L, Q)\
	template<>\
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, int, Q>::vec(int _s) :\
		data(_mm_set1_epi32(_s))\
		{}

#define CTOR_FLOAT4(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(float _x, float _y, float _z, float _w) :\
		data(_mm_set_ps(_w, _z, _y, _x))\
		{}

#define CTOR_FLOAT3(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(float _x, float _y, float _z) :\
	data(_mm_set_ps(_z, _z, _y, _x)){}


#define CTOR_INT4(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, int, Q>::vec(int _x, int _y, int _z, int _w) :\
	data(_mm_set_epi32(_w, _z, _y, _x)){}

#define CTOR_INT3(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, int, Q>::vec(int _x, int _y, int _z) :\
	data(_mm_set_epi32(_z, _z, _y, _x)){}

#define CTOR_VECF_INT4(L, Q)\
	template<>\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(int _x, int _y, int _z, int _w) :\
	data(_mm_cvtepi32_ps(_mm_set_epi32(_w, _z, _y, _x)))\
	{}

#define CTOR_VECF_INT3(L, Q)\
	template<>\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(int _x, int _y, int _z) :\
	data(_mm_cvtepi32_ps(_mm_set_epi32(_z, _z, _y, _x)))\
	{}

#define CTOR_DEFAULT(L, Q)\
	template<>\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec() :\
	data(_mm_setzero_ps())\
	{}

#define CTOR_FLOAT_COPY3(L, Q)\
	template<>\
	template<qualifier P>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, float, Q>::vec(vec<3, float, P> const& v)\
		:data(_mm_set_ps(v.z, v.z, v.y, v.x))\
	{}



}//namespace glm

#endif//GLM_ARCH & GLM_ARCH_SSE2_BIT

#if GLM_ARCH & GLM_ARCH_NEON_BIT

#if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
// the functions below needs to be properly implemented, use unoptimized function fro now.

template<length_t L, qualifier Q, int E0, int E1, int E2, int E3>
struct _swizzle_base1<L, float, Q, E0, E1, E2, E3, true> : public _swizzle_base1<L, float, Q, E0, E1, E2, E3, false>{}; 

template<length_t L, qualifier Q, int E0, int E1, int E2, int E3>
struct _swizzle_base1<L, int, Q, E0, E1, E2, E3, true> : public _swizzle_base1<L, int, Q, E0, E1, E2, E3, false> {};

template<length_t L, qualifier Q, int E0, int E1, int E2, int E3>
struct _swizzle_base1<L, uint, Q, E0, E1, E2, E3, true> : public _swizzle_base1<L, uint, Q, E0, E1, E2, E3, false> {};

#	endif// GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR


	template<length_t L, qualifier Q>
	struct compute_vec_add<L, float, Q, true>
	{
		GLM_FUNC_QUALIFIER static
		vec<L, float, Q>
		call(vec<L, float, Q> const& a, vec<L, float, Q> const& b)
		{
			vec<L, float, Q> Result;
			Result.data = vaddq_f32(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_add<L, uint, Q, true>
	{
		GLM_FUNC_QUALIFIER static
		vec<L, uint, Q>
		call(vec<L, uint, Q> const& a, vec<L, uint, Q> const& b)
		{
			vec<L, uint, Q> Result;
			Result.data = vaddq_u32(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_add<L, int, Q, true>
	{
		static
		vec<L, int, Q>
		call(vec<L, int, Q> const& a, vec<L, int, Q> const& b)
		{
			vec<L, int, Q> Result;
			Result.data = vaddq_s32(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_sub<L, float, Q, true>
	{
		static vec<L, float, Q> call(vec<L, float, Q> const& a, vec<L, float, Q> const& b)
		{
			vec<L, float, Q> Result;
			Result.data = vsubq_f32(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_sub<L, uint, Q, true>
	{
		static vec<L, uint, Q> call(vec<L, uint, Q> const& a, vec<L, uint, Q> const& b)
		{
			vec<L, uint, Q> Result;
			Result.data = vsubq_u32(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_sub<L, int, Q, true>
	{
		static vec<L, int, Q> call(vec<L, int, Q> const& a, vec<L, int, Q> const& b)
		{
			vec<L, int, Q> Result;
			Result.data = vsubq_s32(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_mul<L, float, Q, true>
	{
		static vec<L, float, Q> call(vec<L, float, Q> const& a, vec<L, float, Q> const& b)
		{
			vec<L, float, Q> Result;
			Result.data = vmulq_f32(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_mul<L, uint, Q, true>
	{
		static vec<L, uint, Q> call(vec<L, uint, Q> const& a, vec<L, uint, Q> const& b)
		{
			vec<L, uint, Q> Result;
			Result.data = vmulq_u32(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_mul<L, int, Q, true>
	{
		static vec<L, int, Q> call(vec<L, int, Q> const& a, vec<L, int, Q> const& b)
		{
			vec<L, int, Q> Result;
			Result.data = vmulq_s32(a.data, b.data);
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_div<L, float, Q, true>
	{
		static vec<L, float, Q> call(vec<L, float, Q> const& a, vec<L, float, Q> const& b)
		{
			vec<L, float, Q> Result;
#if GLM_ARCH & GLM_ARCH_ARMV8_BIT
			Result.data = vdivq_f32(a.data, b.data);
#else
			/* Arm assembler reference:
			 *
			 * The Newton-Raphson iteration: x[n+1] = x[n] * (2 - d * x[n])
			 * converges to (1/d) if x0 is the result of VRECPE applied to d.
			 *
			 * Note: The precision usually improves with two interactions, but more than two iterations are not helpful. */
			float32x4_t x = vrecpeq_f32(b.data);
			x = vmulq_f32(vrecpsq_f32(b.data, x), x);
			x = vmulq_f32(vrecpsq_f32(b.data, x), x);
			Result.data = vmulq_f32(a.data, x);
#endif
			return Result;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_equal<L, float, Q, false, 32, true>
	{
		static bool call(vec<L, float, Q> const& v1, vec<L, float, Q> const& v2)
		{
			uint32x4_t cmp = vceqq_f32(v1.data, v2.data);
#if GLM_ARCH & GLM_ARCH_ARMV8_BIT
			cmp = vpminq_u32(cmp, cmp);
			cmp = vpminq_u32(cmp, cmp);
			uint32_t r = cmp[0];
#else
			uint32x2_t cmpx2 = vpmin_u32(vget_low_u32(cmp), vget_high_u32(cmp));
			cmpx2 = vpmin_u32(cmpx2, cmpx2);
			uint32_t r = cmpx2[0];
#endif
			return r == ~0u;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_equal<L, uint, Q, false, 32, true>
	{
		static bool call(vec<L, uint, Q> const& v1, vec<L, uint, Q> const& v2)
		{
			uint32x4_t cmp = vceqq_u32(v1.data, v2.data);
#if GLM_ARCH & GLM_ARCH_ARMV8_BIT
			cmp = vpminq_u32(cmp, cmp);
			cmp = vpminq_u32(cmp, cmp);
			uint32_t r = cmp[0];
#else
			uint32x2_t cmpx2 = vpmin_u32(vget_low_u32(cmp), vget_high_u32(cmp));
			cmpx2 = vpmin_u32(cmpx2, cmpx2);
			uint32_t r = cmpx2[0];
#endif
			return r == ~0u;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_equal<L, int, Q, false, 32, true>
	{
		static bool call(vec<L, int, Q> const& v1, vec<L, int, Q> const& v2)
		{
			uint32x4_t cmp = vceqq_s32(v1.data, v2.data);
#if GLM_ARCH & GLM_ARCH_ARMV8_BIT
			cmp = vpminq_u32(cmp, cmp);
			cmp = vpminq_u32(cmp, cmp);
			uint32_t r = cmp[0];
#else
			uint32x2_t cmpx2 = vpmin_u32(vget_low_u32(cmp), vget_high_u32(cmp));
			cmpx2 = vpmin_u32(cmpx2, cmpx2);
			uint32_t r = cmpx2[0];
#endif
			return r == ~0u;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_nequal<L, float, Q, false, 32, true>
	{
		static bool call(vec<L, float, Q> const& v1, vec<L, float, Q> const& v2)
		{
			return !compute_vec_equal<float, Q, false, 32, true>::call(v1, v2);
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_nequal<L, uint, Q, false, 32, true>
	{
		static bool call(vec<L, uint, Q> const& v1, vec<L, uint, Q> const& v2)
		{
			return !compute_vec_equal<uint, Q, false, 32, true>::call(v1, v2);
		}
	};

	template<length_t L, qualifier Q>
	struct compute_vec_nequal<L, int, Q, false, 32, true>
	{
		static bool call(vec<L, int, Q> const& v1, vec<L, int, Q> const& v2)
		{
			return !compute_vec_equal<int, Q, false, 32, true>::call(v1, v2);
		}
	};


#if !GLM_CONFIG_XYZW_ONLY

#define CTOR_FLOAT(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(float _s) :\
		data(vdupq_n_f32(_s))\
	{}

#define CTOR_INT(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, int, Q>::vec(int _s) :\
		data(vdupq_n_s32(_s))\
	{}

#define CTOR_UINT(L, Q)\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, uint, Q>::vec(uint _s) :\
		data(vdupq_n_u32(_s))\
	{}

#define CTOR_VECF_INT4(L, Q)\
	template<>\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(int _x, int _y, int _z, int _w) :\
		data(vcvtq_f32_s32(vec<L, int, Q>(_x, _y, _z, _w).data))\
	{}

#define CTOR_VECF_UINT4(L, Q)\
	template<>\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(uint _x, uint _y, uint _z, uint _w) :\
		data(vcvtq_f32_u32(vec<L, uint, Q>(_x, _y, _z, _w).data))\
	{}

#define CTOR_VECF_INT3(L, Q)\
	template<>\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(int _x, int _y, int _z) :\
		data(vcvtq_f32_s32(vec<L, int, Q>(_x, _y, _z).data))\
	{}

#define CTOR_VECF_UINT4(L, Q)\
	template<>\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(uint _x, uint _y, uint _z, uint _w) :\
		data(vcvtq_f32_u32(vec<L, uint, Q>(_x, _y, _z, _w).data))\
	{}

#define CTOR_VECF_UINT3(L, Q)\
	template<>\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(uint _x, uint _y, uint _z) :\
		data(vcvtq_f32_u32(vec<L, uint, Q>(_x, _y, _z).data))\
	{}


#define CTOR_VECF_VECF(L, Q)\
	template<>\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(const vec<L, float, Q>& rhs) :\
		data(rhs.data)\
	{}

#define CTOR_VECF_VECI(L, Q)\
	template<>\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(const vec<L, int, Q>& rhs) :\
		data(vcvtq_f32_s32(rhs.data))\
	{}

#define CTOR_VECF_VECU(L, Q)\
	template<>\
	template<>\
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, float, Q>::vec(const vec<L, uint, Q>& rhs) :\
		data(vcvtq_f32_u32(rhs.data))\
	{}


#endif


}//namespace detail

}//namespace glm

#endif
