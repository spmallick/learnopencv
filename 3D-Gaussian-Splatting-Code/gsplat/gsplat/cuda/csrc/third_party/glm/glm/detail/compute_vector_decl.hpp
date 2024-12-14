
#pragma once
#include <functional>
#include "_vectorize.hpp"

namespace glm {
	namespace detail
	{
		template<length_t L, typename T, qualifier Q, bool UseSimd>
		struct compute_vec_add {};

		template<length_t L, typename T, qualifier Q, bool UseSimd>
		struct compute_vec_sub {};

		template<length_t L, typename T, qualifier Q, bool UseSimd>
		struct compute_vec_mul {};

		template<length_t L, typename T, qualifier Q, bool UseSimd>
		struct compute_vec_div {};

		template<length_t L, typename T, qualifier Q, bool UseSimd>
		struct compute_vec_mod {};

		template<length_t L, typename T, qualifier Q, bool UseSimd>
		struct compute_splat {};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size, bool UseSimd>
		struct compute_vec_and {};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size, bool UseSimd>
		struct compute_vec_or {};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size, bool UseSimd>
		struct compute_vec_xor {};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size, bool UseSimd>
		struct compute_vec_shift_left {};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size, bool UseSimd>
		struct compute_vec_shift_right {};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size, bool UseSimd>
		struct compute_vec_equal {};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size, bool UseSimd>
		struct compute_vec_nequal {};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size, bool UseSimd>
		struct compute_vec_bitwise_not {};

		template<length_t L, typename T, qualifier Q>
		struct compute_vec_add<L, T, Q, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
			{
				return detail::functor2<vec, L, T, Q>::call(std::plus<T>(), a, b);
			}
		};

		template<length_t L, typename T, qualifier Q>
		struct compute_vec_sub<L, T, Q, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
			{
				return detail::functor2<vec, L, T, Q>::call(std::minus<T>(), a, b);
			}
		};

		template<length_t L, typename T, qualifier Q>
		struct compute_vec_mul<L, T, Q, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
			{
				return detail::functor2<vec, L, T, Q>::call(std::multiplies<T>(), a, b);
			}
		};

		template<length_t L, typename T, qualifier Q>
		struct compute_vec_div<L, T, Q, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
			{
				return detail::functor2<vec, L, T, Q>::call(std::divides<T>(), a, b);
			}
		};

		template<length_t L, typename T, qualifier Q>
		struct compute_vec_mod<L, T, Q, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
			{
				return detail::functor2<vec, L, T, Q>::call(std::modulus<T>(), a, b);
			}
		};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
		struct compute_vec_and<L, T, Q, IsInt, Size, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
			{
				vec<L, T, Q> v(a);
				for (length_t i = 0; i < L; ++i)
					v[i] &= static_cast<T>(b[i]);
				return v;
			}
		};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
		struct compute_vec_or<L, T, Q, IsInt, Size, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
			{
				vec<L, T, Q> v(a);
				for (length_t i = 0; i < L; ++i)
					v[i] |= static_cast<T>(b[i]);
				return v;
			}
		};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
		struct compute_vec_xor<L, T, Q, IsInt, Size, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
			{
				vec<L, T, Q> v(a);
				for (length_t i = 0; i < L; ++i)
					v[i] ^= static_cast<T>(b[i]);
				return v;
			}
		};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
		struct compute_vec_shift_left<L, T, Q, IsInt, Size, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
			{
				vec<L, T, Q> v(a);
				for (length_t i = 0; i < L; ++i)
					v[i] <<= static_cast<T>(b[i]);
				return v;
			}
		};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
		struct compute_vec_shift_right<L, T, Q, IsInt, Size, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
			{
				vec<L, T, Q> v(a);
				for (length_t i = 0; i < L; ++i)
					v[i] >>= static_cast<T>(b[i]);
				return v;
			}
		};
		
		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
		struct compute_vec_equal<L, T, Q, IsInt, Size, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static bool call(vec<L, T, Q> const& v1, vec<L, T, Q> const& v2)
			{
				bool b = true;
				for (length_t i = 0; b && i < L; ++i)
					b = detail::compute_equal<T, std::numeric_limits<T>::is_iec559>::call(v1[i], v2[i]);
				return b;
			}
		};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
		struct compute_vec_nequal<L, T, Q, IsInt, Size, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static bool call(vec<4, T, Q> const& v1, vec<4, T, Q> const& v2)
			{
				return !compute_vec_equal<L, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(v1, v2);
			}
		};

		template<length_t L, typename T, qualifier Q, int IsInt, std::size_t Size>
		struct compute_vec_bitwise_not<L, T, Q, IsInt, Size, false>
		{
			GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<L, T, Q> call(vec<L, T, Q> const& a)
			{
				vec<L, T, Q> v(a);
				for (length_t i = 0; i < L; ++i)
					v[i] = ~v[i];
				return v;
			}
		};

	}
}
