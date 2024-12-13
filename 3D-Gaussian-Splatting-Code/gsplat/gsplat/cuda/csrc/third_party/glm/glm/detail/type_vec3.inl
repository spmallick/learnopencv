/// @ref core

#include "compute_vector_relational.hpp"
#include "compute_vector_decl.hpp"

namespace glm
{
	// -- Implicit basic constructors --

#	if GLM_CONFIG_DEFAULTED_DEFAULT_CTOR == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_DEFAULTED_DEFAULT_CTOR_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec()
#			if GLM_CONFIG_CTOR_INIT != GLM_CTOR_INIT_DISABLE
				: x(0), y(0), z(0)
#			endif
		{}
#	endif

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_DEFAULTED_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<3, T, Q> const& v)
			: x(v.x), y(v.y), z(v.z)
		{}
#	endif

	template<typename T, qualifier Q>
	template<qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<3, T, P> const& v)
		: x(v.x), y(v.y), z(v.z)
	{}

	// -- Explicit basic constructors --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(T scalar)
		: x(scalar), y(scalar), z(scalar)
	{}

	template <typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(T _x, T _y, T _z)
		: x(_x), y(_y), z(_z)
	{}

	// -- Conversion scalar constructors --

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<1, U, P> const& v)
		: x(static_cast<T>(v.x))
		, y(static_cast<T>(v.x))
		, z(static_cast<T>(v.x))
	{}

	template<typename T, qualifier Q>
	template<typename X, typename Y, typename Z>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(X _x, Y _y, Z _z)
		: x(static_cast<T>(_x))
		, y(static_cast<T>(_y))
		, z(static_cast<T>(_z))
	{}

	template<typename T, qualifier Q>
	template<typename X, typename Y, typename Z>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<1, X, Q> const& _x, Y _y, Z _z)
		: x(static_cast<T>(_x.x))
		, y(static_cast<T>(_y))
		, z(static_cast<T>(_z))
	{}

	template<typename T, qualifier Q>
	template<typename X, typename Y, typename Z>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(X _x, vec<1, Y, Q> const& _y, Z _z)
		: x(static_cast<T>(_x))
		, y(static_cast<T>(_y.x))
		, z(static_cast<T>(_z))
	{}

	template<typename T, qualifier Q>
	template<typename X, typename Y, typename Z>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<1, X, Q> const& _x, vec<1, Y, Q> const& _y, Z _z)
		: x(static_cast<T>(_x.x))
		, y(static_cast<T>(_y.x))
		, z(static_cast<T>(_z))
	{}

	template<typename T, qualifier Q>
	template<typename X, typename Y, typename Z>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(X _x, Y _y, vec<1, Z, Q> const& _z)
		: x(static_cast<T>(_x))
		, y(static_cast<T>(_y))
		, z(static_cast<T>(_z.x))
	{}

	template<typename T, qualifier Q>
	template<typename X, typename Y, typename Z>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<1, X, Q> const& _x, Y _y, vec<1, Z, Q> const& _z)
		: x(static_cast<T>(_x.x))
		, y(static_cast<T>(_y))
		, z(static_cast<T>(_z.x))
	{}

	template<typename T, qualifier Q>
	template<typename X, typename Y, typename Z>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(X _x, vec<1, Y, Q> const& _y, vec<1, Z, Q> const& _z)
		: x(static_cast<T>(_x))
		, y(static_cast<T>(_y.x))
		, z(static_cast<T>(_z.x))
	{}

	template<typename T, qualifier Q>
	template<typename X, typename Y, typename Z>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<1, X, Q> const& _x, vec<1, Y, Q> const& _y, vec<1, Z, Q> const& _z)
		: x(static_cast<T>(_x.x))
		, y(static_cast<T>(_y.x))
		, z(static_cast<T>(_z.x))
	{}

	// -- Conversion vector constructors --

	template<typename T, qualifier Q>
	template<typename A, typename B, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<2, A, P> const& _xy, B _z)
		: x(static_cast<T>(_xy.x))
		, y(static_cast<T>(_xy.y))
		, z(static_cast<T>(_z))
	{}

	template<typename T, qualifier Q>
	template<typename A, typename B, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<2, A, P> const& _xy, vec<1, B, P> const& _z)
		: x(static_cast<T>(_xy.x))
		, y(static_cast<T>(_xy.y))
		, z(static_cast<T>(_z.x))
	{}

	template<typename T, qualifier Q>
	template<typename A, typename B, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(A _x, vec<2, B, P> const& _yz)
		: x(static_cast<T>(_x))
		, y(static_cast<T>(_yz.x))
		, z(static_cast<T>(_yz.y))
	{}

	template<typename T, qualifier Q>
	template<typename A, typename B, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<1, A, P> const& _x, vec<2, B, P> const& _yz)
		: x(static_cast<T>(_x.x))
		, y(static_cast<T>(_yz.x))
		, z(static_cast<T>(_yz.y))
	{}

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<3, U, P> const& v)
		: x(static_cast<T>(v.x))
		, y(static_cast<T>(v.y))
		, z(static_cast<T>(v.z))
	{}

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>::vec(vec<4, U, P> const& v)
		: x(static_cast<T>(v.x))
		, y(static_cast<T>(v.y))
		, z(static_cast<T>(v.z))
	{}

	// -- Component accesses --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T & vec<3, T, Q>::operator[](typename vec<3, T, Q>::length_type i)
	{
		GLM_ASSERT_LENGTH(i, this->length());
		switch(i)
		{
		default:
			case 0:
		return x;
			case 1:
		return y;
			case 2:
		return z;
		}
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T const& vec<3, T, Q>::operator[](typename vec<3, T, Q>::length_type i) const
	{
		GLM_ASSERT_LENGTH(i, this->length());
		switch(i)
		{
		default:
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		}
	}

	// -- Unary arithmetic operators --

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_DEFAULTED_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>& vec<3, T, Q>::operator=(vec<3, T, Q> const& v)
		{
			this->x = v.x;
			this->y = v.y;
			this->z = v.z;
			return *this;
		}
#	endif

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q>& vec<3, T, Q>::operator=(vec<3, U, Q> const& v)
	{
		this->x = static_cast<T>(v.x);
		this->y = static_cast<T>(v.y);
		this->z = static_cast<T>(v.z);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator+=(U scalar)
	{
		return (*this = detail::compute_vec_add<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(scalar)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator+=(vec<1, U, Q> const& v)
	{
		return (*this = detail::compute_vec_add<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<1, T, Q>(v.x)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator+=(vec<3, U, Q> const& v)
	{
		return (*this = detail::compute_vec_add<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator-=(U scalar)
	{
		return (*this = detail::compute_vec_sub<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(scalar)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator-=(vec<1, U, Q> const& v)
	{
		return (*this = detail::compute_vec_sub<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<1, T, Q>(v.x)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator-=(vec<3, U, Q> const& v)
	{
		return (*this = detail::compute_vec_sub<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator*=(U scalar)
	{
		return (*this = detail::compute_vec_mul<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(static_cast<T>(scalar))));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator*=(vec<1, U, Q> const& v)
	{
		return (*this = detail::compute_vec_mul<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v.x)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator*=(vec<3, U, Q> const& v)
	{
		return (*this = detail::compute_vec_mul<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator/=(U v)
	{
		return (*this = detail::compute_vec_div<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator/=(vec<1, U, Q> const& v)
	{
		return (*this = detail::compute_vec_div<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v.x)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator/=(vec<3, U, Q> const& v)
	{
		return (*this = detail::compute_vec_div<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	// -- Increment and decrement operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator++()
	{
		++this->x;
		++this->y;
		++this->z;
		return *this;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator--()
	{
		--this->x;
		--this->y;
		--this->z;
		return *this;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> vec<3, T, Q>::operator++(int)
	{
		vec<3, T, Q> Result(*this);
		++*this;
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> vec<3, T, Q>::operator--(int)
	{
		vec<3, T, Q> Result(*this);
		--*this;
		return Result;
	}

	// -- Unary bit operators --

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator%=(U scalar)
	{
		return (*this = detail::compute_vec_mod<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(scalar)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator%=(vec<1, U, Q> const& v)
	{
		return (*this = detail::compute_vec_mod<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator%=(vec<3, U, Q> const& v)
	{
		return (*this = detail::compute_vec_mod<3, T, Q, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator&=(U scalar)
	{
		return (*this = detail::compute_vec_and<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(scalar)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator&=(vec<1, U, Q> const& v)
	{
		return (*this = detail::compute_vec_and<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator&=(vec<3, U, Q> const& v)
	{
		return (*this = detail::compute_vec_and<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator|=(U scalar)
	{
		return (*this = detail::compute_vec_or<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(scalar)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator|=(vec<1, U, Q> const& v)
	{
		return (*this = detail::compute_vec_or<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator|=(vec<3, U, Q> const& v)
	{
		return (*this = detail::compute_vec_or<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator^=(U scalar)
	{
		return (*this = detail::compute_vec_xor<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(scalar)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator^=(vec<1, U, Q> const& v)
	{
		return (*this = detail::compute_vec_xor<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v.x)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator^=(vec<3, U, Q> const& v)
	{
		return (*this = detail::compute_vec_xor<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator<<=(U scalar)
	{
		return (*this = detail::compute_vec_shift_left<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(scalar)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator<<=(vec<1, U, Q> const& v)
	{
		return (*this = detail::compute_vec_shift_left<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<1, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator<<=(vec<3, U, Q> const& v)
	{
		return (*this = detail::compute_vec_shift_left<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator>>=(U scalar)
	{
		return (*this = detail::compute_vec_shift_right<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(scalar)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator>>=(vec<1, U, Q> const& v)
	{
		return (*this = detail::compute_vec_shift_right<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> & vec<3, T, Q>::operator>>=(vec<3, U, Q> const& v)
	{
		return (*this = detail::compute_vec_shift_right<3, T, Q, detail::is_int<T>::value, sizeof(T) * 8, detail::is_aligned<Q>::value>::call(*this, vec<3, T, Q>(v)));

	}

	// -- Unary arithmetic operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator+(vec<3, T, Q> const& v)
	{
		return v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator-(vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(0) -= v;
	}

	// -- Binary arithmetic operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator+(vec<3, T, Q> const& v, T scalar)
	{
		return vec<3, T, Q>(v) += scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator+(vec<3, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<3, T, Q>(v1) += v2;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator+(T scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(v) += scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator+(vec<1, T, Q> const& scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar) += v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator+(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return vec<3, T, Q>(v1) += v2;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator-(vec<3, T, Q> const& v, T scalar)
	{
		return vec<3, T, Q>(v) -= scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator-(vec<3, T, Q> const& v, vec<1, T, Q> const& scalar)
	{
		return vec<3, T, Q>(v) -= scalar.x;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator-(T scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar) -= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator-(vec<1, T, Q> const& scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar) -= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator-(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return vec<3, T, Q>(v1) -= v2;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator*(vec<3, T, Q> const& v, T scalar)
	{
		return vec<3, T, Q>(v) *= scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator*(vec<3, T, Q> const& v, vec<1, T, Q> const& scalar)
	{
		return vec<3, T, Q>(v) *= scalar.x;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator*(T scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(v) *= scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator*(vec<1, T, Q> const& scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(v) *= scalar.x;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator*(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return vec<3, T, Q>(v1) *= v2;

	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator/(vec<3, T, Q> const& v, T scalar)
	{
		return vec<3, T, Q>(v) /= scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator/(vec<3, T, Q> const& v, vec<1, T, Q> const& scalar)
	{
		return vec<3, T, Q>(v) /= scalar.x;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator/(T scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar) /= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator/(vec<1, T, Q> const& scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar) /= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator/(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return vec<3, T, Q>(v1) /= v2;
	}

	// -- Binary bit operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator%(vec<3, T, Q> const& v, T scalar)
	{
		return vec<3, T, Q>(v) %= scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator%(vec<3, T, Q> const& v, vec<1, T, Q> const& scalar)
	{
		return vec<3, T, Q>(v) %= scalar.x;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator%(T scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar) %= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator%(vec<1, T, Q> const& scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar.x) %= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator%(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return vec<3, T, Q>(v1) %= v2;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator&(vec<3, T, Q> const& v, T scalar)
	{
		return vec<3, T, Q>(v) &= scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator&(vec<3, T, Q> const& v, vec<1, T, Q> const& scalar)
	{
		return vec<3, T, Q>(v) &= scalar.x;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator&(T scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar) &= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator&(vec<1, T, Q> const& scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar.x) &= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator&(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return vec<3, T, Q>(v1) &= v2;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator|(vec<3, T, Q> const& v, T scalar)
	{
		return vec<3, T, Q>(v) |= scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator|(vec<3, T, Q> const& v, vec<1, T, Q> const& scalar)
	{
		return vec<3, T, Q>(v) |= scalar.x;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator|(T scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar) |= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator|(vec<1, T, Q> const& scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar.x) |= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator|(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return vec<3, T, Q>(v1) |= v2;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator^(vec<3, T, Q> const& v, T scalar)
	{
		return vec<3, T, Q>(v) ^= scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator^(vec<3, T, Q> const& v, vec<1, T, Q> const& scalar)
	{
		return vec<3, T, Q>(v) ^= scalar.x;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator^(T scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar) ^= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator^(vec<1, T, Q> const& scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar.x) ^= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator^(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return vec<3, T, Q>(v1) ^= v2;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator<<(vec<3, T, Q> const& v, T scalar)
	{
		return vec<3, T, Q>(v) <<= scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator<<(vec<3, T, Q> const& v, vec<1, T, Q> const& scalar)
	{
		return vec<3, T, Q>(v) <<= scalar.x;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator<<(T scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar) << v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator<<(vec<1, T, Q> const& scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar.x) << v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator<<(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return vec<3, T, Q>(v1) <<= v2;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator>>(vec<3, T, Q> const& v, T scalar)
	{
		return vec<3, T, Q>(v) >>= scalar;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator>>(vec<3, T, Q> const& v, vec<1, T, Q> const& scalar)
	{
		return vec<3, T, Q>(v) >>= scalar.x;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator>>(T scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar) >>= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator>>(vec<1, T, Q> const& scalar, vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(scalar.x) >>= v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator>>(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return vec<3, T, Q>(v1) >>= v2;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator~(vec<3, T, Q> const& v)
	{
		return vec<3, T, Q>(
			~v.x,
			~v.y,
			~v.z);
	}

	// -- Boolean operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR bool operator==(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return
			detail::compute_equal<T, std::numeric_limits<T>::is_iec559>::call(v1.x, v2.x) &&
			detail::compute_equal<T, std::numeric_limits<T>::is_iec559>::call(v1.y, v2.y) &&
			detail::compute_equal<T, std::numeric_limits<T>::is_iec559>::call(v1.z, v2.z);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR bool operator!=(vec<3, T, Q> const& v1, vec<3, T, Q> const& v2)
	{
		return !(v1 == v2);
	}

	template<qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, bool, Q> operator&&(vec<3, bool, Q> const& v1, vec<3, bool, Q> const& v2)
	{
		return vec<3, bool, Q>(v1.x && v2.x, v1.y && v2.y, v1.z && v2.z);
	}

	template<qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, bool, Q> operator||(vec<3, bool, Q> const& v1, vec<3, bool, Q> const& v2)
	{
		return vec<3, bool, Q>(v1.x || v2.x, v1.y || v2.y, v1.z || v2.z);
	}
}//namespace glm


#if GLM_CONFIG_SIMD == GLM_ENABLE
#	include "type_vec_simd.inl"

namespace glm {

#if (GLM_ARCH & GLM_ARCH_NEON_BIT) && !GLM_CONFIG_XYZW_ONLY
	CTORSL(3, CTOR_FLOAT);
	CTORSL(3, CTOR_INT);
	CTORSL(3, CTOR_UINT);
	CTORSL(3, CTOR_VECF_INT3);
	CTORSL(3, CTOR_VECF_UINT3);
	CTORSL(3, CTOR_VECF_VECF);
	CTORSL(3, CTOR_VECF_VECI);
	CTORSL(3, CTOR_VECF_VECU);
#endif

#if GLM_ARCH & GLM_ARCH_SSE2_BIT
	CTORSL(3, CTOR_FLOAT_COPY3);
	CTORSL(3, CTOR_DOUBLE_COPY3);
	CTORSL(3, CTOR_FLOAT);
	CTORSL(3, CTOR_FLOAT3);
	CTORSL(3, CTOR_DOUBLE3);
	CTORSL(3, CTOR_INT);
	CTORSL(3, CTOR_INT3);
	CTORSL(3, CTOR_VECF_INT3);


	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, double, aligned_highp>::vec(const vec<3, double, aligned_highp>& v)
#if (GLM_ARCH & GLM_ARCH_AVX_BIT)
		:data(v.data) {}
#else
	{
		data.setv(0, v.data.getv(0));
		data.setv(1, v.data.getv(1));
	}
#endif




	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, float, aligned_highp>::vec(const vec<3, float, aligned_highp>& v) :
		data(v.data) {}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, float, aligned_highp>::vec(const vec<3, float, packed_highp>& v)
	{
		data = _mm_set_ps(v[2], v[2], v[1], v[0]);
	}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, float, packed_highp>::vec(const vec<3, float, aligned_highp>& v)
	{
		_mm_store_sd(reinterpret_cast<double*>(this), _mm_castps_pd(v.data));
		__m128 mz = _mm_shuffle_ps(v.data, v.data, _MM_SHUFFLE(2, 2, 2, 2));
		_mm_store_ss(reinterpret_cast<float*>(this)+2, mz);
	}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, double, aligned_highp>::vec(const vec<3, double, packed_highp>& v)
	{
#if (GLM_ARCH & GLM_ARCH_AVX_BIT)
		data = _mm256_set_pd(v[2], v[2], v[1], v[0]);
#else
		data.setv(0, _mm_loadu_pd(reinterpret_cast<const double*>(&v)));
		data.setv(1, _mm_loadu_pd(reinterpret_cast<const double*>(&v)+2));
#endif
	}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, double, packed_highp>::vec(const vec<3, double, aligned_highp>& v)
	{
#if (GLM_ARCH & GLM_ARCH_AVX_BIT)
		__m256d T1 = _mm256_permute_pd(v.data, 1);
		_mm_store_sd((reinterpret_cast<double*>(this)) + 0, _mm256_castpd256_pd128(v.data));
		_mm_store_sd((reinterpret_cast<double*>(this)) + 1, _mm256_castpd256_pd128(T1));
		_mm_store_sd((reinterpret_cast<double*>(this)) + 2, _mm256_extractf128_pd(v.data, 1));
#else
		_mm_storeu_pd(reinterpret_cast<double*>(this), v.data.getv(0));
		_mm_store_sd((reinterpret_cast<double*>(this)) + 2, v.data.getv(1));
#endif
	}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, int, aligned_highp>::vec(const vec<3, int, aligned_highp>& v) :
		data(v.data)
	{
	}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, int, aligned_highp>::vec(const vec<3, int, packed_highp>& v)
	{
		__m128 mx = _mm_load_ss(reinterpret_cast<const float*>(&v[0]));
		__m128 my = _mm_load_ss(reinterpret_cast<const float*>(&v[1]));
		__m128 mz = _mm_load_ss(reinterpret_cast<const float*>(&v[2]));
		__m128 mxy = _mm_unpacklo_ps(mx, my);
		data = _mm_castps_si128(_mm_movelh_ps(mxy, mz));
	}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, int, packed_highp>::vec(const vec<3, int, aligned_highp>& v)
	{
		_mm_store_sd(reinterpret_cast<double*>(this), _mm_castsi128_pd(v.data));
		__m128 mz = _mm_shuffle_ps(_mm_castsi128_ps(v.data), _mm_castsi128_ps(v.data), _MM_SHUFFLE(2, 2, 2, 2));
		_mm_store_ss(reinterpret_cast<float*>(this)+2, mz);
	}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, unsigned int, aligned_highp>::vec(const vec<3, unsigned int, aligned_highp>& v) :
		data(v.data)
	{
	}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, unsigned int, aligned_highp>::vec(const vec<3, unsigned int, packed_highp>& v)
	{
		__m128 mx = _mm_load_ss(reinterpret_cast<const float*>(&v[0]));
		__m128 my = _mm_load_ss(reinterpret_cast<const float*>(&v[1]));
		__m128 mz = _mm_load_ss(reinterpret_cast<const float*>(&v[2]));
		__m128 mxy = _mm_unpacklo_ps(mx, my);
		data = _mm_castps_si128(_mm_movelh_ps(mxy, mz));
	}

	template<>
	template<>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, unsigned int, packed_highp>::vec(const vec<3, unsigned int, aligned_highp>& v)
	{
		_mm_store_sd(reinterpret_cast<double*>(this), _mm_castsi128_pd(v.data));
		__m128 mz = _mm_shuffle_ps(_mm_castsi128_ps(v.data), _mm_castsi128_ps(v.data), _MM_SHUFFLE(2, 2, 2, 2));
		_mm_store_ss(reinterpret_cast<float*>(this) + 2, mz);
	}

	CTORSL(3, CTOR_DOUBLE);
	//CTORSL(3, CTOR_INT64);

#endif //GLM_ARCH & GLM_ARCH_SSE2_BITt


}

#endif
