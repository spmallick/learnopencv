#include <glm/gtc/constants.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/matrix_relational.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat2x3.hpp>
#include <glm/mat2x4.hpp>
#include <glm/mat3x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat3x4.hpp>
#include <glm/mat4x2.hpp>
#include <glm/mat4x3.hpp>
#include <glm/mat4x4.hpp>
#include <vector>

static int test_operators()
{
	int Error = 0;

	glm::mat3x2 l(1.0f);
	glm::mat3x2 m(1.0f);
	glm::vec3 u(1.0f);
	glm::vec2 v(1.0f);

	float x = 1.0f;
	glm::vec2 a = m * u;
	Error += glm::all(glm::equal(a, glm::vec2(1.0f), glm::epsilon<float>())) ? 0 : 1;

	glm::vec3 b = v * m;
	Error += glm::all(glm::equal(b, glm::vec3(v, 0.0f), glm::epsilon<float>())) ? 0 : 1;

	glm::mat3x2 n0(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
	glm::mat3x2 n = x / n0;
	Error += glm::all(glm::equal(n, n0, glm::epsilon<float>())) ? 0 : 1;

	glm::mat3x2 o = m / x;
	Error += glm::all(glm::equal(o, m, glm::epsilon<float>())) ? 0 : 1;

	glm::mat3x2 p = x * m;
	Error += glm::all(glm::equal(p, m, glm::epsilon<float>())) ? 0 : 1;

	glm::mat3x2 q = m * x;
	bool R = glm::any(glm::notEqual(m, q, glm::epsilon<float>()));
	bool S = glm::all(glm::equal(m, l, glm::epsilon<float>()));

	Error += (S && !R) ? 0 : 1;

	return Error;
}

static int test_ctr()
{
	int Error(0);

#if(GLM_HAS_INITIALIZER_LISTS)
	glm::mat3x2 m0(
		glm::vec2(0, 1),
		glm::vec2(2, 3),
		glm::vec2(4, 5));
	
	glm::mat3x2 m1{0, 1, 2, 3, 4, 5};
	
	glm::mat3x2 m2{
		{0, 1},
		{2, 3},
		{4, 5}};

	Error += glm::all(glm::equal(m0, m2, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(m1, m2, glm::epsilon<float>())) ? 0 : 1;

	std::vector<glm::mat3x2> v1{
		{0, 1, 2, 3, 4, 5},
		{0, 1, 2, 3, 4, 5}
	};
	
	std::vector<glm::mat3x2> v2{
		{
			{ 0, 1},
			{ 2, 3},
			{ 4, 5}
		},
		{
			{ 0, 1},
			{ 2, 3},
			{ 4, 5}
		}
	};
	
#endif//GLM_HAS_INITIALIZER_LISTS
	
	return Error;
}

namespace cast
{
	template<typename genType>
	static int entry()
	{
		int Error = 0;

		genType A(1.0f);
		glm::mat3x2 B(A);
		glm::mat3x2 Identity(1.0f);

		Error += glm::all(glm::equal(B, Identity, glm::epsilon<float>())) ? 0 : 1;

		return Error;
	}

	static int test()
	{
		int Error = 0;
		
		Error += entry<glm::mat2x2>();
		Error += entry<glm::mat2x3>();
		Error += entry<glm::mat2x4>();
		Error += entry<glm::mat3x2>();
		Error += entry<glm::mat3x3>();
		Error += entry<glm::mat3x4>();
		Error += entry<glm::mat4x2>();
		Error += entry<glm::mat4x3>();
		Error += entry<glm::mat4x4>();

		return Error;
	}
}//namespace cast

static int test_size()
{
	int Error = 0;

	Error += 24 == sizeof(glm::mat3x2) ? 0 : 1;
	Error += 48 == sizeof(glm::dmat3x2) ? 0 : 1;
	Error += glm::mat3x2().length() == 3 ? 0 : 1;
	Error += glm::dmat3x2().length() == 3 ? 0 : 1;
	Error += glm::mat3x2::length() == 3 ? 0 : 1;
	Error += glm::dmat3x2::length() == 3 ? 0 : 1;

	return Error;
}

static int test_constexpr()
{
	int Error = 0;

#if GLM_HAS_CONSTEXPR
	static_assert(glm::mat3x2::length() == 3, "GLM: Failed constexpr");

	constexpr glm::mat3x2 const Z(1.0f);
	Error += glm::all(glm::equal(Z, glm::mat3x2(1.0f), glm::epsilon<float>())) ? 0 : 1;
#endif

	return Error;
}

int main()
{
	int Error = 0;

	Error += cast::test();
	Error += test_ctr();
	Error += test_operators();
	Error += test_size();
	Error += test_constexpr();

	return Error;
}


