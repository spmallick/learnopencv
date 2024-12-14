#include <glm/gtc/epsilon.hpp>
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

	glm::mat2x4 l(1.0f);
	glm::mat2x4 m(1.0f);
	glm::vec2 u(1.0f);
	glm::vec4 v(1.0f);

	float x = 1.0f;
	glm::vec4 a = m * u;
	Error += glm::all(glm::equal(a, glm::vec4(u, 0.0f, 0.0f), glm::epsilon<float>())) ? 0 : 1;

	glm::vec2 b = v * m;
	Error += glm::all(glm::equal(b, glm::vec2(1.0f), glm::epsilon<float>())) ? 0 : 1;

	glm::mat2x4 n0(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
	glm::mat2x4 n = x / n0;
	Error += glm::all(glm::equal(n, n0, glm::epsilon<float>())) ? 0 : 1;

	glm::mat2x4 o = m / x;
	Error += glm::all(glm::equal(o, m, glm::epsilon<float>())) ? 0 : 1;

	glm::mat2x4 p = x * m;
	Error += glm::all(glm::equal(p, m, glm::epsilon<float>())) ? 0 : 1;

	glm::mat2x4 q = m * x;
	bool R = glm::any(glm::notEqual(m, q, glm::epsilon<float>()));
	bool S = glm::all(glm::equal(m, l, glm::epsilon<float>()));

	Error += (S && !R) ? 0 : 1;

	return Error;
}

static int test_ctr()
{
	int Error(0);

#if(GLM_HAS_INITIALIZER_LISTS)
	glm::mat2x4 m0(
		glm::vec4(0, 1, 2, 3),
		glm::vec4(4, 5, 6, 7));

	glm::mat2x4 m1{0, 1, 2, 3, 4, 5, 6, 7};

	glm::mat2x4 m2{
		{0, 1, 2, 3},
		{4, 5, 6, 7}};

	Error += glm::all(glm::equal(m0, m2, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(m1, m2, glm::epsilon<float>())) ? 0 : 1;

	std::vector<glm::mat2x4> v1{
		{0, 1, 2, 3, 4, 5, 6, 7},
		{0, 1, 2, 3, 4, 5, 6, 7}
	};
	
	std::vector<glm::mat2x4> v2{
		{
			{ 0, 1, 2, 3},
			{ 4, 5, 6, 7}
		},
		{
			{ 0, 1, 2, 3},
			{ 4, 5, 6, 7}
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
		glm::mat2x4 B(A);
		glm::mat2x4 Identity(1.0f);

		for(glm::length_t i = 0, length = B.length(); i < length; ++i)
			Error += glm::all(glm::epsilonEqual(B[i], Identity[i], glm::epsilon<float>())) ? 0 : 1;

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

	Error += 32 == sizeof(glm::mat2x4) ? 0 : 1;
	Error += 64 == sizeof(glm::dmat2x4) ? 0 : 1;
	Error += glm::mat2x4().length() == 2 ? 0 : 1;
	Error += glm::dmat2x4().length() == 2 ? 0 : 1;
	Error += glm::mat2x4::length() == 2 ? 0 : 1;
	Error += glm::dmat2x4::length() == 2 ? 0 : 1;

	return Error;
}

static int test_constexpr()
{
	int Error = 0;

#if GLM_HAS_CONSTEXPR
	static_assert(glm::mat2x4::length() == 2, "GLM: Failed constexpr");

	constexpr glm::mat2x4 const Z(1.0f);
	Error += glm::all(glm::equal(Z, glm::mat2x4(1.0f), glm::epsilon<float>())) ? 0 : 1;
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



