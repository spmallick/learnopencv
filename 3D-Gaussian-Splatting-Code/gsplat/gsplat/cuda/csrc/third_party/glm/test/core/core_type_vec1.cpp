#define GLM_FORCE_SWIZZLE
#include <glm/gtc/constants.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/vec2.hpp>
#include <vector>

#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic push
#	pragma clang diagnostic ignored "-Wglobal-constructors"
#	pragma clang diagnostic ignored "-Wunused-variable"
#endif

static glm::vec1 g1;
static glm::vec1 g2(1);

#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic pop
#endif

static int test_operators()
{
	int Error = 0;

	glm::ivec1 A(1);
	glm::ivec1 B(1);
	{
		bool R = A != B;
		bool S = A == B;

		Error += (S && !R) ? 0 : 1;
	}

	{
		A *= 1;
		B *= 1;
		A += 1;
		B += 1;

		bool R = A != B;
		bool S = A == B;

		Error += (S && !R) ? 0 : 1;
	}

	return Error;
}

static int test_ctor()
{
	int Error = 0;

#	if GLM_HAS_TRIVIAL_QUERIES
	//	Error += std::is_trivially_default_constructible<glm::vec1>::value ? 0 : 1;
	//	Error += std::is_trivially_copy_assignable<glm::vec1>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::vec1>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::dvec1>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::ivec1>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::uvec1>::value ? 0 : 1;

		Error += std::is_copy_constructible<glm::vec1>::value ? 0 : 1;
#	endif

/*
#if GLM_HAS_INITIALIZER_LISTS
	{
		glm::vec1 a{ 0 };
		std::vector<glm::vec1> v = {
			{0.f},
			{4.f},
			{8.f}};
	}

	{
		glm::dvec2 a{ 0 };
		std::vector<glm::dvec1> v = {
			{0.0},
			{4.0},
			{8.0}};
	}
#endif
*/

	{
		glm::vec1 A = glm::vec2(2.0f);
		Error += glm::all(glm::equal(A, glm::vec1(2.0f), glm::epsilon<float>())) ? 0 : 1;

		glm::vec1 B = glm::vec2(2.0f, 3.0f);
		Error += glm::all(glm::equal(B, glm::vec1(2.0f), glm::epsilon<float>())) ? 0 : 1;

		glm::vec1 C = glm::vec2(2.0f, 3.0);
		Error += glm::all(glm::equal(C, glm::vec1(2.0f), glm::epsilon<float>())) ? 0 : 1;

		//glm::vec1 D = glm::dvec1(2.0); // Build error TODO: What does the specification says?

		glm::vec1 E(glm::dvec2(2.0));
		Error += glm::all(glm::equal(E, glm::vec1(2.0f), glm::epsilon<float>())) ? 0 : 1;

		glm::vec1 F(glm::ivec2(2));
		Error += glm::all(glm::equal(F, glm::vec1(2.0f), glm::epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

static int test_size()
{
	int Error = 0;

	Error += sizeof(glm::vec1) == sizeof(glm::mediump_vec1) ? 0 : 1;
	Error += 4 == sizeof(glm::mediump_vec1) ? 0 : 1;
	Error += sizeof(glm::dvec1) == sizeof(glm::highp_dvec1) ? 0 : 1;
	Error += 8 == sizeof(glm::highp_dvec1) ? 0 : 1;
	Error += glm::vec1().length() == 1 ? 0 : 1;
	Error += glm::dvec1().length() == 1 ? 0 : 1;
	Error += glm::vec1::length() == 1 ? 0 : 1;
	Error += glm::dvec1::length() == 1 ? 0 : 1;

	return Error;
}

static int test_operator_increment()
{
	int Error(0);

	glm::ivec1 v0(1);
	glm::ivec1 v1(v0);
	glm::ivec1 v2(v0);
	glm::ivec1 v3 = ++v1;
	glm::ivec1 v4 = v2++;

	Error += glm::all(glm::equal(v0, v4)) ? 0 : 1;
	Error += glm::all(glm::equal(v1, v2)) ? 0 : 1;
	Error += glm::all(glm::equal(v1, v3)) ? 0 : 1;

	int i0(1);
	int i1(i0);
	int i2(i0);
	int i3 = ++i1;
	int i4 = i2++;

	Error += i0 == i4 ? 0 : 1;
	Error += i1 == i2 ? 0 : 1;
	Error += i1 == i3 ? 0 : 1;

	return Error;
}

static int test_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::vec1 A = glm::vec1(1.0f);
		//glm::vec1 B = A.x;
		glm::vec1 C(A.x);

		//Error += glm::all(glm::equal(A, B)) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR

	return Error;
}

static int test_constexpr()
{
#if GLM_HAS_CONSTEXPR
	static_assert(glm::vec1::length() == 1, "GLM: Failed constexpr");
	static_assert(glm::vec1(1.0f).x > 0.0f, "GLM: Failed constexpr");
#endif

	return 0;
}

int main()
{
	// Suppress unused variable warnings
	(void)g1;
	(void)g2;

	int Error = 0;

	Error += test_size();
	Error += test_ctor();
	Error += test_operators();
	Error += test_operator_increment();
	Error += test_swizzle();
	Error += test_constexpr();
	
	return Error;
}
