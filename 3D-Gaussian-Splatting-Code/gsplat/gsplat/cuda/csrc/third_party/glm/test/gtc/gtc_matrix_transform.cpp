#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/scalar_relational.hpp>

static int test_perspective()
{
	int Error = 0;

	const float Near = 0.1f;
	const float Far = 100.0f;
	const float Eps = glm::epsilon<float>();

	glm::mat4 Projection = glm::perspective(glm::pi<float>() * 0.25f, 4.0f / 3.0f, Near, Far);

	Projection = glm::perspectiveLH_ZO(glm::pi<float>() * 0.25f, 4.0f / 3.0f, Near, Far);
	{
		glm::vec4 N = Projection * glm::vec4(0.f, 0.f, Near, 1.f);
		glm::vec4 F = Projection * glm::vec4(0.f, 0.f, Far, 1.f);
		N /= N.w;
		F /= F.w;
		Error += glm::notEqual(N.z, 0.f, Eps);
		Error += glm::notEqual(F.z, 1.f, Eps);
	}

	Projection = glm::perspectiveLH_NO(glm::pi<float>() * 0.25f, 4.0f / 3.0f, Near, Far);
	{
		glm::vec4 N = Projection * glm::vec4(0.f, 0.f, Near, 1.f);
		glm::vec4 F = Projection * glm::vec4(0.f, 0.f, Far, 1.f);
		N /= N.w;
		F /= F.w;
		Error += glm::notEqual(N.z, -1.f, Eps);
		Error += glm::notEqual(F.z, 1.f, Eps);
	}
	
	Projection = glm::perspectiveRH_ZO(glm::pi<float>() * 0.25f, 4.0f / 3.0f, Near, Far);
	{
		glm::vec4 N = Projection * glm::vec4(0.f, 0.f, -Near, 1.f);
		glm::vec4 F = Projection * glm::vec4(0.f, 0.f, -Far, 1.f);
		N /= N.w;
		F /= F.w;
		Error += glm::notEqual(N.z, 0.f, Eps);
		Error += glm::notEqual(F.z, 1.f, Eps);
	}

	Projection = glm::perspectiveRH_NO(glm::pi<float>() * 0.25f, 4.0f / 3.0f, Near, Far);
	{
		glm::vec4 N = Projection * glm::vec4(0.f, 0.f, -Near, 1.f);
		glm::vec4 F = Projection * glm::vec4(0.f, 0.f, -Far, 1.f);
		N /= N.w;
		F /= F.w;
		Error += glm::notEqual(N.z, -1.f, Eps);
		Error += glm::notEqual(F.z, 1.f, Eps);
	}

	return Error;
}

static int test_infinitePerspective()
{
	int Error = 0;

	const float Near = 0.1f;
	const float Inf = 1.0e+10f;
	const float Eps = glm::epsilon<float>();

	glm::mat4 Projection = glm::infinitePerspective(glm::pi<float>() * 0.25f, 4.0f / 3.0f, Near);

	Projection = glm::infinitePerspectiveLH_ZO(glm::pi<float>() * 0.25f, 4.0f / 3.0f, Near);
	{
		glm::vec4 N = Projection * glm::vec4(0.f, 0.f, Near, 1.f);
		glm::vec4 F = Projection * glm::vec4(0.f, 0.f, Inf, 1.f);
		N /= N.w;
		F /= F.w;
		Error += glm::notEqual(N.z, 0.f, Eps);
		Error += glm::notEqual(F.z, 1.f, Eps);
	}

	Projection = glm::infinitePerspectiveLH_NO(glm::pi<float>() * 0.25f, 4.0f / 3.0f, Near);
	{
		glm::vec4 N = Projection * glm::vec4(0.f, 0.f, Near, 1.f);
		glm::vec4 F = Projection * glm::vec4(0.f, 0.f, Inf, 1.f);
		N /= N.w;
		F /= F.w;
		Error += glm::notEqual(N.z, -1.f, Eps);
		Error += glm::notEqual(F.z, 1.f, Eps);
	}

	Projection = glm::infinitePerspectiveRH_ZO(glm::pi<float>() * 0.25f, 4.0f / 3.0f, Near);
	{
		glm::vec4 N = Projection * glm::vec4(0.f, 0.f, -Near, 1.f);
		glm::vec4 F = Projection * glm::vec4(0.f, 0.f, -Inf, 1.f);
		N /= N.w;
		F /= F.w;
		Error += glm::notEqual(N.z, 0.f, Eps);
		Error += glm::notEqual(F.z, 1.f, Eps);
	}

	Projection = glm::infinitePerspectiveRH_NO(glm::pi<float>() * 0.25f, 4.0f / 3.0f, Near);
	{
		glm::vec4 N = Projection * glm::vec4(0.f, 0.f, -Near, 1.f);
		glm::vec4 F = Projection * glm::vec4(0.f, 0.f, -Inf, 1.f);
		N /= N.w;
		F /= F.w;
		Error += glm::notEqual(N.z, -1.f, Eps);
		Error += glm::notEqual(F.z, 1.f, Eps);
	}

	return Error;
}

static int test_pick()
{
	int Error = 0;

	glm::mat4 Pick = glm::pickMatrix(glm::vec2(1, 2), glm::vec2(3, 4), glm::ivec4(0, 0, 320, 240));
	Error += !glm::all(glm::notEqual(Pick, glm::mat4(2.0), 0.001f));

	return Error;
}

static int test_tweakedInfinitePerspective()
{
	int Error = 0;

	glm::mat4 ProjectionA = glm::tweakedInfinitePerspective(45.f, 640.f/480.f, 1.0f);
	glm::mat4 ProjectionB = glm::tweakedInfinitePerspective(45.f, 640.f/480.f, 1.0f, 0.001f);

	Error += !glm::all(glm::notEqual(ProjectionA, glm::mat4(1.0), 0.001f));
	Error += !glm::all(glm::notEqual(ProjectionB, glm::mat4(1.0), 0.001f));

	return Error;
}

static int test_translate()
{
	int Error = 0;

	glm::lowp_vec3 v(1.0);
	glm::lowp_mat4 m(0);
	glm::lowp_mat4 t = glm::translate(m, v);

	glm::bvec4 b = glm::notEqual(t, glm::lowp_mat4(1.0), 0.001f);

	Error += !glm::all(b);

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_translate();
	Error += test_tweakedInfinitePerspective();
	Error += test_pick();
	Error += test_perspective();
	Error += test_infinitePerspective();

	return Error;
}
