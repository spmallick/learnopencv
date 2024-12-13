#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/scalar_constants.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/spline.hpp>

namespace catmullRom
{
	static int test()
	{
		int Error = 0;

		glm::vec2 Result2 = glm::catmullRom(
			glm::vec2(0.0f, 0.0f),
			glm::vec2(1.0f, 0.0f),
			glm::vec2(1.0f, 1.0f),
			glm::vec2(0.0f, 1.0f), 0.5f);

		Error += glm::all(glm::equal(Result2, glm::vec2(1.125f, 0.5f), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 Result3 = glm::catmullRom(
			glm::vec3(0.0f, 0.0f, 0.0f), 
			glm::vec3(1.0f, 0.0f, 0.0f), 
			glm::vec3(1.0f, 1.0f, 0.0f), 
			glm::vec3(0.0f, 1.0f, 0.0f), 0.5f);

		Error += glm::all(glm::equal(Result3, glm::vec3(1.125f, 0.5f, 0.0f), glm::epsilon<float>())) ? 0 : 1;

		glm::vec4 Result4 = glm::catmullRom(
			glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 
			glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), 
			glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), 
			glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.5f);

		Error += glm::all(glm::equal(Result4, glm::vec4(1.125f, 0.5f, 0.0f, 1.0f), glm::epsilon<float>())) ? 0 : 1;

		return Error;
	}
}//catmullRom

namespace hermite
{
	static int test()
	{
		int Error = 0;

		glm::vec2 Result2 = glm::hermite(
			glm::vec2(0.0f, 0.0f),
			glm::vec2(1.0f, 0.0f),
			glm::vec2(1.0f, 1.0f),
			glm::vec2(0.0f, 1.0f), 0.5f);

		Error += glm::all(glm::equal(Result2, glm::vec2(0.625f, 0.375f), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 Result3 = glm::hermite(
			glm::vec3(0.0f, 0.0f, 0.0f), 
			glm::vec3(1.0f, 0.0f, 0.0f), 
			glm::vec3(1.0f, 1.0f, 0.0f), 
			glm::vec3(0.0f, 1.0f, 0.0f), 0.5f);

		Error += glm::all(glm::equal(Result3, glm::vec3(0.625f, 0.375f, 0.0f), glm::epsilon<float>())) ? 0 : 1;

		glm::vec4 Result4 = glm::hermite(
			glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 
			glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), 
			glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), 
			glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.5f);

		Error += glm::all(glm::equal(Result4, glm::vec4(0.625f, 0.375f, 0.0f, 1.0f), glm::epsilon<float>())) ? 0 : 1;

		return Error;
	}
}//catmullRom

namespace cubic
{
	static int test()
	{
		int Error = 0;

		glm::vec2 Result2 = glm::cubic(
			glm::vec2(0.0f, 0.0f),
			glm::vec2(1.0f, 0.0f),
			glm::vec2(1.0f, 1.0f),
			glm::vec2(0.0f, 1.0f), 0.5f);

		Error += glm::all(glm::equal(Result2, glm::vec2(0.75f, 1.5f), glm::epsilon<float>())) ? 0 : 1;

		glm::vec3 Result3 = glm::cubic(
			glm::vec3(0.0f, 0.0f, 0.0f), 
			glm::vec3(1.0f, 0.0f, 0.0f), 
			glm::vec3(1.0f, 1.0f, 0.0f), 
			glm::vec3(0.0f, 1.0f, 0.0f), 0.5f);

		Error += glm::all(glm::equal(Result3, glm::vec3(0.75f, 1.5f, 0.0f), glm::epsilon<float>())) ? 0 : 1;

		glm::vec4 Result4 = glm::cubic(
			glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 
			glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), 
			glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), 
			glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.5f);

		Error += glm::all(glm::equal(Result4, glm::vec4(0.75f, 1.5f, 0.0f, 1.875f), glm::epsilon<float>())) ? 0 : 1;

		return Error;
	}
}//catmullRom

int main()
{
	int Error(0);

	Error += catmullRom::test();
	Error += hermite::test();
	Error += cubic::test();

	return Error;
}
