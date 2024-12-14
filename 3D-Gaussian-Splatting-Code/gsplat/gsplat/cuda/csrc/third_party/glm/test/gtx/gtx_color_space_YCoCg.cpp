#include <glm/ext/vector_relational.hpp>
#include <glm/ext/scalar_constants.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/color_space_YCoCg.hpp>

int main()
{
	int Error = 0;

	glm::vec3 colorYCoCg = glm::rgb2YCoCg(glm::vec3(1.0f, 0.5f, 0.0f));
	glm::vec3 colorRGB1 = glm::YCoCg2rgb(colorYCoCg);

	Error += glm::all(glm::equal(colorRGB1, glm::vec3(1.0f, 0.5f, 0.0f), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}
