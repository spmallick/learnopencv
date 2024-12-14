#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>

static int test_identity() {
	int Error = 0;

	glm::mat4 Matrix(1);

	glm::vec3 Scale;
	glm::quat Orientation;
	glm::vec3 Translation;
	glm::vec3 Skew(1);
	glm::vec4 Perspective(1);

	glm::decompose(Matrix, Scale, Orientation, Translation, Skew, Perspective);

	glm::mat4 Out = glm::recompose(Scale, Orientation, Translation, Skew, Perspective);

	Error += glm::all(glm::equal(Matrix, Out, glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

static int test_scale_translate() {
	int Error = 0;

	glm::vec3 const T(2.0f);
	glm::vec3 const S(2.0f);

	glm::mat4 Matrix = glm::translate(glm::scale(glm::mat4(1), S), T);

	glm::vec3 Scale(2);
	glm::quat Orientation;
	glm::vec3 Translation(2);
	glm::vec3 Skew(1);
	glm::vec4 Perspective(1);

	glm::decompose(Matrix, Scale, Orientation, Translation, Skew, Perspective);

	glm::mat4 Out = glm::recompose(Scale, Orientation, Translation, Skew, Perspective);

	Error += glm::all(glm::equal(Matrix, Out, glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_identity();
	Error += test_scale_translate();

	return Error;
}
