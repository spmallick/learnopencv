#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/type_aligned.hpp>
#include <cstdio>

#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic push
#	pragma clang diagnostic ignored "-Wpadded"
#endif

static int test_decl()
{
	int Error(0);

	{
		struct S1
		{
			glm::aligned_vec4 B;
		};

		struct S2
		{
			glm::vec4 B;
		};

		std::printf("vec4 - Aligned: %d, unaligned: %d\n", static_cast<int>(sizeof(S1)), static_cast<int>(sizeof(S2)));

		Error += sizeof(S1) >= sizeof(S2) ? 0 : 1;
	}

	{
		struct S1
		{
			bool A;
			glm::vec3 B;
		};

#		if GLM_COMPILER & GLM_COMPILER_VC
#			pragma warning(push)
#			pragma warning(disable : 4324)
#		endif

		struct S2
		{
			bool A;
			glm::aligned_vec3 B;
		};

#		if GLM_COMPILER & GLM_COMPILER_VC
#			pragma warning(pop)
#		endif

		std::printf("vec3 - Aligned: %d, unaligned: %d\n", static_cast<int>(sizeof(S1)), static_cast<int>(sizeof(S2)));

		Error += sizeof(S1) <= sizeof(S2) ? 0 : 1;
	}

	{
#		if GLM_COMPILER & GLM_COMPILER_VC
#			pragma warning(push)
#			pragma warning(disable : 4324)
#		endif

		struct S1
		{
			bool A;
			glm::aligned_vec4 B;
		};

#		if GLM_COMPILER & GLM_COMPILER_VC
#			pragma warning(pop)
#		endif

		struct S2
		{
			bool A;
			glm::vec4 B;
		};

		std::printf("vec4 - Aligned: %d, unaligned: %d\n", static_cast<int>(sizeof(S1)), static_cast<int>(sizeof(S2)));

		Error += sizeof(S1) >= sizeof(S2) ? 0 : 1;
	}

	{
#		if GLM_COMPILER & GLM_COMPILER_VC
#			pragma warning(push)
#			pragma warning(disable : 4324)
#		endif

		struct S1
		{
			bool A;
			glm::aligned_dvec4 B;
		};

#		if GLM_COMPILER & GLM_COMPILER_VC
#			pragma warning(pop)
#		endif

		struct S2
		{
			bool A;
			glm::dvec4 B;
		};

		std::printf("dvec4 - Aligned: %d, unaligned: %d\n", static_cast<int>(sizeof(S1)), static_cast<int>(sizeof(S2)));

		Error += sizeof(S1) >= sizeof(S2) ? 0 : 1;
	}

	return Error;
}

template<typename genType>
static void print(genType const& Mat0)
{
	std::printf("mat4(\n");
	std::printf("\tvec4(%2.9f, %2.9f, %2.9f, %2.9f)\n", static_cast<double>(Mat0[0][0]), static_cast<double>(Mat0[0][1]), static_cast<double>(Mat0[0][2]), static_cast<double>(Mat0[0][3]));
	std::printf("\tvec4(%2.9f, %2.9f, %2.9f, %2.9f)\n", static_cast<double>(Mat0[1][0]), static_cast<double>(Mat0[1][1]), static_cast<double>(Mat0[1][2]), static_cast<double>(Mat0[1][3]));
	std::printf("\tvec4(%2.9f, %2.9f, %2.9f, %2.9f)\n", static_cast<double>(Mat0[2][0]), static_cast<double>(Mat0[2][1]), static_cast<double>(Mat0[2][2]), static_cast<double>(Mat0[2][3]));
	std::printf("\tvec4(%2.9f, %2.9f, %2.9f, %2.9f))\n\n", static_cast<double>(Mat0[3][0]), static_cast<double>(Mat0[3][1]), static_cast<double>(Mat0[3][2]), static_cast<double>(Mat0[3][3]));
}

static int perf_mul()
{
	int Error = 0;

	glm::mat4 A(1.0f);
	glm::mat4 B(1.0f);

	glm::mat4 C = A * B;

	print(C);

	return Error;
}

#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic pop
#endif

int main()
{
	int Error(0);

	Error += test_decl();
	Error += perf_mul();

	return Error;
}
