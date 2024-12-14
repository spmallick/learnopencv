#define GLM_FORCE_MESSAGES
#include <glm/vec3.hpp>
#include <cstdio>

static int test_compiler()
{
	int Error(0);
	
#	if(GLM_COMPILER & GLM_COMPILER_VC)
	{
		switch(GLM_COMPILER)
		{
		case GLM_COMPILER_VC12:
			std::printf("Visual C++ 12 - 2013\n");
			break;
		case GLM_COMPILER_VC14:
			std::printf("Visual C++ 14 - 2015\n");
			break;
		case GLM_COMPILER_VC15:
			std::printf("Visual C++ 15 - 2017\n");
			break;
		case GLM_COMPILER_VC15_3:
			std::printf("Visual C++ 15.3 - 2017\n");
			break;
		case GLM_COMPILER_VC15_5:
			std::printf("Visual C++ 15.5 - 2017\n");
			break;
		case GLM_COMPILER_VC15_6:
			std::printf("Visual C++ 15.6 - 2017\n");
			break;
		case GLM_COMPILER_VC15_7:
			std::printf("Visual C++ 15.7 - 2017\n");
			break;
		case GLM_COMPILER_VC15_8:
			std::printf("Visual C++ 15.8 - 2017\n");
			break;
		case GLM_COMPILER_VC15_9:
			std::printf("Visual C++ 15.9 - 2017\n");
			break;
		case GLM_COMPILER_VC16:
			std::printf("Visual C++ 16 - 2019\n");
			break;
		case GLM_COMPILER_VC17:
			std::printf("Visual C++ 17 - 2022\n");
			break;
		default:
			std::printf("Visual C++ version not detected\n");
			Error += 1;
			break;
		}
	}
#	elif(GLM_COMPILER & GLM_COMPILER_GCC)
	{
		switch(GLM_COMPILER)
		{
		case GLM_COMPILER_GCC46:
			std::printf("GCC 4.6\n");
			break;
		case GLM_COMPILER_GCC47:
			std::printf("GCC 4.7\n");
			break;
		case GLM_COMPILER_GCC48:
			std::printf("GCC 4.8\n");
			break;
		case GLM_COMPILER_GCC49:
			std::printf("GCC 4.9\n");
			break;
		case GLM_COMPILER_GCC5:
			std::printf("GCC 5\n");
			break;
		case GLM_COMPILER_GCC6:
			std::printf("GCC 6\n");
			break;
		case GLM_COMPILER_GCC7:
			std::printf("GCC 7\n");
			break;
		case GLM_COMPILER_GCC8:
			std::printf("GCC 8\n");
			break;
		case GLM_COMPILER_GCC9:
			std::printf("GCC 9\n");
			break;
		case GLM_COMPILER_GCC10:
			std::printf("GCC 10\n");
			break;
		case GLM_COMPILER_GCC11:
			std::printf("GCC 11\n");
			break;
		case GLM_COMPILER_GCC12:
			std::printf("GCC 12\n");
			break;
		case GLM_COMPILER_GCC13:
			std::printf("GCC 13\n");
			break;
		case GLM_COMPILER_GCC14:
			std::printf("GCC 14\n");
			break;
		default:
			std::printf("GCC version not detected\n");
			Error += 1;
			break;
		}
	}
#	elif(GLM_COMPILER & GLM_COMPILER_CUDA)
	{
		std::printf("CUDA\n");
	}
#	elif(GLM_COMPILER & GLM_COMPILER_CLANG)
	{
		switch(GLM_COMPILER)
		{
		case GLM_COMPILER_CLANG34:
			std::printf("Clang 3.4\n");
			break;
		case GLM_COMPILER_CLANG35:
			std::printf("Clang 3.5\n");
			break;
		case GLM_COMPILER_CLANG36:
			std::printf("Clang 3.6\n");
			break;
		case GLM_COMPILER_CLANG37:
			std::printf("Clang 3.7\n");
			break;
		case GLM_COMPILER_CLANG38:
			std::printf("Clang 3.8\n");
			break;
		case GLM_COMPILER_CLANG39:
			std::printf("Clang 3.9\n");
			break;
		case GLM_COMPILER_CLANG4:
			std::printf("Clang 4\n");
			break;
		case GLM_COMPILER_CLANG5:
			std::printf("Clang 5\n");
			break;
		case GLM_COMPILER_CLANG6:
			std::printf("Clang 6\n");
			break;
		case GLM_COMPILER_CLANG7:
			std::printf("Clang 7\n");
			break;
		case GLM_COMPILER_CLANG8:
			std::printf("Clang 8\n");
			break;
		case GLM_COMPILER_CLANG9:
			std::printf("Clang 9\n");
			break;
		case GLM_COMPILER_CLANG10:
			std::printf("Clang 10\n");
			break;
		case GLM_COMPILER_CLANG11:
			std::printf("Clang 11\n");
			break;
		case GLM_COMPILER_CLANG12:
			std::printf("Clang 12\n");
			break;
		case GLM_COMPILER_CLANG13:
			std::printf("Clang 13\n");
			break;
		case GLM_COMPILER_CLANG14:
			std::printf("Clang 14\n");
			break;
		case GLM_COMPILER_CLANG15:
			std::printf("Clang 15\n");
			break;
		case GLM_COMPILER_CLANG16:
			std::printf("Clang 16\n");
			break;
		case GLM_COMPILER_CLANG17:
			std::printf("Clang 17\n");
			break;
		case GLM_COMPILER_CLANG18:
			std::printf("Clang 18\n");
			break;
		case GLM_COMPILER_CLANG19:
			std::printf("Clang 19\n");
			break;
		default:
			std::printf("LLVM version not detected\n");
			break;
		}
	}
#	elif(GLM_COMPILER & GLM_COMPILER_INTEL)
	{
		switch(GLM_COMPILER)
		{
		case GLM_COMPILER_INTEL14:
			std::printf("ICC 14 - 2013 SP1\n");
			break;
		case GLM_COMPILER_INTEL15:
			std::printf("ICC 15 - 2015\n");
			break;
		case GLM_COMPILER_INTEL16:
			std::printf("ICC 16 - 2015\n");
			break;
		case GLM_COMPILER_INTEL17:
			std::printf("ICC 17 - 2016\n");
			break;
		case GLM_COMPILER_INTEL18:
			std::printf("ICC 18 - 2017\n");
			break;
		case GLM_COMPILER_INTEL19:
			std::printf("ICC 19 - 2018\n");
			break;
		case GLM_COMPILER_INTEL21:
			std::printf("ICC 21 - 2021\n");
			break;
		default:
			std::printf("Intel compiler version not detected\n");
			Error += 1;
			break;
		}
	}
#else
	{
		std::printf("Undetected compiler\n");
		Error += 1;
	}
#endif
	return Error;
}

static int test_model()
{
	int Error = 0;
	
	Error += ((sizeof(void*) == 4) && (GLM_MODEL == GLM_MODEL_32)) || ((sizeof(void*) == 8) && (GLM_MODEL == GLM_MODEL_64)) ? 0 : 1;
	
#	if GLM_MODEL == GLM_MODEL_32
		std::printf("GLM_MODEL_32\n");
#	elif GLM_MODEL == GLM_MODEL_64
		std::printf("GLM_MODEL_64\n");
#	endif

	return Error;
}

static int test_instruction_set()
{
	int Error = 0;

	std::printf("GLM_ARCH: ");

#	if(GLM_ARCH & GLM_ARCH_ARM_BIT)
		std::printf("ARM ");
#	elif(GLM_ARCH & GLM_ARCH_NEON_BIT)
		std::printf("NEON ");
#	elif(GLM_ARCH & GLM_ARCH_AVX2_BIT)
		std::printf("AVX2 ");
#	elif(GLM_ARCH & GLM_ARCH_AVX_BIT)
		std::printf("AVX ");
#	elif(GLM_ARCH & GLM_ARCH_SSE42_BIT)
		std::printf("SSE4.2 ");
#	elif(GLM_ARCH & GLM_ARCH_SSE41_BIT)
		std::printf("SSE4.1 ");
#	elif(GLM_ARCH & GLM_ARCH_SSSE3_BIT)
		std::printf("SSSE3 ");
#	elif(GLM_ARCH & GLM_ARCH_SSE3_BIT)
		std::printf("SSE3 ");
#	elif(GLM_ARCH & GLM_ARCH_SSE2_BIT)
		std::printf("SSE2 ");
#	endif

	std::printf("\n");

	return Error;
}

static int test_cpp_version()
{
	std::printf("__cplusplus: %d\n", static_cast<int>(__cplusplus));
	
	std::printf("GLM_LANG: ");

#	if(GLM_LANG & GLM_LANG_CXX20_FLAG)
		std::printf("C++ 20");
#	elif(GLM_LANG & GLM_LANG_CXX17_FLAG)
		std::printf("C++ 17");
#	elif(GLM_LANG & GLM_LANG_CXX14_FLAG)
		std::printf("C++ 14");
#	elif(GLM_LANG & GLM_LANG_CXX11_FLAG)
		std::printf("C++ 11");
#	elif(GLM_LANG & GLM_LANG_CXX98_FLAG)
		std::printf("C++ 98");
#	endif

	return 0;
}

static int test_operators()
{
	glm::ivec3 A(1);
	glm::ivec3 B(1);
	bool R = A != B;
	bool S = A == B;

	return (S && !R) ? 0 : 1;
}

int main()
{
	int Error = 0;

#	if !defined(GLM_FORCE_PLATFORM_UNKNOWN) && !defined(GLM_FORCE_COMPILER_UNKNOWN) && !defined(GLM_FORCE_ARCH_UNKNOWN) && !defined(GLM_FORCE_CXX_UNKNOWN)
		
		Error += test_cpp_version();
		Error += test_compiler();
		Error += test_model();
		Error += test_instruction_set();
		Error += test_operators();

#	endif
	
	return Error;
}
