#ifndef GLM_FORCE_PLATFORM_UNKNOWN
#	define GLM_FORCE_PLATFORM_UNKNOWN
#endif
#ifndef GLM_FORCE_COMPILER_UNKNOWN
#	define GLM_FORCE_COMPILER_UNKNOWN
#endif
#ifndef GLM_FORCE_ARCH_UNKNOWN
#	define GLM_FORCE_ARCH_UNKNOWN
#endif
#ifndef GLM_FORCE_CXX_UNKNOWN
#	define GLM_FORCE_CXX_UNKNOWN
#endif

#if defined(__clang__)
#	pragma clang diagnostic push
#	pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#	pragma clang diagnostic ignored "-Wnested-anon-types"
#	pragma clang diagnostic ignored "-Wsign-conversion"
#	pragma clang diagnostic ignored "-Wpadded"
#	pragma clang diagnostic ignored "-Wc++11-long-long"
#elif defined(__GNUC__)
#	pragma GCC diagnostic push
#	pragma GCC diagnostic ignored "-Wpedantic"
#elif defined(_MSC_VER)
#	pragma warning(push)
#	pragma warning(disable: 4201)  // nonstandard extension used : nameless struct/union
#endif

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#if defined(__clang__)
#	pragma clang diagnostic pop
#elif defined(__GNUC__)
#	pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#	pragma warning(pop)
#endif

int main()
{
	int Error = 0;

	return Error;
}
