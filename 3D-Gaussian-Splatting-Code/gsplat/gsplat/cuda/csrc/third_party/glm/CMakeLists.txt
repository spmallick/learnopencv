cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
cmake_policy(VERSION 3.6)

file(READ "glm/detail/setup.hpp" GLM_SETUP_FILE)
string(REGEX MATCH "#define[ ]+GLM_VERSION_MAJOR[ ]+([0-9]+)" _ ${GLM_SETUP_FILE})
set(GLM_VERSION_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define[ ]+GLM_VERSION_MINOR[ ]+([0-9]+)" _ ${GLM_SETUP_FILE})
set(GLM_VERSION_MINOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define[ ]+GLM_VERSION_PATCH[ ]+([0-9]+)" _ ${GLM_SETUP_FILE})
set(GLM_VERSION_PATCH "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define[ ]+GLM_VERSION_REVISION[ ]+([0-9]+)" _ ${GLM_SETUP_FILE})
set(GLM_VERSION_REVISION "${CMAKE_MATCH_1}")

set(GLM_VERSION ${GLM_VERSION_MAJOR}.${GLM_VERSION_MINOR}.${GLM_VERSION_PATCH})
project(glm VERSION ${GLM_VERSION} LANGUAGES CXX)
message(STATUS "GLM: Version " ${GLM_VERSION})

set(GLM_IS_MASTER_PROJECT OFF)
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
	set(GLM_IS_MASTER_PROJECT ON)
endif()

option(GLM_BUILD_LIBRARY "Build dynamic/static library" ON)
option(GLM_BUILD_TESTS "Build the test programs" OFF)
option(GLM_BUILD_INSTALL "Generate the install target" ${GLM_IS_MASTER_PROJECT})

include(GNUInstallDirs)

option(GLM_ENABLE_CXX_98 "Enable C++ 98" OFF)
option(GLM_ENABLE_CXX_11 "Enable C++ 11" OFF)
option(GLM_ENABLE_CXX_14 "Enable C++ 14" OFF)
option(GLM_ENABLE_CXX_17 "Enable C++ 17" OFF)
option(GLM_ENABLE_CXX_20 "Enable C++ 20" OFF)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(GLM_ENABLE_CXX_20)
	set(CMAKE_CXX_STANDARD 20)
	add_definitions(-DGLM_FORCE_CXX20)
	if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		message(STATUS "GLM: Disable -Wc++98-compat warnings")
		add_compile_options(-Wno-c++98-compat)
		add_compile_options(-Wno-c++98-compat-pedantic)
	endif()
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with C++20 features")
	endif()

elseif(GLM_ENABLE_CXX_17)
	set(CMAKE_CXX_STANDARD 17)
	add_definitions(-DGLM_FORCE_CXX17)
	if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		message(STATUS "GLM: Disable -Wc++98-compat warnings")
		add_compile_options(-Wno-c++98-compat)
		add_compile_options(-Wno-c++98-compat-pedantic)
	endif()
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with C++17 features")
	endif()

elseif(GLM_ENABLE_CXX_14)
	set(CMAKE_CXX_STANDARD 14)
	add_definitions(-DGLM_FORCE_CXX14)
	if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		message(STATUS "GLM: Disable -Wc++98-compat warnings")
		add_compile_options(-Wno-c++98-compat)
		add_compile_options(-Wno-c++98-compat-pedantic)
	endif()
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with C++14 features")
	endif()

elseif(GLM_ENABLE_CXX_11)
	set(CMAKE_CXX_STANDARD 11)
	add_definitions(-DGLM_FORCE_CXX11)
	if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		message(STATUS "GLM: Disable -Wc++98-compat warnings")
		add_compile_options(-Wno-c++98-compat)
		add_compile_options(-Wno-c++98-compat-pedantic)
	endif()
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with C++11 features")
	endif()

elseif(GLM_ENABLE_CXX_98)
	set(CMAKE_CXX_STANDARD 98)
	add_definitions(-DGLM_FORCE_CXX98)
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with C++98 features")
	endif()

else()
	if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		message(STATUS "GLM: Disable -Wc++98-compat warnings")
		add_compile_options(-Wno-c++98-compat)
		add_compile_options(-Wno-c++98-compat-pedantic)
	endif()
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with C++ features auto detection")
	endif()

endif()

option(GLM_ENABLE_LANG_EXTENSIONS "Enable language extensions" OFF)
option(GLM_DISABLE_AUTO_DETECTION "Disable platform, compiler, arch and C++ language detection" OFF)

if(GLM_DISABLE_AUTO_DETECTION)
	add_definitions(-DGLM_FORCE_PLATFORM_UNKNOWN -DGLM_FORCE_COMPILER_UNKNOWN -DGLM_FORCE_ARCH_UNKNOWN -DGLM_FORCE_CXX_UNKNOWN)
endif()

if(GLM_ENABLE_LANG_EXTENSIONS)
	set(CMAKE_CXX_EXTENSIONS ON)
	if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
		add_compile_options(-fms-extensions)
	endif()
	message(STATUS "GLM: Build with C++ language extensions")
else()
	set(CMAKE_CXX_EXTENSIONS OFF)
	if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/Za)
		if(MSVC15)
			add_compile_options(/permissive-)
		endif()
	endif()
endif()

option(GLM_ENABLE_FAST_MATH "Enable fast math optimizations" OFF)
if(GLM_ENABLE_FAST_MATH)
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with fast math optimizations")
	endif()

	if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
		add_compile_options(-ffast-math)

	elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/fp:fast)
	endif()
else()
	if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/fp:precise)
	endif()
endif()

option(GLM_ENABLE_SIMD_SSE2 "Enable SSE2 optimizations" OFF)
option(GLM_ENABLE_SIMD_SSE3 "Enable SSE3 optimizations" OFF)
option(GLM_ENABLE_SIMD_SSSE3 "Enable SSSE3 optimizations" OFF)
option(GLM_ENABLE_SIMD_SSE4_1 "Enable SSE 4.1 optimizations" OFF)
option(GLM_ENABLE_SIMD_SSE4_2 "Enable SSE 4.2 optimizations" OFF)
option(GLM_ENABLE_SIMD_AVX "Enable AVX optimizations" OFF)
option(GLM_ENABLE_SIMD_AVX2 "Enable AVX2 optimizations" OFF)
option(GLM_TEST_ENABLE_SIMD_NEON "Enable ARM NEON optimizations" OFF)
option(GLM_FORCE_PURE "Force 'pure' instructions" OFF)

if(GLM_FORCE_PURE)
	add_definitions(-DGLM_FORCE_PURE)

	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
		add_compile_options(-mfpmath=387)
	endif()
	message(STATUS "GLM: No SIMD instruction set")

elseif(GLM_ENABLE_SIMD_AVX2)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-mavx2)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxAVX2)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/arch:AVX2)
	endif()
	message(STATUS "GLM: AVX2 instruction set")

elseif(GLM_ENABLE_SIMD_AVX)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-mavx)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxAVX)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/arch:AVX)
	endif()
	message(STATUS "GLM: AVX instruction set")

elseif(GLM_ENABLE_SIMD_SSE4_2)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-msse4.2)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxSSE4.2)
	elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
		add_compile_options(/arch:SSE2) # VC doesn't support SSE4.2
	endif()
	message(STATUS "GLM: SSE4.2 instruction set")

elseif(GLM_ENABLE_SIMD_SSE4_1)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-msse4.1)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxSSE4.1)
	elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
		add_compile_options(/arch:SSE2) # VC doesn't support SSE4.1
	endif()
	message(STATUS "GLM: SSE4.1 instruction set")

elseif(GLM_ENABLE_SIMD_SSSE3)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-mssse3)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxSSSE3)
	elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
		add_compile_options(/arch:SSE2) # VC doesn't support SSSE3
	endif()
	message(STATUS "GLM: SSSE3 instruction set")

elseif(GLM_ENABLE_SIMD_SSE3)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-msse3)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxSSE3)
	elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
		add_compile_options(/arch:SSE2) # VC doesn't support SSE3
	endif()
	message(STATUS "GLM: SSE3 instruction set")

elseif(GLM_ENABLE_SIMD_SSE2)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-msse2)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxSSE2)
	elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
		add_compile_options(/arch:SSE2)
	endif()
	message(STATUS "GLM: SSE2 instruction set")
elseif(GLM_TEST_ENABLE_SIMD_NEON)
	add_definitions(-DGLM_FORCE_NEON)
	message(STATUS "GLM: ARM NEON instruction set")
endif()

add_subdirectory(glm)

if (GLM_BUILD_TESTS)
	include(CTest)
	add_subdirectory(test)
endif()

if (GLM_BUILD_INSTALL)
	include(CPack)

	install(TARGETS glm-header-only glm EXPORT glm)
	install(
		DIRECTORY glm
		DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
		PATTERN "CMakeLists.txt" EXCLUDE
	)
	install(
		EXPORT glm
		NAMESPACE glm::
		DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/glm"
		FILE glmConfig.cmake
	)
	include(CMakePackageConfigHelpers)
	write_basic_package_version_file(
		"${CMAKE_CURRENT_BINARY_DIR}/glmConfigVersion.cmake"
		COMPATIBILITY AnyNewerVersion
	)
	install(
		FILES "${CMAKE_CURRENT_BINARY_DIR}/glmConfigVersion.cmake"
		DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/glm"
	)

	configure_file(
		"${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
		"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
		IMMEDIATE @ONLY
	)

	add_custom_target(
		uninstall
		"${CMAKE_COMMAND}" -P
		"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
	)
endif()
