#define GLM_FORCE_UNRESTRICTED_GENTYPE

#include <glm/ext/vector_integer.hpp>
#include <glm/ext/scalar_int_sized.hpp>
#include <glm/ext/scalar_uint_sized.hpp>
#include <glm/ext/vector_int1.hpp>
#include <glm/ext/vector_int2.hpp>
#include <glm/ext/vector_int3.hpp>
#include <glm/ext/vector_int4.hpp>
#include <glm/ext/vector_bool1.hpp>
#include <glm/ext/vector_bool2.hpp>
#include <glm/ext/vector_bool3.hpp>
#include <glm/ext/vector_bool4.hpp>
#include <vector>
#include <ctime>
#include <cstdio>

namespace isPowerOfTwo
{
#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic push
#	pragma clang diagnostic ignored "-Wpadded"
#endif

	template<typename genType>
	struct type
	{
		genType		Value;
		bool		Return;
	};

#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic pop
#endif

	template <glm::length_t L>
	static int test_int16()
	{
		type<glm::int16> const Data[] =
		{
			{ 0x0001, true },
			{ 0x0002, true },
			{ 0x0004, true },
			{ 0x0080, true },
			{ 0x0000, true },
			{ 0x0003, false }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::int16>); i < n; ++i)
		{
			glm::vec<L, bool> const Result = glm::isPowerOfTwo(glm::vec<L, glm::int16>(Data[i].Value));
			Error += glm::vec<L, bool>(Data[i].Return) == Result ? 0 : 1;
		}

		return Error;
	}

	template <glm::length_t L>
	static int test_uint16()
	{
		type<glm::uint16> const Data[] =
		{
			{ 0x0001, true },
			{ 0x0002, true },
			{ 0x0004, true },
			{ 0x0000, true },
			{ 0x0000, true },
			{ 0x0003, false }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::uint16>); i < n; ++i)
		{
			glm::vec<L, bool> const Result = glm::isPowerOfTwo(glm::vec<L, glm::uint16>(Data[i].Value));
			Error += glm::vec<L, bool>(Data[i].Return) == Result ? 0 : 1;
		}

		return Error;
	}

	template <glm::length_t L>
	static int test_int32()
	{
		type<int> const Data[] =
		{
			{ 0x00000001, true },
			{ 0x00000002, true },
			{ 0x00000004, true },
			{ 0x0000000f, false },
			{ 0x00000000, true },
			{ 0x00000003, false }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<int>); i < n; ++i)
		{
			glm::vec<L, bool> const Result = glm::isPowerOfTwo(glm::vec<L, glm::int32>(Data[i].Value));
			Error += glm::vec<L, bool>(Data[i].Return) == Result ? 0 : 1;
		}

		return Error;
	}

	template <glm::length_t L>
	static int test_uint32()
	{
		type<glm::uint> const Data[] =
		{
			{ 0x00000001, true },
			{ 0x00000002, true },
			{ 0x00000004, true },
			{ 0x80000000, true },
			{ 0x00000000, true },
			{ 0x00000003, false }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::uint>); i < n; ++i)
		{
			glm::vec<L, bool> const Result = glm::isPowerOfTwo(glm::vec<L, glm::uint32>(Data[i].Value));
			Error += glm::vec<L, bool>(Data[i].Return) == Result ? 0 : 1;
		}

		return Error;
	}

	static int test()
	{
		int Error = 0;

		Error += test_int16<1>();
		Error += test_int16<2>();
		Error += test_int16<3>();
		Error += test_int16<4>();

		Error += test_uint16<1>();
		Error += test_uint16<2>();
		Error += test_uint16<3>();
		Error += test_uint16<4>();

		Error += test_int32<1>();
		Error += test_int32<2>();
		Error += test_int32<3>();
		Error += test_int32<4>();

		Error += test_uint32<1>();
		Error += test_uint32<2>();
		Error += test_uint32<3>();
		Error += test_uint32<4>();

		return Error;
	}
}//isPowerOfTwo

namespace prevPowerOfTwo
{
	template <glm::length_t L, typename T>
	static int run()
	{
		int Error = 0;

		glm::vec<L, T> const A = glm::prevPowerOfTwo(glm::vec<L, T>(7));
		Error += A == glm::vec<L, T>(4) ? 0 : 1;

		glm::vec<L, T> const B = glm::prevPowerOfTwo(glm::vec<L, T>(15));
		Error += B == glm::vec<L, T>(8) ? 0 : 1;

		glm::vec<L, T> const C = glm::prevPowerOfTwo(glm::vec<L, T>(31));
		Error += C == glm::vec<L, T>(16) ? 0 : 1;

		glm::vec<L, T> const D = glm::prevPowerOfTwo(glm::vec<L, T>(32));
		Error += D == glm::vec<L, T>(32) ? 0 : 1;

		return Error;
	}

	static int test()
	{
		int Error = 0;

		Error += run<1, glm::int8>();
		Error += run<2, glm::int8>();
		Error += run<3, glm::int8>();
		Error += run<4, glm::int8>();

		Error += run<1, glm::int16>();
		Error += run<2, glm::int16>();
		Error += run<3, glm::int16>();
		Error += run<4, glm::int16>();

		Error += run<1, glm::int32>();
		Error += run<2, glm::int32>();
		Error += run<3, glm::int32>();
		Error += run<4, glm::int32>();

		Error += run<1, glm::int64>();
		Error += run<2, glm::int64>();
		Error += run<3, glm::int64>();
		Error += run<4, glm::int64>();

		Error += run<1, glm::uint8>();
		Error += run<2, glm::uint8>();
		Error += run<3, glm::uint8>();
		Error += run<4, glm::uint8>();

		Error += run<1, glm::uint16>();
		Error += run<2, glm::uint16>();
		Error += run<3, glm::uint16>();
		Error += run<4, glm::uint16>();

		Error += run<1, glm::uint32>();
		Error += run<2, glm::uint32>();
		Error += run<3, glm::uint32>();
		Error += run<4, glm::uint32>();

		Error += run<1, glm::uint64>();
		Error += run<2, glm::uint64>();
		Error += run<3, glm::uint64>();
		Error += run<4, glm::uint64>();

		return Error;
	}
}//namespace prevPowerOfTwo

namespace nextPowerOfTwo
{
	template <glm::length_t L, typename T>
	static int run()
	{
		int Error = 0;

		glm::vec<L, T> const A = glm::nextPowerOfTwo(glm::vec<L, T>(7));
		Error += A == glm::vec<L, T>(8) ? 0 : 1;

		glm::vec<L, T> const B = glm::nextPowerOfTwo(glm::vec<L, T>(15));
		Error += B == glm::vec<L, T>(16) ? 0 : 1;

		glm::vec<L, T> const C = glm::nextPowerOfTwo(glm::vec<L, T>(31));
		Error += C == glm::vec<L, T>(32) ? 0 : 1;

		glm::vec<L, T> const D = glm::nextPowerOfTwo(glm::vec<L, T>(32));
		Error += D == glm::vec<L, T>(32) ? 0 : 1;

		return Error;
	}

	static int test()
	{
		int Error = 0;

		Error += run<1, glm::int8>();
		Error += run<2, glm::int8>();
		Error += run<3, glm::int8>();
		Error += run<4, glm::int8>();

		Error += run<1, glm::int16>();
		Error += run<2, glm::int16>();
		Error += run<3, glm::int16>();
		Error += run<4, glm::int16>();

		Error += run<1, glm::int32>();
		Error += run<2, glm::int32>();
		Error += run<3, glm::int32>();
		Error += run<4, glm::int32>();

		Error += run<1, glm::int64>();
		Error += run<2, glm::int64>();
		Error += run<3, glm::int64>();
		Error += run<4, glm::int64>();

		Error += run<1, glm::uint8>();
		Error += run<2, glm::uint8>();
		Error += run<3, glm::uint8>();
		Error += run<4, glm::uint8>();

		Error += run<1, glm::uint16>();
		Error += run<2, glm::uint16>();
		Error += run<3, glm::uint16>();
		Error += run<4, glm::uint16>();

		Error += run<1, glm::uint32>();
		Error += run<2, glm::uint32>();
		Error += run<3, glm::uint32>();
		Error += run<4, glm::uint32>();

		Error += run<1, glm::uint64>();
		Error += run<2, glm::uint64>();
		Error += run<3, glm::uint64>();
		Error += run<4, glm::uint64>();

		return Error;
	}
}//namespace nextPowerOfTwo

namespace prevMultiple
{
	template<typename genIUType>
	struct type
	{
		genIUType Source;
		genIUType Multiple;
		genIUType Return;
	};

	template <glm::length_t L, typename T>
	static int run()
	{
		type<T> const Data[] =
		{
			{ 8, 3, 6 },
			{ 7, 7, 7 }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<T>); i < n; ++i)
		{
			glm::vec<L, T> const Result0 = glm::prevMultiple(glm::vec<L, T>(Data[i].Source), Data[i].Multiple);
			Error += glm::vec<L, T>(Data[i].Return) == Result0 ? 0 : 1;

			glm::vec<L, T> const Result1 = glm::prevMultiple(glm::vec<L, T>(Data[i].Source), glm::vec<L, T>(Data[i].Multiple));
			Error += glm::vec<L, T>(Data[i].Return) == Result1 ? 0 : 1;
		}

		return Error;
	}

	static int test()
	{
		int Error = 0;

		Error += run<1, glm::int8>();
		Error += run<2, glm::int8>();
		Error += run<3, glm::int8>();
		Error += run<4, glm::int8>();

		Error += run<1, glm::int16>();
		Error += run<2, glm::int16>();
		Error += run<3, glm::int16>();
		Error += run<4, glm::int16>();

		Error += run<1, glm::int32>();
		Error += run<2, glm::int32>();
		Error += run<3, glm::int32>();
		Error += run<4, glm::int32>();

		Error += run<1, glm::int64>();
		Error += run<2, glm::int64>();
		Error += run<3, glm::int64>();
		Error += run<4, glm::int64>();

		Error += run<1, glm::uint8>();
		Error += run<2, glm::uint8>();
		Error += run<3, glm::uint8>();
		Error += run<4, glm::uint8>();

		Error += run<1, glm::uint16>();
		Error += run<2, glm::uint16>();
		Error += run<3, glm::uint16>();
		Error += run<4, glm::uint16>();

		Error += run<1, glm::uint32>();
		Error += run<2, glm::uint32>();
		Error += run<3, glm::uint32>();
		Error += run<4, glm::uint32>();

		Error += run<1, glm::uint64>();
		Error += run<2, glm::uint64>();
		Error += run<3, glm::uint64>();
		Error += run<4, glm::uint64>();

		return Error;
	}
}//namespace prevMultiple

namespace nextMultiple
{
	template<typename genIUType>
	struct type
	{
		genIUType Source;
		genIUType Multiple;
		genIUType Return;
	};

	template <glm::length_t L, typename T>
	static int run()
	{
		type<T> const Data[] =
		{
			{ 3, 4, 4 },
			{ 6, 3, 6 },
			{ 5, 3, 6 },
			{ 7, 7, 7 },
			{ 0, 1, 0 },
			{ 8, 3, 9 }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<T>); i < n; ++i)
		{
			glm::vec<L, T> const Result0 = glm::nextMultiple(glm::vec<L, T>(Data[i].Source), glm::vec<L, T>(Data[i].Multiple));
			Error += glm::vec<L, T>(Data[i].Return) == Result0 ? 0 : 1;

			glm::vec<L, T> const Result1 = glm::nextMultiple(glm::vec<L, T>(Data[i].Source), Data[i].Multiple);
			Error += glm::vec<L, T>(Data[i].Return) == Result1 ? 0 : 1;
		}

		return Error;
	}

	static int test()
	{
		int Error = 0;

		Error += run<1, glm::int8>();
		Error += run<2, glm::int8>();
		Error += run<3, glm::int8>();
		Error += run<4, glm::int8>();

		Error += run<1, glm::int16>();
		Error += run<2, glm::int16>();
		Error += run<3, glm::int16>();
		Error += run<4, glm::int16>();

		Error += run<1, glm::int32>();
		Error += run<2, glm::int32>();
		Error += run<3, glm::int32>();
		Error += run<4, glm::int32>();

		Error += run<1, glm::int64>();
		Error += run<2, glm::int64>();
		Error += run<3, glm::int64>();
		Error += run<4, glm::int64>();

		Error += run<1, glm::uint8>();
		Error += run<2, glm::uint8>();
		Error += run<3, glm::uint8>();
		Error += run<4, glm::uint8>();

		Error += run<1, glm::uint16>();
		Error += run<2, glm::uint16>();
		Error += run<3, glm::uint16>();
		Error += run<4, glm::uint16>();

		Error += run<1, glm::uint32>();
		Error += run<2, glm::uint32>();
		Error += run<3, glm::uint32>();
		Error += run<4, glm::uint32>();

		Error += run<1, glm::uint64>();
		Error += run<2, glm::uint64>();
		Error += run<3, glm::uint64>();
		Error += run<4, glm::uint64>();

		return Error;
	}
}//namespace nextMultiple

namespace findNSB
{
#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic push
#	pragma clang diagnostic ignored "-Wpadded"
#endif

	template<typename T>
	struct type
	{
		T Source;
		int SignificantBitCount;
		int Return;
	};

#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic pop
#endif

	template <glm::length_t L, typename T>
	static int run()
	{
		type<T> const Data[] =
		{
			{ 0x00, 1,-1 },
			{ 0x01, 2,-1 },
			{ 0x02, 2,-1 },
			{ 0x06, 3,-1 },
			{ 0x01, 1, 0 },
			{ 0x03, 1, 0 },
			{ 0x03, 2, 1 },
			{ 0x07, 2, 1 },
			{ 0x05, 2, 2 },
			{ 0x0D, 2, 2 }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<T>); i < n; ++i)
		{
			glm::vec<L, int> const Result0 = glm::findNSB<L, T, glm::defaultp>(glm::vec<L, T>(Data[i].Source), glm::vec<L, int>(Data[i].SignificantBitCount));
			Error += glm::vec<L, int>(Data[i].Return) == Result0 ? 0 : 1;
			assert(!Error);
		}

		return Error;
	}

	static int test()
	{
		int Error = 0;

		Error += run<1, glm::uint8>();
		Error += run<2, glm::uint8>();
		Error += run<3, glm::uint8>();
		Error += run<4, glm::uint8>();

		Error += run<1, glm::uint16>();
		Error += run<2, glm::uint16>();
		Error += run<3, glm::uint16>();
		Error += run<4, glm::uint16>();

		Error += run<1, glm::uint32>();
		Error += run<2, glm::uint32>();
		Error += run<3, glm::uint32>();
		Error += run<4, glm::uint32>();

		Error += run<1, glm::uint64>();
		Error += run<2, glm::uint64>();
		Error += run<3, glm::uint64>();
		Error += run<4, glm::uint64>();

		Error += run<1, glm::int8>();
		Error += run<2, glm::int8>();
		Error += run<3, glm::int8>();
		Error += run<4, glm::int8>();

		Error += run<1, glm::int16>();
		Error += run<2, glm::int16>();
		Error += run<3, glm::int16>();
		Error += run<4, glm::int16>();

		Error += run<1, glm::int32>();
		Error += run<2, glm::int32>();
		Error += run<3, glm::int32>();
		Error += run<4, glm::int32>();

		Error += run<1, glm::int64>();
		Error += run<2, glm::int64>();
		Error += run<3, glm::int64>();
		Error += run<4, glm::int64>();


		return Error;
	}
}//namespace findNSB

#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic push
#	pragma clang diagnostic ignored "-Wpadded"
#endif

template<typename T, typename B>
struct test_mix_entry
{
	T x;
	T y;
	B a;
	T Result;
};

#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic pop
#endif

static int test_mix()
{
	test_mix_entry<int, bool> const TestBool[] =
	{
		{0, 1, false, 0},
		{0, 1, true, 1},
		{-1, 1, false, -1},
		{-1, 1, true, 1}
	};

	test_mix_entry<int, int> const TestInt[] =
	{
		{0, 1, 0, 0},
		{0, 1, 1, 1},
		{-1, 1, 0, -1},
		{-1, 1, 1, 1}
	};

	test_mix_entry<glm::ivec2, bool> const TestVec2Bool[] =
	{
		{glm::ivec2(0), glm::ivec2(1), false, glm::ivec2(0)},
		{glm::ivec2(0), glm::ivec2(1), true, glm::ivec2(1)},
		{glm::ivec2(-1), glm::ivec2(1), false, glm::ivec2(-1)},
		{glm::ivec2(-1), glm::ivec2(1), true, glm::ivec2(1)}
	};

	test_mix_entry<glm::ivec2, glm::bvec2> const TestBVec2[] =
	{
		{glm::ivec2(0), glm::ivec2(1), glm::bvec2(false), glm::ivec2(0)},
		{glm::ivec2(0), glm::ivec2(1), glm::bvec2(true), glm::ivec2(1)},
		{glm::ivec2(-1), glm::ivec2(1), glm::bvec2(false), glm::ivec2(-1)},
		{glm::ivec2(-1), glm::ivec2(1), glm::bvec2(true), glm::ivec2(1)},
		{glm::ivec2(-1), glm::ivec2(1), glm::bvec2(true, false), glm::ivec2(1, -1)}
	};

	test_mix_entry<glm::ivec3, bool> const TestVec3Bool[] =
	{
		{glm::ivec3(0), glm::ivec3(1), false, glm::ivec3(0)},
		{glm::ivec3(0), glm::ivec3(1), true, glm::ivec3(1)},
		{glm::ivec3(-1), glm::ivec3(1), false, glm::ivec3(-1)},
		{glm::ivec3(-1), glm::ivec3(1), true, glm::ivec3(1)}
	};

	test_mix_entry<glm::ivec3, glm::bvec3> const TestBVec3[] =
	{
		{glm::ivec3(0), glm::ivec3(1), glm::bvec3(false), glm::ivec3(0)},
		{glm::ivec3(0), glm::ivec3(1), glm::bvec3(true), glm::ivec3(1)},
		{glm::ivec3(-1), glm::ivec3(1), glm::bvec3(false), glm::ivec3(-1)},
		{glm::ivec3(-1), glm::ivec3(1), glm::bvec3(true), glm::ivec3(1)},
		{glm::ivec3(1, 2, 3), glm::ivec3(4, 5, 6), glm::bvec3(true, false, true), glm::ivec3(4, 2, 6)}
	};

	test_mix_entry<glm::ivec4, bool> const TestVec4Bool[] = 
	{
		{glm::ivec4(0), glm::ivec4(1), false, glm::ivec4(0)},
		{glm::ivec4(0), glm::ivec4(1), true, glm::ivec4(1)},
		{glm::ivec4(-1), glm::ivec4(1), false, glm::ivec4(-1)},
		{glm::ivec4(-1), glm::ivec4(1), true, glm::ivec4(1)}
	};

	test_mix_entry<glm::ivec4, glm::bvec4> const TestBVec4[] = 
	{
		{glm::ivec4(0, 0, 1, 1), glm::ivec4(2, 2, 3, 3), glm::bvec4(false, true, false, true), glm::ivec4(0, 2, 1, 3)},
		{glm::ivec4(0), glm::ivec4(1), glm::bvec4(true), glm::ivec4(1)},
		{glm::ivec4(-1), glm::ivec4(1), glm::bvec4(false), glm::ivec4(-1)},
		{glm::ivec4(-1), glm::ivec4(1), glm::bvec4(true), glm::ivec4(1)},
		{glm::ivec4(1, 2, 3, 4), glm::ivec4(5, 6, 7, 8), glm::bvec4(true, false, true, false), glm::ivec4(5, 2, 7, 4)}
	};

	int Error = 0;

	// Float with bool
	{
		for(std::size_t i = 0; i < sizeof(TestBool) / sizeof(test_mix_entry<int, bool>); ++i)
		{
			int const Result = glm::mix(TestBool[i].x, TestBool[i].y, TestBool[i].a);
			Error += Result == TestBool[i].Result ? 0 : 1;
		}
	}

	// Float with float
	{
		for(std::size_t i = 0; i < sizeof(TestInt) / sizeof(test_mix_entry<int, int>); ++i)
		{
			int const Result = glm::mix(TestInt[i].x, TestInt[i].y, TestInt[i].a);
			Error += Result == TestInt[i].Result ? 0 : 1;
		}
	}

	// vec2 with bool
	{
		for(std::size_t i = 0; i < sizeof(TestVec2Bool) / sizeof(test_mix_entry<glm::ivec2, bool>); ++i)
		{
			glm::ivec2 const Result = glm::mix(TestVec2Bool[i].x, TestVec2Bool[i].y, TestVec2Bool[i].a);
			Error += glm::all(glm::equal(Result, TestVec2Bool[i].Result)) ? 0 : 1;
		}
	}

	// vec2 with bvec2
	{
		for(std::size_t i = 0; i < sizeof(TestBVec2) / sizeof(test_mix_entry<glm::ivec2, glm::bvec2>); ++i)
		{
			glm::ivec2 const Result = glm::mix(TestBVec2[i].x, TestBVec2[i].y, TestBVec2[i].a);
			Error += glm::all(glm::equal(Result, TestBVec2[i].Result)) ? 0 : 1;
		}
	}

	// vec3 with bool
	{
		for(std::size_t i = 0; i < sizeof(TestVec3Bool) / sizeof(test_mix_entry<glm::ivec3, bool>); ++i)
		{
			glm::ivec3 const Result = glm::mix(TestVec3Bool[i].x, TestVec3Bool[i].y, TestVec3Bool[i].a);
			Error += glm::all(glm::equal(Result, TestVec3Bool[i].Result)) ? 0 : 1;
		}
	}

	// vec3 with bvec3
	{
		for(std::size_t i = 0; i < sizeof(TestBVec3) / sizeof(test_mix_entry<glm::ivec3, glm::bvec3>); ++i)
		{
			glm::ivec3 const Result = glm::mix(TestBVec3[i].x, TestBVec3[i].y, TestBVec3[i].a);
			Error += glm::all(glm::equal(Result, TestBVec3[i].Result)) ? 0 : 1;
		}
	}

	// vec4 with bool
	{
		for(std::size_t i = 0; i < sizeof(TestVec4Bool) / sizeof(test_mix_entry<glm::ivec4, bool>); ++i)
		{
			glm::ivec4 const Result = glm::mix(TestVec4Bool[i].x, TestVec4Bool[i].y, TestVec4Bool[i].a);
			Error += glm::all(glm::equal(Result, TestVec4Bool[i].Result)) ? 0 : 1;
		}
	}

	// vec4 with bvec4
	{
		for(std::size_t i = 0; i < sizeof(TestBVec4) / sizeof(test_mix_entry<glm::ivec4, glm::bvec4>); ++i)
		{
			glm::ivec4 const Result = glm::mix(TestBVec4[i].x, TestBVec4[i].y, TestBVec4[i].a);
			Error += glm::all(glm::equal(Result, TestBVec4[i].Result)) ? 0 : 1;
		}
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += isPowerOfTwo::test();
	Error += prevPowerOfTwo::test();
	Error += nextPowerOfTwo::test();
	Error += prevMultiple::test();
	Error += nextMultiple::test();
	Error += findNSB::test();

	Error += test_mix();

	return Error;
}
