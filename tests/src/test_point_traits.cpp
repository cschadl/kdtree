#include <kdtree/point_traits.hpp>

#include <tut/tut.hpp>

using namespace cds::kdtree;

namespace tut
{
	struct test_point_traits_data
	{
		using point2d_t = std::array<double, 2>;
		using point3d_t = std::array<double, 3>;
	};

	using test_point_traits_t = test_group<test_point_traits_data, 10>;
	test_point_traits_t test_point_traits("point traits");

	template <> template <>
	void test_point_traits_t::object::test<1>()
	{
		set_test_name("point_traits::create()");

		point3d_t point = point_traits<point3d_t>::create(12.0);
		ensure(point[0] == 12.0);
		ensure(point[1] == 12.0);
		ensure(point[2] == 12.0);
	}
};
