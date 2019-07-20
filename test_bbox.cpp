#include "bbox.h"

#include <tut/tut.hpp>

#include <iostream>
#include <array>

namespace tut
{
	struct bbox_test_data
	{
		using point2d_t = std::array<double, 2>;
		using bbox2d_t = bbox<point2d_t>;
	};

	using bbox_test_t = test_group<bbox_test_data, 10>;
	bbox_test_t bbox_tests("bbox tests");

	template<> template<>
	void bbox_test_t::object::test<1>()
	{
		set_test_name("Intersects");

		bbox2d_t bbox1( {-4,-1}, {6, 6} );
		bbox2d_t bbox2( {-1,-3}, {4, 7} );

		ensure(bbox1.intersects(bbox2));
		ensure(bbox2.intersects(bbox1));
	}

	template <> template <>
	void bbox_test_t::object::test<2>()
	{
		set_test_name("Disjoint");

		bbox2d_t bbox1( {-4, -8}, {-2, 2} );
		bbox2d_t bbox2( { 2, 0 }, { 4, 4} );

		ensure(bbox1.disjoint(bbox2));
		ensure(bbox2.disjoint(bbox1));
	}

	template <> template <>
	void bbox_test_t::object::test<3>()
	{
		set_test_name("Contains");

		bbox2d_t bbox({2, 0} , { 4, 4});
		ensure(bbox.contains({2, 2}));
		ensure(!bbox.contains({-2, 2}));
	}

	template <> template<>
	void bbox_test_t::object::test<4>()
	{
		set_test_name("Intersection");

		bbox2d_t bbox1( {-4,-1}, { 6, 6} );
		bbox2d_t bbox2( {-1,-3}, { 4, 7} );

		bbox2d_t bbox1_2({1, 1}, {-1, -1});
		ensure(bbox1.intersection(bbox2, bbox1_2));
		ensure(bbox1_2 == bbox2d_t({-1, -1}, {4, 6}));
	}
};


























