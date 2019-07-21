#include "kdtree.h"

#include <tut/tut.hpp>

#include <iostream>
#include <array>
#include <vector>
#include <limits>

namespace tut
{
	struct kdtree_test_data
	{
		using point2d_t = std::array<double, 2>;

		static constexpr double min_val = std::numeric_limits<double>::min();
		static constexpr double max_val = std::numeric_limits<double>::max();
	};

	using kdtree_test_t = test_group<kdtree_test_data, 5>;
	kdtree_test_t kdtree_test("kd_tree tests");

	template <> template <>
	void kdtree_test_t::object::test<1>()
	{
		set_test_name("build kdtree");

		std::vector<point2d_t> points = {
			{ 2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}
		};

		kd_tree<point2d_t> tree;
		tree.build(points.begin(), points.end());

		ensure(true);	// TODO - test more shit here about the tree that was built
	}

	template <> template <>
	void kdtree_test_t::object::test<2>()
	{
		set_test_name("Query kdtree");

		std::vector<point2d_t> points = {
			{ 2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}
		};

		kd_tree<point2d_t> tree;
		tree.build(points.begin(), points.end());

		point2d_t q_pt1{7.5, 5.0};
		point2d_t near_q1 = tree.nn(q_pt1);

		ensure(near_q1 == point2d_t{9, 6});

		point2d_t q_pt2{7.5, 0.5};
		point2d_t near_q2 = tree.nn(q_pt2);

		ensure(near_q2 == point2d_t{8, 1});
	}
};
