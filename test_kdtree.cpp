#include "kdtree.h"

#include <tut/tut.hpp>

#include <iostream>
#include <array>
#include <vector>
#include <limits>
#include <random>

namespace tut
{
	struct kdtree_test_data
	{
		using point2d_t = std::array<double, 2>;
		using point3d_t = std::array<double, 3>;

		static constexpr double min_val = std::numeric_limits<double>::min();
		static constexpr double max_val = std::numeric_limits<double>::max();

		template <size_t N>
		static double dist(std::array<double, N> const& p1, std::array<double, N> const& p2)
		{
			double dist_sq = 0;
			for (size_t i = 0 ; i < N ; i++)
			{
				double dp1p2_i = p1[i] - p2[i];
				dist_sq += (dp1p2_i * dp1p2_i);
			}

			return std::sqrt(dist_sq);
		}
	};

	using kdtree_test_t = test_group<kdtree_test_data, 5>;
	kdtree_test_t kdtree_test("kd_tree tests");

	template <> template <>
	void kdtree_test_t::object::test<1>()
	{
		set_test_name("Build kdtree");

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
		set_test_name("Basic query kdtree");

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

	template <> template <>
	void kdtree_test_t::object::test<3>()
	{
		set_test_name("Complex query");

		const size_t n_pts = 100;
		std::vector<point3d_t> points(n_pts);

		size_t const random_seed = 0xdeadbeefdeadbeef;
		std::mt19937_64 pt_generator(random_seed);
		std::uniform_real_distribution<double> rand_pt(-1.0, 1.0);

		for (size_t i = 0 ; i < n_pts ; i++)
			points[i] = point3d_t{ rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator) };

		kd_tree<point3d_t> tree;
		tree.build(points.begin(), points.end());

		point3d_t const test_pt = {0.5, -0.25, 1.0};
		point3d_t const nn_kdtree = tree.nn(test_pt);

		point3d_t const nn_actual = *std::min_element(points.begin(), points.end(),
			[&test_pt](point3d_t const& p1, point3d_t const& p2)
			{
				return dist(p1, test_pt) < dist(p2, test_pt);
			});

		ensure(nn_kdtree == nn_actual);
	}
};











