#include "kdtree.h"

#include <tut/tut.hpp>

#include <boost/format.hpp>

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
		ensure(tree.last_q_nodes_visited() < n_pts);
	}

	template <> template <>
	void kdtree_test_t::object::test<4>()
	{
		set_test_name("knn search (2d)");

		std::vector<point2d_t> points = {
			{ 2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}
		};

		point2d_t const q = { 4, 2 };

		kd_tree<point2d_t> tree;
		tree.build(points.begin(), points.end());

		auto knn_3 = tree.k_nn(q, 3);
		ensure(knn_3[0] == point2d_t{ 2, 3 });
		ensure(knn_3[1] == point2d_t{ 5, 4 });
		ensure(knn_3[2] == point2d_t{ 7, 2 });
	}

	template <> template <>
	void kdtree_test_t::object::test<5>()
	{
		set_test_name("knn search (3d)");

		const size_t n_pts = 1000;
		std::vector<point3d_t> points(n_pts);

		std::mt19937_64 pt_generator(0xfeebdaedfeebdaed);
		std::uniform_real_distribution<double> rand_pt(-1.0, 1.0);

		for (size_t i = 0 ; i < n_pts ; i++)
			points[i] = point3d_t{ rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator) };

		kd_tree<point3d_t> tree;
		tree.build(points.begin(), points.end());

		const size_t n_q_pts = 100;
		const size_t n_neighbors = 1;	// Fails if n_neighbors > 1

		// FAILS with 0x38fab2d4ef482a42
		std::mt19937_64 q_pt_generator(0xfeebdaedfeebdaed);
		for (size_t i = 0 ; i < n_q_pts ; i++)
		{
			point3d_t q = { rand_pt(q_pt_generator), rand_pt(q_pt_generator), rand_pt(q_pt_generator) };

			std::vector<point3d_t> nn_pts = tree.k_nn(q, n_neighbors);

			auto cmp_dist_q = [&q](point3d_t const& p1, point3d_t const& p2)
			{
				return dist(p1, q) > dist(p2, q);	// min priority queue, so reversed
			};

			fixed_priority_queue<point3d_t, decltype(cmp_dist_q)> nn_min_pq(n_neighbors, cmp_dist_q);
			for (point3d_t const& p : points)
				nn_min_pq.push(p);

			size_t j = 0;
			while (!nn_min_pq.empty())
			{
				point3d_t const p_nq = nn_min_pq.top();
				nn_min_pq.pop();

				double const dist_q_p_nq = dist(q, p_nq);
				double const dist_q_nn_j = dist(q, nn_pts[j]);

				ensure(
					(boost::format("Point (%.4f, %.4f, %.4f) (i: %u) nn %d, Expected: (%.4f, %.4f, %.4f) (dist %.6f), got: (%.4f, %.4f, %.4f) (dist %.6f)") %
							q[0] % q[1] % q[2] % i % j %
							p_nq[0] % p_nq[1] % p_nq[2] % dist_q_p_nq %
							nn_pts[j][0] % nn_pts[j][1] % nn_pts[j][2] % dist_q_nn_j).str(),
					nn_pts[j] == p_nq || abs(dist_q_p_nq - dist_q_nn_j) < 1.0e-12);

				j++;
			}
		}
	}
};











