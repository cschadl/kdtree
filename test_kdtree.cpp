#include "kdtree.h"

#include <tut/tut.hpp>

#include <iomanip>
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

		static std::string make_test_knn_err_str(
			point3d_t const & q,
			point3d_t const & pt_expected,
			point3d_t const & pt_actual,
			size_t i, size_t j,
			double dist_q_p_nq,
			double dist_q_nn_j)
		{
			std::ostringstream error_oss;
			error_oss << "Point " << std::setprecision(4)
						<< "(" << q[0] << ", " << q[1] << ", " << q[2] << ") (i: " << i << ") nn " << j
						<< ", Expected: "
						<< "(" << pt_expected[0] << "," << pt_expected[1] << "," << pt_expected[2] << ") (dist: " << std::setprecision(6) << dist_q_p_nq
						<< "), got: (" << std::setprecision(4) << pt_actual[0] << ", " << pt_actual[1] << ", " << pt_actual[2] << ") (dist: "
						<< std::setprecision(6) << dist_q_nn_j << ")";
			
			return error_oss.str();
		}

		static void test_knn(point3d_t const& q,
									std::vector<point3d_t> const& points,
									size_t n_neighbors,
									std::vector<point3d_t> const& nn_pts,
									size_t i, size_t seed)
		{
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
					make_test_knn_err_str(q, p_nq, nn_pts[j], i, j, dist_q_p_nq, dist_q_nn_j),
					std::abs(dist_q_p_nq - dist_q_nn_j) < std::numeric_limits<double>::epsilon());

				j++;
			}
		}
	};

	using kdtree_test_t = test_group<kdtree_test_data>;
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
		std::uniform_real_distribution<double> rand_pt(-10.0, 10.0);

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

		ensure(tree.last_q_nodes_visited() < n_pts / 3);
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
		std::uniform_real_distribution<double> rand_pt(-10.0, 10.0);

		for (size_t i = 0 ; i < n_pts ; i++)
			points[i] = point3d_t{ rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator) };

		kd_tree<point3d_t> tree;
		tree.build(points.begin(), points.end());

		const size_t n_q_pts = 100;
		const size_t n_neighbors = 5;

		std::vector<size_t> seeds = {
			0xf3bd3f842d4fab01,
			0xe847ab01d3f48a49,
			0xdeadbeefdeadbeef,
			0xd5ba7284eef00fee
		};

		size_t total_nodes_visited = 0;

		for (size_t seed : seeds)
		{
			std::mt19937_64 q_pt_generator(seed);
			for (size_t i = 0 ; i < n_q_pts ; i++)
			{
				point3d_t q = { rand_pt(q_pt_generator), rand_pt(q_pt_generator), rand_pt(q_pt_generator) };

				std::vector<point3d_t> nn_pts = tree.k_nn(q, n_neighbors);

				total_nodes_visited += tree.last_q_nodes_visited();

				test_knn(q, points, n_neighbors, nn_pts, i, seed);
			}
		}
	}

	template <> template <>
	void kdtree_test_t::object::test<6>()
	{
		set_test_name("duplicate points");

		const size_t n_pts = 100;
		const size_t n_duplicates = 15;

		std::vector<point3d_t> points(n_pts + n_duplicates);

		std::mt19937_64 pt_generator(0xdeaf3ba3de950afb);
		std::uniform_real_distribution<double> rand_pt(-10.0, 10.0);

		for (size_t i = 0 ; i < n_pts ; i++)
			points[i] = point3d_t{ rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator) };

		// Just use the 1st n_duplicates as our duplicates
		for (size_t i = 0 ; i < n_duplicates ; i++)
			points[n_pts + i] = points[i];

		// Shuffle, for the hell of it
		auto points_shuffled = points;
		std::mt19937_64 shuffle_gen(0xdeadbeefdeadbeef);
		std::shuffle(points_shuffled.begin(), points_shuffled.end(), shuffle_gen);

		// Build the tree
		kd_tree<point3d_t> tree;
		tree.build(points_shuffled.begin(), points_shuffled.end());

		// Make sure we can query it
		size_t n_q_pts = 20;
		size_t n_neighbors = 5;
		for (size_t i = 0 ; i < n_q_pts ; i++)
		{
			point3d_t q = { rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator) };

			std::vector<point3d_t> nn_pts = tree.k_nn(q, n_neighbors);
			test_knn(q, points, n_neighbors, nn_pts, i, 0);
		}

		// Can we find our duplicate points?
		for (size_t i = 0 ; i < n_duplicates ; i++)
		{
			point3d_t dup_nn = tree.nn(points[i]);
			ensure(dist(dup_nn, points[i]) <= std::numeric_limits<double>::epsilon());
		}
	}

	template <> template <>
	void kdtree_test_t::object::test<7>()
	{
		set_test_name("range search 2d");

		std::vector<point2d_t> points = {
			{ 2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}
		};

		kd_tree<point2d_t> tree;
		tree.build(points.begin(), points.end());

		bbox<point2d_t> search_bbox({3, 2}, {7.5, 8});
		std::vector<point2d_t> range_pts = tree.range_search(search_bbox);

		ensure(range_pts.size() == 3);

		ensure(std::find(range_pts.begin(), range_pts.end(), point2d_t{7, 2}) != range_pts.end());
		ensure(std::find(range_pts.begin(), range_pts.end(), point2d_t{4, 7}) != range_pts.end());
		ensure(std::find(range_pts.begin(), range_pts.end(), point2d_t{5, 4}) != range_pts.end());
	}

	template <> template <>
	void kdtree_test_t::object::test<8>()
	{
		set_test_name("range search 3d");

		using bbox_t = bbox<point3d_t>;

		const size_t n_pts = 1000;
		std::vector<point3d_t> points(n_pts);

		std::mt19937_64 pt_generator(0x19efa8471bb936a0);
		std::uniform_real_distribution<double> rand_pt(-10.0, 10.0);

		for (size_t i = 0 ; i < n_pts ; i++)
			points[i] = point3d_t{ rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator) };

		kd_tree<point3d_t> tree;
		tree.build(points.begin(), points.end());

		point3d_t pt_min{-2, -2, -2};
		point3d_t pt_max{ 2,  2,  2};
		bbox_t search_bbox(pt_min, pt_max);
		std::vector<point3d_t> points_in_range = tree.range_search(search_bbox);

		ensure("Search returned empty set", !points_in_range.empty());

		for (point3d_t const& p : points)
		{
			if (p[0] >= pt_min[0] && p[1] >= pt_min[1] && p[2] >= pt_min[2]
				 && p[0] <= pt_max[0] && p[1] <= pt_max[1] && p[2] <= pt_max[2])
			{
				ensure(
					"Point not found in search results",
					std::find(points_in_range.begin(), points_in_range.end(), p) != points_in_range.end());
			}
		}

		size_t const nodes_visited = tree.last_q_nodes_visited();
		ensure("Too many nodes visited", nodes_visited < 100);
	}
};











































