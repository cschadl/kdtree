#include <gtest/gtest.h>

#include <kdtree/kdtree.hpp>
#include <kdtree/detail/fixed_priority_queue.hpp>

#include <iomanip>
#include <iostream>
#include <array>
#include <vector>
#include <limits>
#include <random>
#include <optional>

using namespace cds::kdtree;

namespace
{
	using point2d_t = std::array<double, 2>;
	using point3d_t = std::array<double, 3>;

	static constexpr double min_val = std::numeric_limits<double>::min();
	static constexpr double max_val = std::numeric_limits<double>::max();

	template <size_t N>
	static double dist_sq(std::array<double, N> const &p1, std::array<double, N> const &p2)
	{
		double dist_sq = 0;
		for (size_t i = 0; i < N; i++)
		{
			double dp1p2_i = p1[i] - p2[i];
			dist_sq += (dp1p2_i * dp1p2_i);
		}

		return dist_sq;
	}

	template <size_t N>
	double dist(std::array<double, N> const &p1, std::array<double, N> const &p2)
	{
		return std::sqrt(dist_sq(p1, p2));
	}

	struct knn_search_fail
	{
		point3d_t q;
		point3d_t pt_expected;
		point3d_t pt_actual;
		size_t i;
		size_t j;
		double dist_q_p_nq;
		double dist_q_nn_j;
	};

	std::string make_test_knn_err_str(knn_search_fail &fail)
	{
		std::ostringstream error_oss;
		error_oss << "Point " << std::setprecision(4)
					 << "(" << fail.q[0] << ", " << fail.q[1] << ", " << fail.q[2] << ") (i: " << fail.i << ") nn " << fail.j
					 << ", Expected: "
					 << "(" << fail.pt_expected[0] << "," << fail.pt_expected[1] << "," << fail.pt_expected[2]
					 << ") (dist: " << std::setprecision(6) << fail.dist_q_p_nq
					 << "), got: (" << std::setprecision(4)
					 << fail.pt_actual[0] << ", " << fail.pt_actual[1] << ", " << fail.pt_actual[2] << ") (dist: "
					 << std::setprecision(6) << fail.dist_q_nn_j << ")";

		return error_oss.str();
	}

	std::optional<knn_search_fail> test_knn(
		 point3d_t const &q,
		 std::vector<point3d_t> const &points,
		 size_t n_neighbors,
		 std::vector<point3d_t> const &nn_pts,
		 size_t i, size_t seed)
	{
		auto cmp_dist_q = [&q](point3d_t const &p1, point3d_t const &p2)
		{
			return dist(p1, q) > dist(p2, q); // min priority queue, so reversed
		};

		detail_::fixed_priority_queue<point3d_t, decltype(cmp_dist_q)> nn_min_pq(n_neighbors, cmp_dist_q);
		for (point3d_t const &p : points)
			nn_min_pq.push(p);

		size_t j = 0;
		while (!nn_min_pq.empty())
		{
			point3d_t const p_nq = nn_min_pq.top();
			nn_min_pq.pop();

			double const dist_q_p_nq = dist(q, p_nq);
			double const dist_q_nn_j = dist(q, nn_pts[j]);

			if (std::abs(dist_q_p_nq - dist_q_nn_j) > std::numeric_limits<double>::epsilon())
				return knn_search_fail{q, p_nq, nn_pts[j], i, j, dist_q_p_nq, dist_q_nn_j};

			j++;
		}

		return std::nullopt;
	}
}

TEST(kd_tree, build)
{
	std::vector<point2d_t> points = {
		 {2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}};

	kd_tree<point2d_t> tree;
	tree.build(points.begin(), points.end());

	point2d_t q_pt1{7.5, 5.0};
	point2d_t near_q1 = tree.nn(q_pt1);

	EXPECT_EQ(near_q1, point2d_t({9, 6}));

	point2d_t q_pt2{7.5, 0.5};
	point2d_t near_q2 = tree.nn(q_pt2);

	EXPECT_EQ(near_q2, point2d_t({8, 1}));
}

TEST(kd_tree, NNRandomPoints)
{
	const size_t n_pts = 100;
	std::vector<point3d_t> points(n_pts);

	size_t const random_seed = 0xdeadbeefdeadbeef;
	std::mt19937_64 pt_generator(random_seed);
	std::uniform_real_distribution<double> rand_pt(-10.0, 10.0);

	for (size_t i = 0; i < n_pts; i++)
		points[i] = point3d_t{rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator)};

	kd_tree<point3d_t> tree;
	tree.build(points.begin(), points.end());

	point3d_t const test_pt = {0.5, -0.25, 1.0};
	point3d_t const nn_kdtree = tree.nn(test_pt);

	point3d_t const nn_actual = *std::min_element(points.begin(), points.end(),
																 [&test_pt](point3d_t const &p1, point3d_t const &p2)
																 {
																	 return dist(p1, test_pt) < dist(p2, test_pt);
																 });

	EXPECT_EQ(nn_kdtree, nn_actual);

	EXPECT_LT(tree.last_q_nodes_visited(), n_pts / 3);
}

TEST(kd_tree, kNN2d)
{
	std::vector<point2d_t> points = {
		 {2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}};

	point2d_t const q = {4, 2};

	kd_tree<point2d_t> tree;
	tree.build(points.begin(), points.end());

	auto knn_3 = tree.k_nn(q, 3);
	EXPECT_EQ(knn_3[0], point2d_t({2, 3}));
	EXPECT_EQ(knn_3[1], point2d_t({5, 4}));
	EXPECT_EQ(knn_3[2], point2d_t({7, 2}));
}

TEST(kd_tree, kNN3d)
{
	const size_t n_pts = 1000;
	std::vector<point3d_t> points(n_pts);

	std::mt19937_64 pt_generator(0xfeebdaedfeebdaed);
	std::uniform_real_distribution<double> rand_pt(-10.0, 10.0);

	for (size_t i = 0; i < n_pts; i++)
		points[i] = point3d_t{rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator)};

	kd_tree<point3d_t> tree;
	tree.build(points.begin(), points.end());

	const size_t n_q_pts = 100;
	const size_t n_neighbors = 5;

	std::vector<size_t> seeds = {
		 0xf3bd3f842d4fab01,
		 0xe847ab01d3f48a49,
		 0xdeadbeefdeadbeef,
		 0xd5ba7284eef00fee};

	size_t total_nodes_visited = 0;

	for (size_t seed : seeds)
	{
		std::mt19937_64 q_pt_generator(seed);
		for (size_t i = 0; i < n_q_pts; i++)
		{
			point3d_t q = {rand_pt(q_pt_generator), rand_pt(q_pt_generator), rand_pt(q_pt_generator)};

			std::vector<point3d_t> nn_pts = tree.k_nn(q, n_neighbors);

			total_nodes_visited += tree.last_q_nodes_visited();

			auto failure = test_knn(q, points, n_neighbors, nn_pts, i, seed);
			EXPECT_FALSE(failure.has_value()) << make_test_knn_err_str(*failure);
		}
	}
}

TEST(kd_tree, DuplicatePoints)
{
	const size_t n_pts = 100;
	const size_t n_duplicates = 15;

	std::vector<point3d_t> points(n_pts + n_duplicates);

	std::mt19937_64 pt_generator(0xdeaf3ba3de950afb);
	std::uniform_real_distribution<double> rand_pt(-10.0, 10.0);

	for (size_t i = 0; i < n_pts; i++)
		points[i] = point3d_t{rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator)};

	// Just use the 1st n_duplicates as our duplicates
	for (size_t i = 0; i < n_duplicates; i++)
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
	for (size_t i = 0; i < n_q_pts; i++)
	{
		point3d_t q = {rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator)};

		std::vector<point3d_t> nn_pts = tree.k_nn(q, n_neighbors);

		auto fail = test_knn(q, points, n_neighbors, nn_pts, i, 0);
		EXPECT_FALSE(fail.has_value()) << make_test_knn_err_str(*fail);
	}

	// Can we find our duplicate points?
	for (size_t i = 0; i < n_duplicates; i++)
	{
		point3d_t dup_nn = tree.nn(points[i]);
		EXPECT_LE(dist(dup_nn, points[i]), std::numeric_limits<double>::epsilon());
	}
}

TEST(kd_tree, RangeSearch2d)
{
	std::vector<point2d_t> points = {
		 {2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}};

	kd_tree<point2d_t> tree;
	tree.build(points.begin(), points.end());

	bbox<point2d_t> search_bbox({3, 2}, {7.5, 8});
	std::vector<point2d_t> range_pts = tree.range_search(search_bbox);

	EXPECT_EQ(range_pts.size(), 3);

	EXPECT_NE(std::find(range_pts.begin(), range_pts.end(), point2d_t{7, 2}), range_pts.end());
	EXPECT_NE(std::find(range_pts.begin(), range_pts.end(), point2d_t{4, 7}), range_pts.end());
	EXPECT_NE(std::find(range_pts.begin(), range_pts.end(), point2d_t{5, 4}), range_pts.end());
}

TEST(kd_tree, RangeSearch3d)
{
	using bbox_t = bbox<point3d_t>;

	const size_t n_pts = 1000;
	std::vector<point3d_t> points(n_pts);

	std::mt19937_64 pt_generator(0x19efa8471bb936a0);
	std::uniform_real_distribution<double> rand_pt(-10.0, 10.0);

	for (size_t i = 0; i < n_pts; i++)
		points[i] = point3d_t{rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator)};

	kd_tree<point3d_t> tree;
	tree.build(points.begin(), points.end());

	point3d_t pt_min{-2, -2, -2};
	point3d_t pt_max{2, 2, 2};
	bbox_t search_bbox(pt_min, pt_max);
	std::vector<point3d_t> points_in_range = tree.range_search(search_bbox);

	EXPECT_FALSE(points_in_range.empty()) << "Search returned empty set";

	for (point3d_t const &p : points)
	{
		if (p[0] >= pt_min[0] && p[1] >= pt_min[1] && p[2] >= pt_min[2] && p[0] <= pt_max[0] && p[1] <= pt_max[1] && p[2] <= pt_max[2])
		{
			EXPECT_NE(std::find(points_in_range.begin(), points_in_range.end(), p), points_in_range.end()) 
				<< "Point not found in search results";
		}
	}

	size_t const nodes_visited = tree.last_q_nodes_visited();
	EXPECT_LT(nodes_visited, 100) << "Too many nodes visited";
}

TEST(kd_tree, RadiusSearch2d)
{
	point2d_t const test_pt{4, 3};

	std::vector<point2d_t> points = {
		 {3.1, 3.5},  // dist = 1.02
		 {3.0, 2.0},  // dist = 1.41
		 {4.0, 3.3},  // dist = 0.09
		 {4.25, 3.5}, // dist = 0.3125
		 {4.78, 1.7}, // dist = 1.516
		 {4.0, 1.2}	  // dist = 1.8
	};

	kd_tree<point2d_t> tree;
	tree.build(points.begin(), points.end());

	{
		auto near_pts1 = tree.radius_search(test_pt, 1.0);
		EXPECT_EQ(near_pts1.size(), 2);

		EXPECT_NE(std::find(near_pts1.begin(), near_pts1.end(), point2d_t{4.0, 3.3}), near_pts1.end())
			<< "point (4.0, 3.3) not found in search radius = 1";

		EXPECT_NE(std::find(near_pts1.begin(), near_pts1.end(), point2d_t{4.25, 3.5}), near_pts1.end())
			<< "point (4.25, 3.5) not found in search radius = 1";
	}

	{
		auto near_pts2 = tree.radius_search(test_pt, 1.25);
		EXPECT_EQ(near_pts2.size(), 3);

		EXPECT_NE(std::find(near_pts2.begin(), near_pts2.end(), point2d_t{4.0, 3.3}), near_pts2.end())
				  << "point (4.0, 3.3) not found in search radius = 1.25";

		EXPECT_NE(std::find(near_pts2.begin(), near_pts2.end(), point2d_t{4.25, 3.5}), near_pts2.end())
				  << "point (4.25, 3.5) not found in search radius = 1.25";

		EXPECT_NE(std::find(near_pts2.begin(), near_pts2.end(), point2d_t{3.1, 3.5}), near_pts2.end())
				  << "point (3.1, 3.5) not found in search radius = 1.25";
	}

	{
		auto near_pts3 = tree.radius_search(test_pt, 1.52);
		EXPECT_EQ(near_pts3.size(), 5);

		EXPECT_NE(std::find(near_pts3.begin(), near_pts3.end(), point2d_t{4.0, 3.3}), near_pts3.end())
				  << "point (4.0, 3.3) not found in search radius = 1.52";

		EXPECT_NE(std::find(near_pts3.begin(), near_pts3.end(), point2d_t{4.25, 3.5}), near_pts3.end())
				  << "point (4.25, 3.5) not found in search radius = 1.52";

		EXPECT_NE(std::find(near_pts3.begin(), near_pts3.end(), point2d_t{3.1, 3.5}), near_pts3.end())
				  << "point (3.1, 3.5) not found in search radius = 1.52";

		EXPECT_NE(std::find(near_pts3.begin(), near_pts3.end(), point2d_t{3.0, 2.0}), near_pts3.end())
				  << "point (3.0, 2.0) not found in search radius = 1.52";

		EXPECT_NE(std::find(near_pts3.begin(), near_pts3.end(), point2d_t{4.78, 1.7}), near_pts3.end())
				  << "point (4.78, 1.7) not found in search radius = 1.52";
	}
}

TEST(kd_tree, RadiusSearch3d)
{
	std::mt19937_64 pt_generator(0xbeefbeebbaaacccf);
	std::uniform_real_distribution<double> rand_pt(-2.0, 2.0);

	// Generate a cloud of random points in the box with min_pt (-2.0, -2.0, -2.0) max_pt (2.0, 2.0, 2.0)
	constexpr size_t n_pts = 5000;
	std::array<point3d_t, n_pts> points;
	for (size_t i = 0; i < n_pts; i++)
		points[i] = point3d_t{rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator)};

	// build the tree
	kd_tree<point3d_t> tree;
	tree.build(points.begin(), points.end());

	// Generate a random test points points
	constexpr size_t n_test_pts = 10;
	for (size_t i = 0; i < n_test_pts; i++)
	{
		point3d_t test_pt{rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator)};

		double const r = 0.5;
		auto near_pts = tree.radius_search(test_pt, r);

		// Check near-pts
		std::vector<point3d_t> near_pts_check;
		for (point3d_t const &p : points)
		{
			if (dist_sq(p, test_pt) < r * r)
				near_pts_check.push_back(p);
		}

		auto dist_sq_test_pt =
			 [&test_pt](point3d_t const &p1, point3d_t const &p2)
		{
			return dist_sq(p1, test_pt) < dist_sq(p2, test_pt);
		};

		std::sort(near_pts_check.begin(), near_pts_check.end(), dist_sq_test_pt);

		auto ge_begin = std::lower_bound(near_pts_check.begin(), near_pts_check.end(), r * r,
													[&test_pt](point3d_t const &p, double d_sq)
													{
														return dist_sq(p, test_pt) < d_sq;
													});

		size_t const n_near_pts = std::distance(near_pts_check.begin(), ge_begin);
		EXPECT_EQ(n_near_pts, near_pts.size());

		for (size_t i = 0; i < n_near_pts; i++)
			EXPECT_EQ(near_pts_check[i], near_pts[i]);
	}
}

TEST(kd_tree, NoPtsInSearchRadius)
{
	std::mt19937_64 pt_generator(0xefeebeebaaeaf987);
	std::uniform_real_distribution<double> rand_pt(-1.0, 1.0);

	constexpr size_t n_pts = 1000;
	std::array<point3d_t, n_pts> points;
	for (size_t i = 0; i < n_pts; i++)
		points[i] = point3d_t{rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator)};

	kd_tree<point3d_t> tree;
	tree.build(points.begin(), points.end());

	point3d_t const test_pt{2.0, 0, 0};
	auto near_pts = tree.radius_search(test_pt, 0.5);

	EXPECT_TRUE(near_pts.empty());
}

TEST(kd_tree, NoPtsInRange)
{
	using bbox_t = bbox<point3d_t>;

	std::mt19937_64 pt_generator(0xefeebeebaaeaf987);
	std::uniform_real_distribution<double> rand_pt(-1.0, 1.0);

	constexpr size_t n_pts = 1000;
	std::array<point3d_t, n_pts> points;
	for (size_t i = 0; i < n_pts; i++)
		points[i] = point3d_t{rand_pt(pt_generator), rand_pt(pt_generator), rand_pt(pt_generator)};

	kd_tree<point3d_t> tree;
	tree.build(points.begin(), points.end());

	auto pts_in_range = tree.range_search(bbox_t{point3d_t{1.1, 1.1, 1.1}, point3d_t{1.5, 1.5, 1.5}});

	EXPECT_TRUE(pts_in_range.empty());
}
