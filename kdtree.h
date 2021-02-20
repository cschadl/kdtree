#pragma once

#include <algorithm>
#include <vector>
#include <utility>
#include <cstdint>
#include <memory>
#include <stack>
#include <limits>
#include <cmath>

#include <point_traits.h>
#include <bbox.h>
#include <fixed_priority_queue.h>

template <typename PointType, size_t Dim = point_traits<PointType>::dim()>
class kd_tree
{
	static_assert(Dim > 1, "Dim must be greater than 1");

private:
	template <typename PointType_>
	struct node
	{
		size_t										n_dim;
		PointType									val;

		std::unique_ptr< node<PointType_> >	left_child;
		std::unique_ptr< node<PointType_> >	right_child;

		node(size_t dim, PointType val, std::unique_ptr< node<PointType_> > left, std::unique_ptr< node<PointType_> > right)
		: n_dim(dim)
		, val(std::move(val))
		, left_child(std::move(left))
		, right_child(std::move(right))
		{

		}

		node()
		: n_dim(0)
		{

		}
	};

	using node_t = node<PointType>;
	using value_type = typename point_traits<PointType>::value_type;

	std::unique_ptr<node_t>	m_root;
	size_t 						m_q_nodes_visited;	// Nodes visited for last query (debugging)

	template <typename InputIterator>
	struct node_stack_entry
	{
		node_t* 			node;
		InputIterator	begin;
		InputIterator	end;
		size_t			dim;

		node_stack_entry(node_t * node_, InputIterator begin_, InputIterator end_, size_t dim_)
			: node(node_)
			, begin(begin_)
			, end(end_)
			, dim(dim_)
		{

		}
	};

	struct knn_query
	{
		PointType	point;
		value_type	dist;

		bool operator<(const knn_query& rhs) const
		{
			// Backwards, for min-priority queue
			return dist > rhs.dist;
		}
	};

	struct range_search_query
	{
		node_t*				node;
		bbox<PointType>	extent;
	};

	static typename point_traits<PointType>::value_type distance_sq(PointType const& pt1, PointType const& pt2)
	{
		value_type distance_sq = value_type(0);
		for (size_t i = 0 ; i < Dim ; i++)
		{
			value_type const dist_i = pt2[i] - pt1[i];
			distance_sq += (dist_i * dist_i);
		}

		return distance_sq;
	}

	static typename point_traits<PointType>::value_type distance(PointType const& pt1, PointType const& pt2)
	{
		return std::sqrt(distance_sq(pt1, pt2));
	}

public:
	kd_tree()
		: m_q_nodes_visited(0)
	{

	}

	template <typename InputIterator>
	std::unique_ptr<node_t> build_recursive_(InputIterator begin, InputIterator end, size_t dim)
	{
		size_t const n_nodes = std::distance(begin, end);
		if (n_nodes == 0)
			return nullptr;

		std::nth_element(begin, begin + n_nodes / 2, end,
			[dim](auto pt1, auto pt2)
			{
				return pt1[dim] < pt2[dim];
			});

		InputIterator median = begin + n_nodes / 2;

		size_t dim_n = (dim + 1) % Dim;

		return std::make_unique<node_t>(
				dim,
				*median,
				build_recursive_(begin, median, dim_n),
				build_recursive_(std::next(median), end, dim_n));
	}

	template <typename InputIterator>
	void build_recursive(InputIterator begin, InputIterator end)
	{
		m_root = build_recursive_(begin, end, 0);
	}

	template <typename InputIterator>
	void build(InputIterator begin, InputIterator end)
	{
#if 1
		using ns_entry_t = node_stack_entry<InputIterator>;

		m_root = std::make_unique<node_t>();

		std::stack<ns_entry_t> node_stack;
		node_stack.emplace(ns_entry_t{m_root.get(), begin, end, 0});

		while (!node_stack.empty())
		{
			ns_entry_t entry = node_stack.top();
			node_stack.pop();

			node_t* node = entry.node;
			size_t dim = entry.dim;

			size_t const n_elements = std::distance(entry.begin, entry.end);

			std::nth_element(entry.begin, entry.begin + n_elements / 2, entry.end,
				[dim](auto pt1, auto pt2)
				{
					return pt1[dim] < pt2[dim];
				});

			InputIterator median = entry.begin + n_elements / 2;

			node->val = *median;
			node->n_dim = dim;

			size_t dim_n = ++dim % Dim;

			if (std::distance(entry.begin, median) > 0)
			{
				node->left_child = std::make_unique<node_t>();
				node_stack.emplace(node->left_child.get(), entry.begin, median, dim_n);
			}

			auto median_1 = std::next(median);
			if (median_1 != entry.end)
			{
				node->right_child = std::make_unique<node_t>();
				node_stack.emplace(node->right_child.get(), median_1, entry.end, dim_n);
			}
		}
#else
		build_recursive(begin, end);
#endif
	}

	void k_nn_recursive_(PointType const& p, size_t const k, node_t* node, fixed_priority_queue<knn_query>& knn_pq) const
	{
		if (!node)
			return;

		const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited++;

		// Get the distance from the p to this node
		value_type const dist_this_node = distance_sq(p, node->val);
		knn_pq.push(knn_query{node->val, dist_this_node});

		size_t s = node->n_dim;

		value_type const dist_to_plane = p[s] - node->val[s];
		value_type const dist_to_plane_sq = dist_to_plane * dist_to_plane;

		if (dist_to_plane <= 0)
		{
			// Traverse left, then right if the search sphere crosses the split plane
			k_nn_recursive_(p, k, node->left_child.get(), knn_pq);

			if (dist_to_plane_sq < knn_pq.bottom().dist)
				k_nn_recursive_(p, k, node->right_child.get(), knn_pq);
		}
		else
		{
			k_nn_recursive_(p, k, node->right_child.get(), knn_pq);

			if (dist_to_plane_sq < knn_pq.bottom().dist)
				k_nn_recursive_(p, k, node->left_child.get(), knn_pq);
		}
	}

	std::vector<PointType> k_nn_recursive(PointType const& p, size_t k) const
	{
		const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited = 0;

		constexpr value_type max_dist = std::numeric_limits<value_type>::max();

		fixed_priority_queue<knn_query> knn_pq(k);

		// Initialize max_dist_pt to ( max, max, ..., max)
		PointType max_dist_pt = point_traits<PointType>::create(max_dist);

		for (size_t i = 0 ; i < k ; i++)
			knn_pq.push(knn_query{max_dist_pt, max_dist});

		if (!m_root)
			return { knn_pq.top().point };

		k_nn_recursive_(p, k, m_root.get(), knn_pq);

		std::vector<PointType> k_nn_pts(k, max_dist_pt);
		size_t i = 0;

		while (!knn_pq.empty())
		{
			knn_query const& pt_dist = knn_pq.top();
			if (pt_dist.dist < max_dist)
				k_nn_pts[i++] = pt_dist.point;

			knn_pq.pop();
		}

		return k_nn_pts;
	}

	std::vector<PointType> k_nn(PointType const& p, size_t k) const
	{
#if 1
		const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited = 0;

		constexpr value_type max_dist = std::numeric_limits<value_type>::max();

		fixed_priority_queue<knn_query> knn_pq(k);

		// Initialize max_dist_pt to ( max, max, ..., max)
		PointType max_dist_pt = point_traits<PointType>::create(max_dist);

		for (size_t i = 0 ; i < k ; i++)
			knn_pq.push(knn_query{max_dist_pt, max_dist});

		if (!m_root)
			return { knn_pq.top().point };

		using ns_entry_t = std::tuple<node_t*, size_t, value_type>;

		std::stack<ns_entry_t> node_stack;
		value_type const d = p[0] - m_root->val[0];
		node_stack.emplace(m_root.get(), 0, d * d);

		// To search, we explore the tree, pruning nodes that are
		// too far away from the search point.

		while (!node_stack.empty())
		{
			node_t* node;
			size_t s;
			value_type dist_to_plane_sq;
			std::tie(node, s, dist_to_plane_sq) = node_stack.top();

			node_stack.pop();

			if (!node)	// Traversed to leaf node
				continue;

			// Prune this branch of the tree, since the query point is
			// too far away from the splitting hyperplane
			if (dist_to_plane_sq >= knn_pq.bottom().dist)
				continue;

			// Get the distance from the p to this node
			value_type const dist_this_node = distance_sq(p, node->val);
			knn_pq.push(knn_query{node->val, dist_this_node});

			const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited++;

			value_type const dist_this_to_plane = p[node->n_dim] - node->val[node->n_dim];
			value_type const dist_this_to_plane_sq = dist_this_to_plane * dist_this_to_plane;

			if (dist_this_to_plane <= 0)
			{
				node_stack.emplace((node->right_child).get(), node->n_dim, dist_this_to_plane_sq);
				node_stack.emplace((node->left_child).get(), node->n_dim, value_type(-1));
			}
			else
			{
				node_stack.emplace((node->left_child).get(), node->n_dim, dist_this_to_plane_sq);
				node_stack.emplace((node->right_child).get(), node->n_dim, value_type(-1));
			}
		}

		std::vector<PointType> k_nn_pts(k, max_dist_pt);
		size_t i = 0;

		while (!knn_pq.empty())
		{
			knn_query const& pt_dist = knn_pq.top();
			if (pt_dist.dist < max_dist)
				k_nn_pts[i++] = pt_dist.point;

			knn_pq.pop();
		}

		return k_nn_pts;
#else
		return k_nn_recursive(p, k);
#endif
	}

	PointType nn(PointType const& p) const
	{
		std::vector<PointType> nn_pt = k_nn_recursive(p, 1 /* k */);
		return nn_pt.front();
	}

	std::vector<PointType> range_search(bbox<PointType>& range_bbox) const
	{
		using bbox_t = bbox<PointType>;

		const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited = 0;

		auto max_val = std::numeric_limits<value_type>::max();

		PointType min_pt = point_traits<PointType>::create(-max_val);
		PointType max_pt = point_traits<PointType>::create( max_val);

		std::stack<range_search_query> query_stack;
		query_stack.emplace(range_search_query{m_root.get(), bbox_t(min_pt, max_pt)});

		std::vector<PointType> points_in_range;

		while (!query_stack.empty())
		{
			range_search_query q = query_stack.top();
			query_stack.pop();

			node_t* q_n = q.node;
			bbox_t const& q_bbox = q.extent;

			if (!q_n)
				continue;

			if (!range_bbox.intersects(q_bbox))
				continue;

			const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited++;

			if (range_bbox.contains(q_n->val))
				points_in_range.push_back(q_n->val);

			bbox_t left_bbox, right_bbox;
			value_type split_val = q_n->val[q_n->n_dim];

			if (q_bbox.split(q_n->n_dim, split_val, left_bbox, right_bbox))
			{
				query_stack.emplace(range_search_query{q_n->left_child.get(), left_bbox});
				query_stack.emplace(range_search_query{q_n->right_child.get(), right_bbox});
			}
		}

		return points_in_range;
	}

	std::vector<PointType> radius_search(PointType const& p, double r)
	{
		double const dist_sq = r * r;

		// Make a bbox that contains the sphere of radius r
		double const cube_vert_dist = ::sqrt(dist_sq * 3);
		
		PointType min_pt = p;
		for (size_t i = 0 ; i < p.size() ; i++)
			min_pt[i] -= cube_vert_dist;	// TODO - add something to point_traits<PointType> for this

		PointType max_pt = p;
		for (size_t i = 0 ; i < p.size() ; i++)
			max_pt[i] += cube_vert_dist;

		bbox<PointType> search_bbox{min_pt, max_pt};
		
		auto box_results = range_search(search_bbox);

		// Remove any results outside the search radius
		auto points_outside_sphere = std::remove_if(
			box_results.begin(), box_results.end(),
			[dist_sq, &p](auto const& res_p)
			{
				return distance_sq(p, res_p) > dist_sq;
			});
		
		box_results.erase(points_outside_sphere, box_results.end());

		return box_results;
	}

	size_t last_q_nodes_visited() const
	{
		return m_q_nodes_visited;
	}
};




