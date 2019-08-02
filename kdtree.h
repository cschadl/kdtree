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
			//std::sort(entry.begin, entry.end,
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
		value_type const dist_this_node = distance(p, node->val);
		knn_pq.push(knn_query{node->val, dist_this_node});

		size_t s = node->n_dim;

		value_type const dist_to_plane = p[s] - node->val[s];

		if (dist_to_plane <= 0)
		{
			// Traverse left, then right if the search sphere crosses the split plane
			k_nn_recursive_(p, k, node->left_child.get(), knn_pq);

			if (std::abs(dist_to_plane) < knn_pq.bottom().dist)
				k_nn_recursive_(p, k, node->right_child.get(), knn_pq);
		}
		else
		{
			k_nn_recursive_(p, k, node->right_child.get(), knn_pq);

			if (std::abs(dist_to_plane) < knn_pq.bottom().dist)
				k_nn_recursive_(p, k, node->left_child.get(), knn_pq);
		}
	}

	std::vector<PointType> k_nn_recursive(PointType const& p, size_t k) const
	{
		const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited = 0;

		constexpr value_type max_dist = std::numeric_limits<value_type>::max();

		fixed_priority_queue<knn_query> knn_pq(k);

		// Initialize max_dist_pt to ( max, max, ..., max)
		PointType max_dist_pt;
		for (size_t i = 0 ; i < Dim ; i++)
			max_dist_pt[i] = max_dist;

		for (size_t i = 0 ; i < k ; i++)
			knn_pq.push(knn_query{max_dist_pt, max_dist});

		if (!m_root)
			return { knn_pq.top().point };

		constexpr auto max_val = std::numeric_limits<value_type>::max();
		PointType root_min, root_max;
		for (size_t i = 0 ; i < Dim ; i++)
		{
			root_min[i] = -max_val;
			root_max[i] =  max_val;
		}

		k_nn_recursive_(p, k, m_root.get(), knn_pq);

		std::vector<PointType> k_nn_pts(k, max_dist_pt);
		size_t i = 0;

		while (!knn_pq.empty())
		{
			knn_query const& pt_dist = knn_pq.top();
			if (pt_dist.dist < max_val)
				k_nn_pts[i++] = pt_dist.point;

			knn_pq.pop();
		}

		return k_nn_pts;
	}

	std::vector<PointType> k_nn(PointType const& p, size_t k) const
	{
#if 0
		// This needs some work (too many nodes are visited)
		// Strategy should be something like:
		// - DFS left / right according to val[dim] w.r.t. split[dim]
		// - When no nodes are left from this, explore nodes where the
		//   splitting hypersphere crosses the splitting hyperplane
		const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited = 0;

		constexpr value_type max_dist = std::numeric_limits<value_type>::max();

		fixed_priority_queue<knn_query> knn_pq(k);

		// Initialize max_dist_pt to ( max, max, ..., max)
		PointType max_dist_pt;
		for (size_t i = 0 ; i < Dim ; i++)
			max_dist_pt[i] = max_dist;

		for (size_t i = 0 ; i < k ; i++)
			knn_pq.push(knn_query{max_dist_pt, max_dist});

		if (!m_root)
			return { knn_pq.top().point };

		constexpr auto max_val = std::numeric_limits<value_type>::max();
		PointType root_min, root_max;
		for (size_t i = 0 ; i < Dim ; i++)
		{
			root_min[i] = -max_val;
			root_max[i] =  max_val;
		}

		std::stack<node_t*> node_stack;
		node_stack.emplace(m_root.get());

		// To search, we explore the tree, pruning nodes that are
		// too far away from the search point.

		while (!node_stack.empty())
		{
			auto node = node_stack.top();
			node_stack.pop();

			if (!node)
				continue;

			const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited++;

			size_t const s = node->n_dim;

			// Get the distance from the p to this node
			value_type const dist_this_node = distance(p, node->val);
			knn_pq.push(knn_query{node->val, dist_this_node});

			value_type const dist_to_plane = p[s] - node->val[s];

			if (dist_to_plane <= 0)
			{
				if (std::abs(dist_to_plane) < knn_pq.bottom().dist)
					node_stack.emplace((node->right_child).get());

				node_stack.emplace((node->left_child).get());
			}
			else
			{
				if (std::abs(dist_to_plane) < knn_pq.bottom().dist)
					node_stack.emplace((node->left_child).get());

				node_stack.emplace((node->right_child).get());
			}
		}

		std::vector<PointType> k_nn_pts(k, max_dist_pt);
		size_t i = 0;

		while (!knn_pq.empty())
		{
			knn_query const& pt_dist = knn_pq.top();
			if (pt_dist.dist < max_val)
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

	size_t last_q_nodes_visited() const
	{
		return m_q_nodes_visited;
	}
};













