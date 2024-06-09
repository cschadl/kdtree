// Copyright (C) 2018 by Christopher Schadl <cschadl@gmail.com>

// Permission to use, copy, modify, and/or distribute this software for any purpose
// with or without fee is hereby granted.

// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD 
// TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
// DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
// WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
// ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#pragma once

#include <algorithm>
#include <vector>
#include <utility>
#include <cstdint>
#include <memory>
#include <stack>
#include <limits>
#include <cmath>

#include <kdtree/point_traits.hpp>
#include <kdtree/bbox.hpp>
#include <kdtree/detail/fixed_priority_queue.hpp>

namespace cds
{

namespace kdtree
{

template <typename PointType, size_t Dim = point_traits<PointType>::dim()>
class kd_tree
{
	static_assert(Dim > 1, "Dim must be greater than 1");

private:
	template <typename PointType_>
	struct node
	{
		size_t				n_dim;
		PointType			val;

		node<PointType_>*	left_child;
		node<PointType_>*	right_child;

		node(size_t dim, PointType val, node<PointType_>* left, node<PointType_>* right)
		: n_dim(dim)
		, val(std::move(val))
		, left_child(std::move(left))
		, right_child(std::move(right))
		{

		}

		node()
		: n_dim(0)
		, left_child(nullptr)
		, right_child(nullptr)
		{

		}
	};

	using node_t = node<PointType>;
	using value_type = typename point_traits<PointType>::value_type;

	std::vector<node<PointType>> 	m_nodes;					// node[0] is the root of the tree
	size_t 								m_q_nodes_visited;	// Nodes visited for last query (debugging)

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
		node_t const*		node;
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

	node<PointType> const* get_root() const
	{
		if (m_nodes.empty())
			return nullptr;

		return &(m_nodes[0]);
	}

	node<PointType>* get_root()
	{
		if (m_nodes.empty())
			return nullptr;

		return &(m_nodes[0]);
	}

public:
	kd_tree()
		: m_q_nodes_visited(0)
	{

	}

	template <typename InputIterator>
	void build(InputIterator begin, InputIterator end)
	{
		using ns_entry_t = node_stack_entry<InputIterator>;

		// We are using the points themselves as the split points,
		// so the number of nodes in the tree will simply be the
		// number of points.
		// TODO - if we somehow add *more* nodes that this (how?)
		// the resize operation will thrash all of our pointers.
		// Maybe this should be just a regular array?
		m_nodes.resize(std::distance(begin, end));

		// root node 0 is the default-constructed node
		size_t next_node_idx = 1;

		std::stack<ns_entry_t> node_stack;
		node_stack.emplace(ns_entry_t{get_root(), begin, end, 0});

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
				node->left_child = &(m_nodes[next_node_idx++]);
				node_stack.emplace(node->left_child, entry.begin, median, dim_n);
			}

			auto median_1 = std::next(median);
			if (median_1 != entry.end)
			{
				node->right_child = &(m_nodes[next_node_idx++]);
				node_stack.emplace(node->right_child, median_1, entry.end, dim_n);
			}
		}
	}

private:
	std::vector<PointType> k_nn_(PointType const& p, size_t k, value_type max_dist_sq) const
	{
		const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited = 0;

		detail_::fixed_priority_queue<knn_query> knn_pq(k);

		// Initialize max_dist_pt to ( max, max, ..., max )
		constexpr value_type max_val = std::numeric_limits<value_type>::max();
		PointType max_dist_pt = point_traits<PointType>::create(max_val);
		knn_pq.push(knn_query{max_dist_pt, max_val});

		if (!get_root())
			return { knn_pq.top().point };

		using ns_search_entry_t = std::tuple<node_t const*, size_t, value_type>;

		node<PointType> const* root_node = get_root();

		std::stack<ns_search_entry_t> node_stack;
		value_type const d = p[0] - root_node->val[0];
		node_stack.emplace(root_node, 0, d * d);

		// To search, we explore the tree, pruning nodes that are
		// too far away from the search point.

		while (!node_stack.empty())
		{
			node_t const* node;
			size_t s;
			value_type dist_to_plane_sq;
			std::tie(node, s, dist_to_plane_sq) = node_stack.top();

			node_stack.pop();

			if (!node)	// Traversed to leaf node
				continue;

			if (dist_to_plane_sq >= knn_pq.bottom().dist)
			{
				// Prune this branch of the tree, since the query point is
				// too far away from the splitting hyperplane
				continue;

				// It seems like there should also be some optimization that we can do
				// where we prune nodes that are greater than max_dist_sq away
				// from the search point, but that seems to break things
			}

			// Get the distance from the p to this node
			value_type const dist_this_node = distance_sq(p, node->val);
			if (dist_this_node < max_dist_sq)
				knn_pq.push(knn_query{node->val, dist_this_node});

			const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited++;

			value_type const dist_this_to_plane = p[node->n_dim] - node->val[node->n_dim];
			value_type const dist_this_to_plane_sq = dist_this_to_plane * dist_this_to_plane;

			if (dist_this_to_plane <= 0)
			{
				node_stack.emplace(node->right_child, node->n_dim, dist_this_to_plane_sq);
				node_stack.emplace(node->left_child, node->n_dim, value_type(-1));
			}
			else
			{
				node_stack.emplace(node->left_child, node->n_dim, dist_this_to_plane_sq);
				node_stack.emplace(node->right_child, node->n_dim, value_type(-1));
			}
		}

		std::vector<PointType> k_nn_pts(knn_pq.size(), max_dist_pt);
		size_t i = 0;

		while (!knn_pq.empty())
		{
			knn_query const& pt_dist = knn_pq.top();
			if (pt_dist.dist < max_dist_sq)
				k_nn_pts[i++] = pt_dist.point;

			knn_pq.pop();
		}
		k_nn_pts.erase(k_nn_pts.begin() + i, k_nn_pts.end());

		return k_nn_pts;
	}

public:
	std::vector<PointType> k_nn(PointType const& p, size_t k) const
	{
		return k_nn_(p, k, std::numeric_limits<value_type>::max());
	}

	PointType nn(PointType const& p) const
	{
		std::vector<PointType> nn_pt = k_nn(p, 1 /* k */);
		return nn_pt.front();
	}

	std::vector<PointType> range_search(bbox<PointType> const& range_bbox) const
	{
		using bbox_t = bbox<PointType>;

		const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited = 0;

		auto max_val = std::numeric_limits<value_type>::max();

		PointType min_pt = point_traits<PointType>::create(-max_val);
		PointType max_pt = point_traits<PointType>::create( max_val);

		std::stack<range_search_query> query_stack;
		query_stack.emplace(range_search_query{get_root(), bbox_t(min_pt, max_pt)});

		std::vector<PointType> points_in_range;

		while (!query_stack.empty())
		{
			range_search_query q = query_stack.top();
			query_stack.pop();

			node_t const* q_n = q.node;
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
				query_stack.emplace(range_search_query{q_n->left_child, left_bbox});
				query_stack.emplace(range_search_query{q_n->right_child, right_bbox});
			}
		}

		return points_in_range;
	}

	std::vector<PointType> radius_search(PointType const& p, double r)
	{
		return k_nn_(p, std::numeric_limits<size_t>::max(), r * r);
	}

	size_t last_q_nodes_visited() const
	{
		return m_q_nodes_visited;
	}
};

} // namespace kdtree

}	// namespace cds
