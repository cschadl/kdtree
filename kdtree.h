#pragma once

#include <algorithm>
#include <vector>
#include <utility>
#include <cstdint>
#include <memory>
#include <stack>
#include <limits>
#include <cmath>

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

		node_stack_entry(node_t * node_, InputIterator begin_, InputIterator end_)
			: node(node_)
			, begin(begin_)
			, end(end_)
		{

		}
	};

	struct query_stack_entry
	{
		node_t*				node;
		bbox<PointType>	node_bbox;

		query_stack_entry(node_t* node_, bbox<PointType> bbox_)
		: node(node_)
		, node_bbox(std::move(bbox_))
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
	void build(InputIterator begin, InputIterator end)
	{
		using ns_entry_t = node_stack_entry<InputIterator>;

		size_t depth = 0;

		m_root = std::make_unique<node_t>();

		std::stack<ns_entry_t> node_stack;
		node_stack.emplace(ns_entry_t{m_root.get(), begin, end});

		while (!node_stack.empty())
		{
			ns_entry_t entry = node_stack.top();
			node_stack.pop();

			node_t* node = entry.node;

			size_t const dim = depth++ % Dim;
			size_t const n_elements = std::distance(entry.begin, entry.end);

			std::nth_element(entry.begin, entry.begin + n_elements / 2, entry.end,
				[dim](auto pt1, auto pt2)
				{
					return pt1[dim] < pt2[dim];
				});

			InputIterator median = entry.begin + n_elements / 2;

			node->val = *median;

			if (std::distance(entry.begin, median) > 0)
			{
				node->left_child = std::make_unique<node_t>();
				node_stack.emplace(node->left_child.get(), entry.begin, median);
			}

			auto median_1 = std::next(median);
			if (median_1 != entry.end)
			{
				node->right_child = std::make_unique<node_t>();
				node_stack.emplace(node->right_child.get(), median_1, entry.end);
			}
		}
	}

	std::vector<PointType> k_nn(PointType const& p, size_t k) const
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

		std::stack<query_stack_entry> node_stack;
		node_stack.emplace(query_stack_entry{m_root.get(), bbox<PointType>(root_min, root_max)});

		size_t depth = 0;

		// To search, we explore the tree, pruning nodes that are
		// too far away from the search point.

		while (!node_stack.empty())
		{
			auto ns_entry = node_stack.top();
			node_stack.pop();

			if (!ns_entry.node)
				continue;

			const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited++;

			node_t* node = ns_entry.node;
			bbox<PointType> const& node_bbox = ns_entry.node_bbox;

			size_t const s = depth++ % Dim;

			bbox<PointType> left_bbox, right_bbox;
			node_bbox.split(s, node->val[s], left_bbox, right_bbox);

			// Get the distance from the p to this node
			value_type const dist_this_node = distance(p, node->val);
			knn_pq.push(knn_query{node->val, dist_this_node});

			value_type const dist_left = node->left_child ? std::abs(p[s] - node->left_child->val[s]) : max_val;
			value_type const dist_right = node->right_child ? std::abs(p[s] - node->right_child->val[s]) : max_val;

			if (p[s] > node->val[s])
			{
				if (dist_this_node > dist_left || left_bbox.contains(p))
					node_stack.emplace(query_stack_entry{(node->left_child).get(), left_bbox});
				if (dist_this_node > dist_right || right_bbox.contains(p))
					node_stack.emplace(query_stack_entry{(node->right_child).get(), right_bbox});
			}
			else
			{
				if (dist_this_node > dist_right || right_bbox.contains(p))
					node_stack.emplace(query_stack_entry{(node->right_child).get(), right_bbox});
				if (dist_this_node > dist_left || left_bbox.contains(p))
					node_stack.emplace(query_stack_entry{(node->left_child).get(), left_bbox});
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
	}

	PointType nn(PointType const& p) const
	{
		std::vector<PointType> nn_pt = k_nn(p, 1 /* k */);
		return nn_pt.front();
	}

	size_t last_q_nodes_visited() const
	{
		return m_q_nodes_visited;
	}
};













