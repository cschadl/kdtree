#pragma once

#include <algorithm>
#include <vector>
#include <utility>
#include <cstdint>
#include <memory>
#include <stack>
#include <limits>

#include <bbox.h>


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

	using value_type = typename point_traits<PointType>::value_type;

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

	PointType nn(PointType const& p) const
	{
		const_cast<kd_tree<PointType, Dim>&>(*this).m_q_nodes_visited = 0;

		value_type min_dist_sq = std::numeric_limits<value_type>::max();

		// Initialize min_pt to ( max, max, ..., max)
		PointType min_pt;
		for (size_t i = 0 ; i < Dim ; i++)
			min_pt[i] = min_dist_sq;

		if (!m_root)
			return min_pt;

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

			// prune this node if the point does not lie within the node bbox
			if (!node_bbox.contains(p))
				continue;

			// Get the distance from the min_pt to this node
			value_type const dist_this_node = distance(p, node->val);
			if (dist_this_node < min_dist_sq)
			{
				min_dist_sq = dist_this_node;
				min_pt = node->val;
			}

			size_t const s = depth++ % Dim;

			bbox<PointType> left_bbox, right_bbox;
			node_bbox.split(s, node->val[s], left_bbox, right_bbox);

			if (p[s] < node->val[s])
			{
				node_stack.emplace(query_stack_entry{(node->left_child).get(), left_bbox});
				node_stack.emplace(query_stack_entry{(node->right_child).get(), right_bbox});
			}
			else
			{
				node_stack.emplace(query_stack_entry{(node->right_child).get(), right_bbox});
				node_stack.emplace(query_stack_entry{(node->left_child).get(), left_bbox});
			}
		}

		return min_pt;
	}

	size_t last_q_nodes_visited() const
	{
		return m_q_nodes_visited;
	}
};













