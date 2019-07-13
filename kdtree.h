#pragma once

#include <algorithm>
#include <vector>
#include <utility>
#include <cstdint>
#include <memory>
#include <stack>
#include <limits>

template <typename PointType>
struct point_traits
{
	using value_type = typename PointType::value_type;
	static constexpr size_t dim() { return PointType::Dim; }
};

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

public:
	kd_tree() = default;

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
		value_type min_dist_sq = std::numeric_limits<value_type>::max();
		PointType min_pt;
		for (size_t i = 0 ; i < Dim ; i++)
			min_pt[i] = min_dist_sq;

		if (!m_root)
			return min_pt;

		std::stack<node_t*> node_stack;
		node_stack.push(m_root.get());

		while (!node_stack.empty())
		{
			node_t* node = node_stack.top();
			node_stack.pop();

			// Get the distance from the min_pt to this node
			value_type const dist_sq = distance_sq(p, node->val);
			if (dist_sq < min_dist_sq)
			{
				min_dist_sq = dist_sq;
				min_pt = node->val;
			}

			// Add the left and right nodes to the stack and keep going
			value_type const left_dist_sq = node->left_child ?
					distance_sq(p, node->left_child->val) :
					std::numeric_limits<value_type>::max();

			value_type const right_dist_sq = node->right_child ?
					distance_sq(p, node->right_child->val) :
					std::numeric_limits<value_type>::max();

			node_t* min_node = nullptr;
			if (left_dist_sq < right_dist_sq)
				min_node = node->left_child.get();
			else if (right_dist_sq < left_dist_sq)
				min_node = node->right_child.get();

			if (min_node)
				node_stack.push(min_node);
		}

		return min_pt;
	}
};













