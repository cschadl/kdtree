#pragma once

#include <algorithm>
#include <vector>
#include <utility>
#include <cstdint>
#include <memory>
#include <stack>

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

	template <typename IteratorType>
	struct point_index
	{
		PointType 		pt;
		size_t			dim;
		IteratorType	idx;

		bool const operator<(point_index const& cmp) const
		{
			return pt[dim] < cmp.pt[dim];
		}
	};

	template <typename InputIterator>
	struct node_stack_entry
	{
		node_t * 		node;
		InputIterator	begin;
		InputIterator	end;

		node_stack_entry(node_t * node_, InputIterator begin_, InputIterator end_)
			: node(node_)
			, begin(begin_)
			, end(end_)
		{

		}
	};

public:	// TODO - test and make private
	template <typename InputIterator>
	static InputIterator get_median(InputIterator begin, InputIterator end, size_t dim)
	{
		std::vector< point_index<InputIterator> > point_its(std::distance(begin, end));
		auto oi = point_its.begin();
		for (auto it = begin ; it != end ; ++it)
			*oi++ = point_index<InputIterator>{*it, dim, it};

		std::nth_element(point_its.begin(), point_its.begin() + point_its.size() / 2, point_its.end());

		auto point_its_median = point_its.begin() + point_its.size() / 2;
		return point_its_median->idx;
	}

	template <typename InputIterator>
	InputIterator add_node(node_t * node, InputIterator begin, InputIterator end, size_t depth)
	{
		size_t dim = depth % Dim;
		auto median_it = get_median(begin, end, dim);

		node->val = *median_it;
		node->n_dim = dim;

		return median_it;
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

			auto median = add_node(entry.node, entry.begin, entry.end, ++depth);

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
};
