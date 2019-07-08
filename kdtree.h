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
		PointType									val;			// TODO - maybe an iterator?

		std::unique_ptr< node<PointType_> >	left_child;
		std::unique_ptr< node<PointType_> >	right_child;
	};

	std::unique_ptr< node<PointType> >	m_root;

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

public:
	kd_tree() = default;

	template <typename InputIterator>
	void build(InputIterator begin, InputIterator end)
	{
		//using node_t = node<PointType>;

		//size_t depth = 0;

		//std::stack<node_t> node_stack;
	}
};
