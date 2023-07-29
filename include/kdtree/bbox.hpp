#pragma once

#include <kdtree/point_traits.hpp>

#include <limits>
#include <cmath>

namespace cds
{

namespace kdtree
{

/// Axis-aligned bounding box in N dimensions.
/// This bbox class only contains the operations needed for kd_tree range search queries
template <typename PointType, size_t Dim = point_traits<PointType>::dim()>
class bbox
{
private:
	static_assert(Dim <= point_traits<PointType>::dim(), "Dimension must be <= point dimension");

	PointType m_min;
	PointType m_max;

	using value_type = typename point_traits<PointType>::value_type;

public:
	bbox()
		: m_min(point_traits<PointType>::create(value_type(0)))
		, m_max(point_traits<PointType>::create(value_type(0)))
	{

	}

	bbox(PointType min_, PointType max_)
	: m_min(min_)
	, m_max(max_)
	{

	}

	PointType const& min() const { return m_min; }
	PointType const& max() const { return m_max; }

	bool operator==(bbox<PointType, Dim> const& rhs) const
	{
		return m_min == rhs.m_min && m_max == rhs.m_max;
	}

	bool valid() const
	{
		for (size_t i = 0 ; i < Dim ; i++)
			if (m_min[i] > m_max[i])
				return false;

		return true;
	}

	bool empty() const { return m_min == m_max; }

	bool contains(PointType const& pt) const
	{
		for (size_t i = 0 ; i < Dim ; i++)
			if (pt[i] < m_min[i] || pt[i] > m_max[i])
				return false;

		return true;
	}

	bool intersects(bbox const& rhs) const
	{
		// All intervals must overlap
		for (size_t i = 0 ; i < Dim ; i++)
			if (m_min[i] > rhs.m_max[i] || rhs.m_min[i] > m_max[i])
				return false;

		return true;
	}

	bool disjoint(bbox const& rhs) const
	{
		return !intersects(rhs);
	}

	bool intersection(bbox const& rhs, bbox& out_intersection) const
	{
		if (disjoint(rhs))
			return false;

		PointType intersect_min;
		for (size_t i = 0 ; i < Dim ; i++)
			intersect_min[i] = std::max(m_min[i], rhs.m_min[i]);

		PointType intersect_max;
		for (size_t i = 0 ; i < Dim ; i++)
			intersect_max[i] = std::min(m_max[i], rhs.m_max[i]);

		out_intersection.m_min = intersect_min;
		out_intersection.m_max = intersect_max;

		return true;
	}

	bool split(size_t dim, value_type val, bbox& out_lt_bbox, bbox& out_gt_bbox) const
	{
		if (dim >= Dim)
			return false;

		if (val < m_min[dim] || val > m_max[dim])
			return false;

		auto lt_bbox_max = m_max;
		lt_bbox_max[dim] = val;

		auto gt_bbox_min = m_min;
		gt_bbox_min[dim] = val;

		out_lt_bbox = { m_min, lt_bbox_max };
		out_gt_bbox = { gt_bbox_min, m_max };

		return true;
	}
};

}	// namespace kdtree

}	// namespace cds