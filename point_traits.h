#pragma once

#include <array>

template <typename PointType>
struct point_traits
{
	using value_type = typename PointType::value_type;
	static constexpr size_t dim() { return PointType::Dim; }
};

// specialization for the common case where PointType is an std::array
template <typename T, size_t N>
struct point_traits< std::array<T, N> >
{
	using value_type = typename std::array<T, N>::value_type;
	static constexpr size_t dim() { return N; }
};
