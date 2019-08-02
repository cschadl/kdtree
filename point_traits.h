#pragma once

#include <array>
#include <utility>

template <typename PointType>
struct point_traits
{
	using value_type = typename PointType::value_type;
	static constexpr size_t dim() { return PointType::Dim; }
	static PointType create(value_type val) { return PointType(val); }
};

namespace detail_
{
	template <typename U, size_t... Is>
	constexpr std::array<U, sizeof...(Is)> make_array(U const& value, std::index_sequence<Is...>)
	{
		// The static_cast<void> is in case U has overloaded operator, (who does this?)
		// Also it prevents an annoying compiler warning in GCC
		// warning: left operand of comma operator has no effect [-Wunused-value]
		return {{(static_cast<void>(Is), value)...}};
	}
}

// specialization for the common case where PointType is an std::array
template <typename T, size_t N>
struct point_traits< std::array<T, N> >
{
	using value_type = typename std::array<T, N>::value_type;
	static constexpr size_t dim() { return N; }

	static std::array<T, N> create(value_type val)
	{
		return detail_::make_array(val, std::make_index_sequence<N>{});
	}
};

