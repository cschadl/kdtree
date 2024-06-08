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

#include <array>
#include <utility>

namespace cds
{
namespace kdtree
{

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

} // kdtree_detail_

} // cds