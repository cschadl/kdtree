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

#include <queue>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>

namespace cds
{

namespace kdtree
{

namespace detail_
{

template <typename T, typename C = std::less<T>>
class fixed_priority_queue
{
private:
	std::vector<T>						m_heap;
	size_t								m_max_size;
	C										m_comp;

public:
	fixed_priority_queue() = delete;

	fixed_priority_queue(size_t max_size)
		: m_max_size(max_size)
	{

	}

	fixed_priority_queue(size_t max_size, C comp)
		: m_max_size(max_size)
		, m_comp(comp)
	{

	}

	bool empty() const { return m_heap.empty(); }

	size_t size() const { return m_heap.size(); }
	size_t const max_size() const { return m_max_size; }

	T const& top() const { return m_heap.front(); }

	T const& bottom() const { return *std::min_element(m_heap.begin(), m_heap.end(), m_comp); }

	// return true if a new element was inserted into the priority queue
	bool push(T const& t)
	{
		if (size() == max_size())
		{
			auto min_element = std::min_element(m_heap.begin(), m_heap.end(), m_comp);
			if (!m_comp(t, *min_element))
			{
				*min_element = t;
				std::make_heap(m_heap.begin(), m_heap.end(), m_comp);

				return true;
			}
		}
		else
		{
			m_heap.push_back(t);
			std::push_heap(m_heap.begin(), m_heap.end(), m_comp);

			return true;
		}

		return false;
	}

	template <typename... Args>
	bool emplace(Args&&... args)
	{
		T t(std::forward<Args>(args)...);

		if (size() == max_size())
		{
			auto min_element = std::min_element(m_heap.begin(), m_heap.end(), m_comp);
			if (!m_comp(t, *min_element))
			{
				*min_element = std::move(t);
				std::make_heap(m_heap.begin(), m_heap.end(), m_comp);

				return true;
			}
		}
		else
		{
			m_heap.emplace_back(std::move(t));
			std::push_heap(m_heap.begin(), m_heap.end(), m_comp);

			return true;
		}

		return false;
	}

	void pop()
	{
		std::pop_heap(m_heap.begin(), m_heap.end(), m_comp);
		m_heap.pop_back();
	}
};

} // namespace kdtree

} // namespace detail

} // namespace cds
