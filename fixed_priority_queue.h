#pragma once

#include <queue>
#include <memory>
#include <vector>
#include <algorithm>

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

	bool empty() const { return m_heap.empty(); }

	size_t size() const { return m_heap.size(); }
	size_t const max_size() const { return m_max_size; }

	T const& top() const { return m_heap.front(); }

	void push(T const& t)
	{
		if (size() == max_size())
		{
			auto min_element = std::min_element(m_heap.begin(), m_heap.end());
			if (t > *min_element)
			{
				*min_element = t;
				std::make_heap(m_heap.begin(), m_heap.end(), m_comp);
			}
		}
		else
		{
			m_heap.push_back(t);
			std::push_heap(m_heap.begin(), m_heap.end(), m_comp);
		}
	}

	void pop()
	{
		std::pop_heap(m_heap.begin(), m_heap.end());
		m_heap.pop_back();
	}
};