#include <kdtree/detail/fixed_priority_queue.hpp>

#include <tut/tut.hpp>

using namespace cds::kdtree_detail_;

namespace tut
{

	struct fixed_priority_queue_data
	{

	};

	using fixed_priority_queue_test_t = test_group<fixed_priority_queue_data>;
	fixed_priority_queue_test_t fixed_priority_queue("fixed_priority_queue");

	template <> template <>
	void fixed_priority_queue_test_t::object::test<1>()
	{
		set_test_name("insert max heap");

		::fixed_priority_queue<int> fx_max_queue(5);
		ensure(fx_max_queue.empty());
		ensure(fx_max_queue.max_size() == 5);

		ensure(fx_max_queue.push(1));
		ensure(fx_max_queue.top() == 1);

		ensure(fx_max_queue.push(-3));
		ensure(fx_max_queue.top() == 1);

		ensure(fx_max_queue.push(5));
		ensure(fx_max_queue.top() == 5);

		ensure(fx_max_queue.push(0));
		ensure(fx_max_queue.top() == 5);

		ensure(fx_max_queue.push(7));
		ensure(fx_max_queue.top() == 7);

		//queue is full
		ensure(!fx_max_queue.push(-5));	// < lowest priority element, not added
		ensure(fx_max_queue.top() == 7);

		ensure(fx_max_queue.push(-1));
		ensure(fx_max_queue.top() == 7);

		ensure(fx_max_queue.push(10));
		ensure(fx_max_queue.top() == 10);	// Adding this replaces -1 as smallest element

		fx_max_queue.pop();
		ensure(fx_max_queue.top() == 7);

		fx_max_queue.pop();
		ensure(fx_max_queue.top() == 5);

		fx_max_queue.pop();
		ensure(fx_max_queue.top() == 1);

		fx_max_queue.pop();
		ensure(fx_max_queue.top() == 0);

		fx_max_queue.pop();
		ensure(fx_max_queue.empty());
	}

	template <> template <>
	void fixed_priority_queue_test_t::object::test<2>()
	{
		set_test_name ("min heap full");

		::fixed_priority_queue<int, std::greater<int>> fx_min_queue(5);
		for (size_t i = 0 ; i < 5 ; i++)
			ensure(fx_min_queue.push(std::numeric_limits<int>::max()));

		ensure(!fx_min_queue.empty());
		ensure(fx_min_queue.top() == std::numeric_limits<int>::max());

		ensure(fx_min_queue.push(10));
		ensure(fx_min_queue.top() == 10);

		ensure(fx_min_queue.push(5));
		ensure(fx_min_queue.top() == 5);

		ensure(fx_min_queue.push(7));
		ensure(fx_min_queue.top() == 5);

		ensure(fx_min_queue.push(-1));
		ensure(fx_min_queue.top() == -1);

		ensure(fx_min_queue.push(0));
		ensure(fx_min_queue.top() == -1);

		fx_min_queue.pop();
		ensure(fx_min_queue.top() == 0);

		fx_min_queue.pop();
		ensure(fx_min_queue.top() == 5);

		fx_min_queue.pop();
		ensure(fx_min_queue.top() == 7);

		fx_min_queue.pop();
		ensure(fx_min_queue.top() == 10);

		fx_min_queue.pop();
		ensure(fx_min_queue.empty());
	}
};










