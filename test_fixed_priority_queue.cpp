#include "fixed_priority_queue.h"

#include <tut/tut.hpp>

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

		fx_max_queue.push(1);
		ensure(fx_max_queue.top() == 1);

		fx_max_queue.push(-3);
		ensure(fx_max_queue.top() == 1);

		fx_max_queue.push(5);
		ensure(fx_max_queue.top() == 5);

		fx_max_queue.push(0);
		ensure(fx_max_queue.top() == 5);

		fx_max_queue.push(7);
		ensure(fx_max_queue.top() == 7);

		//queue is full
		fx_max_queue.push(-5);
		ensure(fx_max_queue.top() == 7);

		fx_max_queue.push(-1);
		ensure(fx_max_queue.top() == 7);

		fx_max_queue.push(10);
		ensure(fx_max_queue.top() == 10);

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
};
