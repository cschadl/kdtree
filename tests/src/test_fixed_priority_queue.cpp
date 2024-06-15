// Copyright (C) 2018 by Christopher Schadl <cschadl@gmail.com>

// Permission to use, copy, modify, and/or distribute this software for any purpose
// with or without fee is hereby granted.

// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD 
// TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
// DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
// WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
// ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#include <gtest/gtest.h>

#include <kdtree/detail/fixed_priority_queue.hpp>

using namespace cds::kdtree::detail_;

TEST(fixed_priority_queue, MaxHeapInsert)
{
	fixed_priority_queue<int> fx_max_queue(5);
	EXPECT_TRUE(fx_max_queue.empty());
	EXPECT_EQ(fx_max_queue.max_size(), 5);

	EXPECT_TRUE(fx_max_queue.push(1));
	EXPECT_EQ(fx_max_queue.top(), 1);

	EXPECT_TRUE(fx_max_queue.push(-3));
	EXPECT_EQ(fx_max_queue.top(), 1);

	EXPECT_TRUE(fx_max_queue.push(5));
	EXPECT_EQ(fx_max_queue.top(), 5);

	EXPECT_TRUE(fx_max_queue.push(0));
	EXPECT_EQ(fx_max_queue.top(), 5);

	EXPECT_TRUE(fx_max_queue.push(7));
	EXPECT_EQ(fx_max_queue.top(), 7);

	//queue is full
	EXPECT_TRUE(!fx_max_queue.push(-5));	// < lowest priority element, not added
	EXPECT_EQ(fx_max_queue.top(), 7);

	EXPECT_TRUE(fx_max_queue.push(-1));
	EXPECT_EQ(fx_max_queue.top(), 7);

	EXPECT_TRUE(fx_max_queue.push(10));
	EXPECT_EQ(fx_max_queue.top(), 10);	// Adding this replaces -1 as smallest element

	fx_max_queue.pop();
	EXPECT_EQ(fx_max_queue.top(), 7);

	fx_max_queue.pop();
	EXPECT_EQ(fx_max_queue.top(), 5);

	fx_max_queue.pop();
	EXPECT_EQ(fx_max_queue.top(), 1);

	fx_max_queue.pop();
	EXPECT_EQ(fx_max_queue.top(), 0);

	fx_max_queue.pop();
	EXPECT_TRUE(fx_max_queue.empty());
}

TEST(fixed_priority_queue, MinHeapFull)
{
	fixed_priority_queue<int, std::greater<int>> fx_min_queue(5);
	for (size_t i = 0 ; i < 5 ; i++)
		EXPECT_TRUE(fx_min_queue.push(std::numeric_limits<int>::max()));

	EXPECT_FALSE(fx_min_queue.empty());
	EXPECT_EQ(fx_min_queue.top(), std::numeric_limits<int>::max());

	EXPECT_TRUE(fx_min_queue.push(10));
	EXPECT_EQ(fx_min_queue.top(), 10);

	EXPECT_TRUE(fx_min_queue.push(5));
	EXPECT_EQ(fx_min_queue.top(), 5);

	EXPECT_TRUE(fx_min_queue.push(7));
	EXPECT_EQ(fx_min_queue.top(), 5);

	EXPECT_TRUE(fx_min_queue.push(-1));
	EXPECT_EQ(fx_min_queue.top(), -1);

	EXPECT_TRUE(fx_min_queue.push(0));
	EXPECT_EQ(fx_min_queue.top(), -1);

	fx_min_queue.pop();
	EXPECT_EQ(fx_min_queue.top(), 0);

	fx_min_queue.pop();
	EXPECT_EQ(fx_min_queue.top(), 5);

	fx_min_queue.pop();
	EXPECT_EQ(fx_min_queue.top(), 7);

	fx_min_queue.pop();
	EXPECT_EQ(fx_min_queue.top(), 10);

	fx_min_queue.pop();
	EXPECT_TRUE(fx_min_queue.empty());
}

namespace
{
	struct thing
	{
		std::string name;
		int val{0};

		thing(std::string const& the_name, int the_val)
			: name(the_name)
			, val(the_val)
		{
			
		}

		bool operator<(thing const& t) const
		{
			return this->val < t.val;
		}

		bool operator==(thing const& t) const
		{
			return this->name == t.name &&
				this->val == t.val;
		}
	};
}

TEST(fixed_priority_queue, MaxHeapEmplace)
{
	fixed_priority_queue<thing> fx_max_queue(5);
	EXPECT_TRUE(fx_max_queue.empty());
	EXPECT_EQ(fx_max_queue.max_size(), 5);

	EXPECT_TRUE(fx_max_queue.emplace("orange", 1));
	EXPECT_EQ(fx_max_queue.top(), thing("orange", 1));

	EXPECT_TRUE(fx_max_queue.emplace("apple", -3));
	EXPECT_EQ(fx_max_queue.top(), thing("orange", 1));

	EXPECT_TRUE(fx_max_queue.emplace("pear", 5));
	EXPECT_EQ(fx_max_queue.top(), thing("pear", 5));

	EXPECT_TRUE(fx_max_queue.emplace("grape", 0));
	EXPECT_EQ(fx_max_queue.top(), thing("pear", 5));

	EXPECT_TRUE(fx_max_queue.emplace("durian", 7));
	EXPECT_EQ(fx_max_queue.top(), thing("durian", 7));

	//queue is full
	EXPECT_TRUE(!fx_max_queue.emplace("jackfruit", -5));	// < lowest priority element, not added
	EXPECT_EQ(fx_max_queue.top(), thing("durian", 7));

	EXPECT_TRUE(fx_max_queue.emplace("tomato", -1));
	EXPECT_EQ(fx_max_queue.top(), thing("durian", 7));

	EXPECT_TRUE(fx_max_queue.emplace("blueberry", 10));
	EXPECT_EQ(fx_max_queue.top(), thing("blueberry", 10));	// Adding this replaces ("tomato", -1) as smallest element

	fx_max_queue.pop();
	EXPECT_EQ(fx_max_queue.top(), thing("durian", 7));

	fx_max_queue.pop();
	EXPECT_EQ(fx_max_queue.top(), thing("pear", 5));

	fx_max_queue.pop();
	EXPECT_EQ(fx_max_queue.top(), thing("orange", 1));

	fx_max_queue.pop();
	EXPECT_EQ(fx_max_queue.top(), thing("grape", 0));

	fx_max_queue.pop();
	EXPECT_TRUE(fx_max_queue.empty());
}