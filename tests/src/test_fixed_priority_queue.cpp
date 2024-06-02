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

