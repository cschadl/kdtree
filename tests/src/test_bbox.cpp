// Copyright (C) 2018 by Christopher Schadl <cschadl@gmail.com>

// Permission to use, copy, modify, and/or distribute this software for any purpose
// with or without fee is hereby granted.

// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD 
// TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
// DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
// WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
// ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

// Unit tests for bbox class
#include <gtest/gtest.h>

#include <kdtree/bbox.hpp>

#include <iostream>
#include <array>

using namespace cds::kdtree;

namespace
{
	using point2d_t = std::array<double, 2>;
	using point3d_t = std::array<double, 3>;
	using bbox2d_t = bbox<point2d_t>;
	using bbox3d_t = bbox<point3d_t>;
}

TEST(bbox, DefaultConstructor)
{
	point3d_t origin{0, 0, 0};

	bbox3d_t bbox_empty;
	EXPECT_TRUE(bbox_empty.empty());
	EXPECT_EQ(bbox_empty.min(), origin);
	EXPECT_EQ(bbox_empty.max(), origin);
	EXPECT_TRUE(bbox_empty.valid());
}

TEST(bbox, Invalid)
{
	bbox2d_t bbox_invalid( { 1.0, 1.0 }, { -1.0, -1.0 } );
	ASSERT_FALSE(bbox_invalid.valid());
}

TEST(bbox, intersects)
{
	bbox2d_t bbox1( {-4,-1}, {6, 6} );
	bbox2d_t bbox2( {-1,-3}, {4, 7} );

	EXPECT_TRUE(bbox1.intersects(bbox2));
	EXPECT_TRUE(bbox2.intersects(bbox1));
}

TEST(bbox, disjoint)
{
	bbox2d_t bbox1( {-4, -8}, {-2, 2} );
	bbox2d_t bbox2( { 2, 0 }, { 4, 4} );

	EXPECT_TRUE(bbox1.disjoint(bbox2));
	EXPECT_TRUE(bbox2.disjoint(bbox1));
}

TEST(bbox, contains)
{
	bbox2d_t bbox({2, 0} , { 4, 4});
	EXPECT_TRUE(bbox.contains({2, 2}));
	EXPECT_FALSE(bbox.contains({-2, 2}));
}

TEST(bbox, intersection)
{
	bbox2d_t bbox1( {-4,-1}, { 6, 6} );
	bbox2d_t bbox2( {-1,-3}, { 4, 7} );

	bbox2d_t bbox1_2({1, 1}, {-1, -1});
	EXPECT_TRUE(bbox1.intersection(bbox2, bbox1_2));
	EXPECT_EQ(bbox1_2, bbox2d_t({-1, -1}, {4, 6}));
}

TEST(bbox, split)
{
	bbox2d_t bbox( {-1,-3}, { 4, 7} );

	bbox2d_t bbox_lt = { { 1, 1}, {-1, -1}};
	bbox2d_t bbox_gt = { { 1, 1}, {-1, -1}};

	EXPECT_TRUE(bbox.split(0, 1.5, bbox_lt, bbox_gt));
	EXPECT_EQ(bbox_lt, bbox2d_t({ -1, -3 }, {1.5, 7}));
	EXPECT_EQ(bbox_gt, bbox2d_t({ 1.5,-3}, { 4, 7 }));
}
