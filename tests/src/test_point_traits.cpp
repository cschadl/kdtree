#include <gtest/gtest.h>

#include <kdtree/point_traits.hpp>

using namespace cds::kdtree;

namespace
{
	using point2d_t = std::array<double, 2>;
	using point3d_t = std::array<double, 3>;
}


TEST(point_traits, create)
{
	point3d_t point = point_traits<point3d_t>::create(12.0);
	
	EXPECT_EQ(point[0], 12.0);
	EXPECT_EQ(point[1], 12.0);
	EXPECT_EQ(point[2], 12.0);
}