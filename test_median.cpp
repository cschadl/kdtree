#include "kdtree.h"

#include <iostream>
#include <array>
#include <vector>

using point_t = std::array<double, 3>;

template <>
struct point_traits<point_t>
{
	using value_type = point_t::value_type;
	static constexpr size_t dim() { return 3; }
};

int main(int argc, char** argv)
{
	std::vector<point_t> points = {
			{ 0., 1., 2. },
			{ 1., 3., 8. },
			{ 2., 4., 7. },
			{ 6., 0., 1. } };

	auto median_x_it = kd_tree<point_t>::get_median(points.begin(), points.end(), 0);
	auto median_y_it = kd_tree<point_t>::get_median(points.begin(), points.end(), 1);
	auto median_z_it = kd_tree<point_t>::get_median(points.begin(), points.end(), 2);

	std::cout << "The X median is " << (*median_x_it)[0] << std::endl;
	std::cout << "The Y median is " << (*median_y_it)[1] << std::endl;
	std::cout << "The Z median is " << (*median_z_it)[2] << std::endl;

	return 1;
}
