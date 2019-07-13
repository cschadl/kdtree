#include "kdtree.h"

#include <iostream>
#include <array>
#include <vector>

using point_t = std::array<double, 2>;

template <>
struct point_traits<point_t>
{
	using value_type = point_t::value_type;
	static constexpr size_t dim() { return 2; }
};

int main(int argc, char** argv)
{
	std::vector<point_t> points = {
			{ 2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}
	};

	kd_tree<point_t> tree;
	tree.build(points.begin(), points.end());

	std::cout << "Created tree" << std::endl;
}
