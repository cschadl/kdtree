#include "bbox.h"

#include <iostream>

using point2d_t = std::array<double, 2>;
using point3d_t = std::array<double, 3>;
using bbox2d_t = bbox<point2d_t>;
using bbox3d_t = bbox<point3d_t>;

int main(int argc, char** argv)
{
	{
		bbox2d_t bbox1( {-4, 1}, {6, 6} );
		bbox2d_t bbox2( {-1,-3}, {4, 7} );

		if (bbox1.disjoint(bbox2))
		{
			std::cout << "Intersection test failed!" << std::endl;
		}
		else
		{
			std::cout << "Intersection test succeeded!" << std::endl;
		}
	}

	{
		bbox2d_t bbox1( {-4, -8}, {-2, 2} );
		bbox2d_t bbox2( { 2, 0 }, { 4, 4} );

		if (!bbox1.disjoint(bbox2))
		{
			std::cout << "Disjoint test failed!" << std::endl;
		}
		else
		{
			std::cout << "Disjoint test succeeded!" << std::endl;
		}
	}
}
