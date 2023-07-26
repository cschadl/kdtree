set(
    TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_bbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_fixed_priority_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_kdtree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_point_traits.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tester.cpp
)

add_executable("kdtree-tests" ${TESTS_SRC})