set(
    TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_bbox.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/test_fixed_priority_queue.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/test_kdtree.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/test_point_traits.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

set(KDTREE_TESTS "kdtree-tests")

add_executable(${KDTREE_TESTS} ${TESTS_SRC})

find_package(Threads REQUIRED)

target_include_directories(${KDTREE_TESTS} PRIVATE ${KDTREE_INCLUDE_DIR})
target_link_libraries(${KDTREE_TESTS} gtest Threads::Threads)

gtest_discover_tests(${KDTREE_TESTS})