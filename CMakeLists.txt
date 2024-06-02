cmake_minimum_required(VERSION 3.16)
project("kdtree" VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(KDTREE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(GTest REQUIRED)
enable_testing()
include(GoogleTest)

add_subdirectory(tests)