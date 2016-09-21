project(opencl-example)

cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Werror -pedantic")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
find_package(OpenCL REQUIRED)

add_subdirectory(src)
