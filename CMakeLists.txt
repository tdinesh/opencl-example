cmake_minimum_required(VERSION 2.8)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

project(opencl-example)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Werror -pedantic")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
find_package(OpenCL REQUIRED)
find_package(Boost 1.61 COMPONENTS program_options REQUIRED)

add_subdirectory(src)
