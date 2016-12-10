set(sources
    example.cpp)

add_executable(example ${sources})

include_directories(${Boost_INCLUDE_DIR})
target_link_libraries(example
    ${OPENCL_LIBRARIES}
    ${Boost_LIBRARIES})
