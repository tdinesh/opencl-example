set(sources
    main.cpp
    )

add_executable(example ${sources})

target_link_libraries(example
    ${OPENCL_LIBRARIES}
    )
