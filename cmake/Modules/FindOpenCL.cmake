# Simple tool to find the OpenCL library.
#
# Should not be used in full production setups.
# Add this package to your cmake system module directory (or per-project)
# Use the command target_link_libraries to add ${OPENCL_LIBRARY} to your project
# to add -lOpenCL to your project.

message(STATUS "Looking for OpenCL headers and library.")

find_library(OPENCL_LIBRARY OpenCL)
find_path(OPENCL_INCLUDE_DIR CL/cl.hpp)

if(OPENCL_LIBRARY-NOTFOUND)
    message(FATAL_ERROR "OpenCL runtime not found on your platform.")
else(OPENCL_LIBRARY)
    message(STATUS "Found OpenCL runtime.")
endif(OPENCL_LIBRARY-NOTFOUND)

if(OPENCL_INCLUDE_DIR-NOTFOUND)
    message(FATAL_ERROR "OpenCL headers not found on your platform.")
else(OPENCL_INCLUDE_DIR)
    message(STATUS "Found OpenCL headers.")
endif(OPENCL_INCLUDE_DIR-NOTFOUND)

set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
set(OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIR})
