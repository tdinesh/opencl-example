# Simple tool to find the OpenCL library.
#
# Should not be used in full production setups.
# Add this package to your cmake system module directory (or per-project)
# Use the command target_link_libraries to add ${OPENCL_LIBRARY} to your project
# to add -lOpenCL to your project.

find_library(OPENCL_LIBRARY OpenCL)
find_path(OPENCL_INCLUDE_DIR CL/cl.hpp)

set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
set(OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIR})
