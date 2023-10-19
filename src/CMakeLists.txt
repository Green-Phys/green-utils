project(utils_lib)

find_package(MPI COMPONENTS C CXX REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(utils INTERFACE)
target_link_libraries(utils INTERFACE MPI::MPI_CXX)
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
