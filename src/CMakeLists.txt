project(utils_lib)

find_package(MPI COMPONENTS C CXX REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(utils mpi_utils.cpp)
target_link_libraries(utils PUBLIC MPI::MPI_CXX)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
