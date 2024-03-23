project(utils_test)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)


add_executable(nompi_utils_test nompi_utils_test.cpp)
target_link_libraries(nompi_utils_test
        PRIVATE
        Catch2::Catch2WithMain
        GREEN::UTILS)

add_executable(utils_test utils_test.cpp main_test.cpp)
target_link_libraries(utils_test
        PRIVATE
        Catch2::Catch2
        GREEN::UTILS)

add_executable(mpi_utils_test mpi_utils_test.cpp
        main_test.cpp)
target_link_libraries(mpi_utils_test
        PRIVATE
        Catch2::Catch2
        GREEN::UTILS)


include(CTest)
include(Catch)
catch_discover_tests(utils_test)
catch_discover_tests(nompi_utils_test)
