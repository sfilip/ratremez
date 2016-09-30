cmake_minimum_required (VERSION 2.8.8)
set(PROJECT_NAME_STR ratremez)
project(${PROJECT_NAME_STR})
# The project version number.
set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   1   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   1   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)
set(doxy_main_page ${PROJECT_SOURCE_DIR}/README.md)

#----------------------------------------
# dependencies testing
#----------------------------------------
list (INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(Eigen3 REQUIRED)
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)
find_package(GTest)

#----------------------------------------
# build configuration
#----------------------------------------
set(CMAKE_BUILD_TYPE Release)
add_definitions(-std=c++11)

#----------------------------------------
# common include directories
#----------------------------------------
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)


#----------------------------------------
# tests
#----------------------------------------
if(GTEST_FOUND)
    enable_testing()
    set(PROJECT_TEST_RAT ${PROJECT_NAME_STR}_tests)
    set(CMAKE_CTEST_COMMAND ctest -V)
    include_directories(${GTEST_INCLUDE_DIR} ${COMMON_INCLUDES})


    set(TEST_SRC_RAT ${PROJECT_SOURCE_DIR}/test/rat_tests.cpp)

    add_executable(${PROJECT_TEST_RAT} ${TEST_SRC_RAT})

    target_link_libraries(${PROJECT_TEST_RAT}
        gtest
        gtest_main
        pthread
        ratremez
        gmp
        mpfr
        gomp
    )


add_test(RationalTests ${PROJECT_TEST_RAT})
endif()
