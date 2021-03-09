cmake_minimum_required(VERSION 3.4...3.18)

set(ENV{CXX} g++-10)

project(raytracer CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the default build type to RELEASE.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE CACHE STRING
        "Choose the type of build, options are: None Debug Release." FORCE)
else()
    string(TOUPPER ${CMAKE_BUILD_TYPE} TEMP)
    set(CMAKE_BUILD_TYPE ${TEMP} CACHE STRING
        "Choose the type of build, options are: None Debug Release." FORCE)
endif()

if(NOT HAS_CACHE)
    set(CMAKE_CXX_FLAGS "-fopenmp" CACHE STRING
        "Flags used by the CXX-compiler during all build types." FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native" CACHE STRING
        "Flags used by the CXX-compiler during release build types." FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING
        "Flags used by the CXX-compiler during debug build types." FORCE)

    message(STATUS "Build Type: " ${CMAKE_BUILD_TYPE})
    set(HASCACHE TRUE CACHE BOOL "CMakeCache.txt created." FORCE)

    # Make sure that ccmake only contains build type.
    mark_as_advanced(HAS_CACHE)
endif()

add_executable(raytracer src/raytracer.cpp)
