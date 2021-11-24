cmake_minimum_required(VERSION 3.4...3.18)

if (USECUDA)
    project(raytracer CXX CUDA)
else()
    project(raytracer CXX)
endif()


# Compiler settings.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(USER_CXX_FLAGS "-Wall")
    set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
    set(USER_CXX_FLAGS_DEBUG "-g -O0")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(USER_CXX_FLAGS "-Wall -fopenmp")
    set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
    set(USER_CXX_FLAGS_DEBUG "-g -O0")
endif()

if(NOT HAS_CACHE)
    set(CMAKE_CXX_FLAGS ${USER_CXX_FLAGS} CACHE STRING
        "Flags used by the CXX-compiler during all build types." FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE ${USER_CXX_FLAGS_RELEASE} CACHE STRING
        "Flags used by the CXX-compiler during release build types." FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG ${USER_CXX_FLAGS_DEBUG} CACHE STRING
        "Flags used by the CXX-compiler during debug build types." FORCE)

    message(STATUS "Build Type: " ${CMAKE_BUILD_TYPE})
    set(HAS_CACHE TRUE CACHE BOOL "CMakeCache.txt created." FORCE)

    # Make sure that ccmake only contains build type.
    mark_as_advanced(HAS_CACHE)
endif()


# Set the default build type to RELEASE.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE CACHE STRING
        "Choose the type of build, options are: None Debug Release." FORCE)
else()
    string(TOUPPER ${CMAKE_BUILD_TYPE} TEMP)
    set(CMAKE_BUILD_TYPE ${TEMP} CACHE STRING
        "Choose the type of build, options are: None Debug Release." FORCE)
endif()


add_executable(raytracer src/raytracer.cpp)
add_executable(raytracer_3d src/raytracer_3d.cpp)
if (USECUDA)
    add_executable(raytracer_3d_gpu src/raytracer_3d.cu)
endif()
