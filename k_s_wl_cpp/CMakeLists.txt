cmake_minimum_required(VERSION 3.15)
project(k_s_wl_cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(k_s_wl_cpp
        src/AuxiliaryMethods.cpp
        src/AuxiliaryMethods.h
        src/ColorRefinementKernel.cpp
        src/ColorRefinementKernel.h
        src/GenerateTwo.cpp
        src/GenerateTwo.h
        src/GenerateThree.cpp
        src/GenerateThree.h
        src/Graph.cpp
        src/Graph.h
        main.cpp)
