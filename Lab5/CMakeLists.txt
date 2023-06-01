cmake_minimum_required(VERSION 3.25)
project(Lab5)

set(CMAKE_CXX_STANDARD 11)
add_compile_options(-mavx2 -march=haswell)
add_executable(Lab5 src/task3.cpp)
