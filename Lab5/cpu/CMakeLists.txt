cmake_minimum_required(VERSION 3.25)
project(Lab5)

set(CMAKE_CXX_STANDARD 14)
#add_executable(Lab5 src/task1.cpp)
add_compile_options(-mavx2 -march=haswell)
#add_executable(Lab5 src/task2.cpp)
add_executable(Lab5 src/task3.cpp )
