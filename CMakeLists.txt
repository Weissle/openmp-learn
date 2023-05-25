cmake_minimum_required(VERSION 3.6)
project(openmp)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

set(CMAKE_CXX_STANDARD 17)

add_executable(main main.cpp)
 