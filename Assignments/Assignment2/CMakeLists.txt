cmake_minimum_required(VERSION 3.10.0)
project(ASSIGNMENT2 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(ASSIGNMENT2 main.cpp)

target_compile_options(ASSIGNMENT2 PRIVATE -O3 -march=native)

