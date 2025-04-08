cmake_minimum_required(VERSION 3.10.0)
project(ASSIGNMENT1 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(ASSIGNMENT1 Assignment1.cpp)

target_compile_options(ASSIGNMENT1 PRIVATE -O2 -march=native)

