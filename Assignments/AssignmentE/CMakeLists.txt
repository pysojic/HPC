cmake_minimum_required(VERSION 3.10.0)
project(HPC_ASSIGNMENT_E VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(HPC_ASSIGNMENT_E main.cpp)

target_compile_options(HPC_ASSIGNMENT_E PRIVATE -O2 -march=native)