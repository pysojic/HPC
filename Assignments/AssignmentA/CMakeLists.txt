cmake_minimum_required(VERSION 3.10.0)
project(HPC1 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3; NO_MODULE means CMake will use the config file installed by Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(include)

add_executable(HPC1 Assignment_A.cpp)

target_compile_options(HPC1 PRIVATE -O2 -march=native) # Use for debug: -O0 -g -Wall -Wextra -Wpedantic

# Link Eigen3 to the target
target_link_libraries(HPC1 PRIVATE Eigen3::Eigen)
