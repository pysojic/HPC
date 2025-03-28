/* 
To build and run this program you must run the following commands (Mac):
1) "mkdir build" (Create the build folder)
2) "cd build" (Navigate tot the build folder)
3) "cmake .." (Run Cmake and point it to the parent directory)
4) "cmake --build ." (Build the project)
5) "./HPC_ASSIGNMENT_B" (Run the executable)
*/

/*
▶ Write a function to multiply two NxN matrices.
▶ Use the function you wrote to multiply two 100x100 matrices.
▶ Measure the execution time.
*/

#include <iostream>
#include <Eigen/Dense>
#include "../Utilities/StopWatch.hpp"
#include "include/Matrix.h"
#include <random>

int main()
{
    Eigen::Matrix<int, 100, 100> matrix1 = Eigen::MatrixXi::NullaryExpr(100, 100, [] { return std::rand() % 10; });
    Eigen::Matrix<int, 100, 100> matrix2 = Eigen::MatrixXi::NullaryExpr(100, 100, [] { return std::rand() % 10;;});

    // std::cout << matrix1 << std::endl;
    // std::cout << matrix2 << std::endl;

    Eigen::Matrix<int, 100, 100> matrix3;
    {
        ScopedTimer timer{"Matrix multiplication duration (Eigen): "};
        matrix3 = matrix1 * matrix2;
    }
    
    //std::cout << matrix3 << std::endl;

    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<int> dist(0, 100);

    Matrix<int, 100, 100> m(dist, mt);
    Matrix<int, 100, 100> m2(dist, mt);

    Matrix<int, 100, 100> mult;
    {
        ScopedTimer timer{"Matrix multiplication duration (STL): "};
        mult = matrix_mult(m, m2); 
    }   

    // mult.print();
}