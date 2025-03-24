/*
▶ Due: March 24 by 2 PM.
▶ Write a function to price European Call options using Black
Scholes formula.
▶ Measure time taken to price 1 million (distinct) options. Use
random data to initialize parameters for each option.
▶ You are not required to use techniques such as
vectorization/multithreading for this assignment.
▶ The Aim of this assignment is to get the students to think
about performance and set the stage for the week 1 lecture.
*/

#include <random>
#include <algorithm>
#include <iostream>
#include <vector>
#include <thread>
#include <Eigen/Dense>

#include "StopWatch.hpp"

inline float norm_cdf(float x) 
{
    static float inv_sqrt = std::sqrtf(2.0f);
    // Use std::erff and std::sqrtf for float functions
    return 0.5f * (1.0f + std::erff(x / inv_sqrt));
}

Eigen::MatrixXf generate_data(size_t size)
{
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<int> unif(10, 200); // Strikes and prices between 10 and 200
    std::uniform_int_distribution<int> unif2(1, 20); // Time to expiry between 3 months and 5 years
    std::uniform_real_distribution<float> real_unif_rates(0.01f, 0.15f); // Interest rates between 1% and 15%
    std::uniform_real_distribution<float> real_unif_vol(0.01f, 0.5f); // Volatility between 1% and 50%

    Eigen::VectorXi prices = Eigen::VectorXi::NullaryExpr(size, [&](){ return unif(mt); });
    Eigen::VectorXi strikes = Eigen::VectorXi::NullaryExpr(size, [&](){ return unif(mt); });
    Eigen::VectorXf rates = Eigen::VectorXf::NullaryExpr(size, [&](){ return real_unif_rates(mt); });
    Eigen::VectorXf volatilities = Eigen::VectorXf::NullaryExpr(size, [&](){ return real_unif_vol(mt); });
    Eigen::VectorXf expirations = Eigen::VectorXf::NullaryExpr(size, [&](){ return unif2(mt) * 0.25f; });

    Eigen::MatrixXf M(size, 5);
    M.col(0) = prices.cast<float>();
    M.col(1) = strikes.cast<float>();
    M.col(2) = rates;
    M.col(3) = volatilities;
    M.col(4) = expirations;

    return M;
}

Eigen::VectorXf generate_call_prices(const Eigen::MatrixXf& matrix)
{
    Eigen::VectorXf S = matrix.col(0);
    Eigen::VectorXf K = matrix.col(1);
    Eigen::VectorXf r = matrix.col(2);
    Eigen::VectorXf sigma = matrix.col(3);
    Eigen::VectorXf T = matrix.col(4);

    Eigen::ArrayXf sqrtT = T.array().sqrt();
    Eigen::ArrayXf d1 = ((S.array() / K.array()).log() + (r.array() + 0.5f * sigma.array().square()) * T.array())
                        / (sigma.array() * sqrtT);
    Eigen::ArrayXf d2 = d1 - sigma.array() * sqrtT;

    Eigen::VectorXf Phi_d1 = d1.unaryExpr([](float x) { return norm_cdf(x); });
    Eigen::VectorXf Phi_d2 = d2.unaryExpr([](float x) { return norm_cdf(x); });

    Eigen::VectorXf callPrices = S.array() * Phi_d1.array()
                                - K.array() * (-r.array() * T.array()).exp() * Phi_d2.array();

    return callPrices;
}

int main()
{
    constexpr size_t N = 10'000'000;
    std::cout << '\n';

    // Generate data in a single thread
    Eigen::MatrixXf data;
    {
        ScopedTimer timerGen("Data Generation Duration: ");
        data = generate_data(N);
    }

    Eigen::VectorXf callPrices(N);

    // Determine the number of threads to use
    const size_t numThreads = std::thread::hardware_concurrency();
    size_t blockSize = N / numThreads; // Size of block per thread
    std::vector<std::thread> threads;

    {
        ScopedTimer timerCalc("Call Prices Calculation Duration: ");

        // Launch threads to compute call prices on blocks of rows
        for (size_t t = 0; t < numThreads; ++t)
        {
            size_t start = t * blockSize;
            // Last thread processes until the end
            size_t currentBlockSize = (t == numThreads - 1) ? (N - start) : blockSize;

            threads.emplace_back([&, start, currentBlockSize]() // Need to capture start and blocksize by value
            {
                // Extract the block corresponding to these rows
                Eigen::MatrixXf block = data.middleRows(start, currentBlockSize);
                // Compute call prices for this block
                Eigen::VectorXf localPrices = generate_call_prices(block);
                // Store the results in the corresponding segment of callPrices
                callPrices.segment(start, currentBlockSize) = localPrices;
            });
        }

        for (auto& thread : threads)
        {
            thread.join();
        }
    }

    std::cout << "\nInput Data (first 5 rows):\n"
              << "\tS\tK\tr\tVol\t\tT\n"
              << data.topRows(5) << "\n\n";
    std::cout << "Call Prices (first 5 values):\n" << callPrices.head(5) << "\n\n";

    return 0;
}
