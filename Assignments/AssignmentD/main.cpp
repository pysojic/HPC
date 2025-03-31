/*
▶Calculating the value of π using Monte Carlo involves the following steps:
1. Assume the circle is centered at coordinates (0, 0).
2. Generate N random points with coordinates (x, y) where x and y are independently drawn from a 
   uniform distribution over the interval [-1, 1].
3. Determine if each point lies inside the circle or not.
▶ The value of π can be estimated using the simulation results as follows.
ρ = M/N = π/4
where, N = total number of points generated and M = number of random points inside the circle.
▶ Write a program to calculate the value of π using information above, for N = 100, 1000 and 10000. 
▶Write the results to the standard output.
*/

#include <iostream>
#include <random>
#include <format>

int main()
{
    std::array<size_t, 10> trials_numbers{100, 1'000, 5000, 10'000, 50000, 100'000, 1'000'000, 10'000'000, 50'000'000, 100'000'000};

    for (auto N : trials_numbers)
    {
        size_t M = 0;

        std::random_device rd;
        std::mt19937_64 mt(rd());
        std::uniform_real_distribution unif(-1.0, 1.0);

        for (size_t i = 0; i < N; ++i)
        {
            double x = unif(mt);
            double y = unif(mt);

            if (std::sqrt(x*x + y*y) <= 1.0) 
                ++M;
        }

        double pi = static_cast<double>(M) / N * 4;

        std::cout << std::format("The value of pi with {} trials is approximately: {}", N, pi) << std::endl;
    }
}