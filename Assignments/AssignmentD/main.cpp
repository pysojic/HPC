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