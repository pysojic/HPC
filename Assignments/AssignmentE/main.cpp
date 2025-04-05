#include <iostream>
#include <random>
#include <queue>

int main()
{
    size_t N = 1'000'000;

    double S = 50;
    double K = 30;
    double r = 0.03;
    double T = 1.0;
    double sig = 0.25;

    std::queue<double> normals;
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::normal_distribution<double> nrm(0, 1);

    for (size_t i = 0; i < N; ++i)
    {
        normals.push(nrm(mt));
    }

    double optionPrice = 0;
    double factor1 = (r - (sig * sig) * 0.5)*T;
    double factor2 = sig * std::sqrt(T);
    double disc = std::exp(-r * T);

    while (!normals.empty())
    {
        double nrm = normals.front();
        normals.pop();
        double path = S*std::exp(factor1 + factor2 * nrm);
        optionPrice += disc * std::max(path - K, 0.0);
    }

    std::cout << "Price: " << optionPrice / N << std::endl; 

    return 0;
}