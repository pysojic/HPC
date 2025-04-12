#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm> 

struct OptionParams
{
    OptionParams(double s, double k, double t, double sig, double rate, bool type)
        : S{s}, K{k}, T{t}, vol{sig}, r{rate}, OptionType{type}
    {}
    double S;
    double K;
    double T;
    double vol;
    double r;
    bool OptionType;
};

// Function to price a European option (call or put) using a Jarrow-Rudd binomial tree
double PriceOptionJR(const OptionParams& params, size_t N) 
{
    double dt = params.T / N;
    
    double u = exp((params.r - 0.5 * params.vol * params.vol) * dt + params.vol * sqrt(dt));
    double d = exp((params.r - 0.5 * params.vol * params.vol) * dt - params.vol * sqrt(dt));
    
    double p = 0.5;
    
    double discount = exp(-params.r * dt);
    
    std::vector<double> optionValues(N + 1);
    
    for (int j = 0; j <= N; j++) 
    {
        double S = params.S * pow(u, j) * pow(d, N - j);
        if (params.OptionType) 
        {
            optionValues[j] = std::max(S - params.K, 0.0);
        } else
        {
            optionValues[j] = std::max(params.K - S, 0.0);
        }
    }
    
    for (int i = N - 1; i >= 0; i--) 
    {
        for (int j = 0; j <= i; j++) 
        {
            optionValues[j] = discount * (p * optionValues[j + 1] + (1 - p) * optionValues[j]);
        }
    }
    
    return optionValues[0];
}

int main()
{
    std::cout << "--------Part 1--------\n";
    size_t N = 1000;
    OptionParams option1(90.0, 100.0, 1.0, 0.3, 0.03, true);
    OptionParams option2(95.0, 100.0, 1.0, 0.3, 0.03, true);
    OptionParams option3(100.0, 100.0, 1.0, 0.3, 0.03, true);
    OptionParams option4(105.0, 100.0, 1.0, 0.3, 0.03, true);
    OptionParams option5(110.0, 100.0, 1.0, 0.3, 0.03, true);

    double optionPrice1 = PriceOptionJR(option1, N);
    double optionPrice2 = PriceOptionJR(option2, N);
    double optionPrice3 = PriceOptionJR(option3, N);
    double optionPrice4 = PriceOptionJR(option4, N);
    double optionPrice5 = PriceOptionJR(option5, N);


    std::cout << "Option1 price: " << optionPrice1 << std::endl;
    std::cout << "Option2 price: " << optionPrice2 << std::endl;
    std::cout << "Option3 price: " << optionPrice3 << std::endl;
    std::cout << "Option4 price: " << optionPrice4 << std::endl;
    std::cout << "Option5 price: " << optionPrice5 << std::endl;

    
    std::cout << "\n--------Part 2--------\n";

    N = 1'000'00;
    OptionParams option6(110.0, 100.0, 1.0, 0.3, 0.03, true);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double optionPrice6 = PriceOptionJR(option6, N);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Elapsed time: " <<
    std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " ms\n";

    std::cout << "Option (call) price: " << optionPrice5 << std::endl;

    return 0;
}