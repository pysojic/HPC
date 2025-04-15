#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

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

OptionParams generate_params()
{
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<int> prices(1, 200);
    std::uniform_real_distribution<double> rates(0.01, 0.15);
    std::uniform_real_distribution<double> vol(0.1, 0.4);
    std::uniform_int_distribution<int> time(1, 10);
    std::bernoulli_distribution call_put(0.5);

    return OptionParams((double)prices(mt), (double)prices(mt), (double)time(mt), vol(mt), rates(mt), call_put(mt));
}

double PriceOptionJR_MT(const OptionParams& params, size_t N) 
{
    double dt = params.T / N;
    
    double u = exp((params.r - 0.5 * params.vol * params.vol) * dt + params.vol * sqrt(dt));
    double d = exp((params.r - 0.5 * params.vol * params.vol) * dt - params.vol * sqrt(dt));

    double p = 0.5;
    double discount = exp(-params.r * dt);
    
    std::vector<double> prevRow(N + 1);
    double s = params.S * pow(d, N);   
    double ratio = u / d;              
    
    // Sequential initialization because of dependence on s
    for (size_t j = 0; j <= N; ++j)
    {
        if (params.OptionType) 
        {
            prevRow[j] = std::max(s - params.K, 0.0);
        } 
        else 
        {
            prevRow[j] = std::max(params.K - s, 0.0);
        }
        s *= ratio; 
    }
    
    std::vector<double> currRow(N);
    
    // Backward induction: outer loop is sequential but inner loop is parallelized
    for (int i = N - 1; i >= 0; i--) 
    {
        #pragma omp parallel for
        for (int j = 0; j <= i; j++) 
        {
            currRow[j] = discount * (p * prevRow[j + 1] + (1 - p) * prevRow[j]);
        }
        prevRow.swap(currRow);
    }
    
    return prevRow[0];
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
    N = 1'000'000;

    auto option_single_threaded = generate_params();
    std::string call_put = option_single_threaded.OptionType ? "Call" : "Put";

    auto t1 = std::chrono::high_resolution_clock::now();
    double price = PriceOptionJR(option_single_threaded, N);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "\nSINGLE-THREADED\n---------------------\nOption Parameters:\n\n"
              << "- Option Type: " << call_put << '\n'
              << "- Stock Price: " << option_single_threaded.S << '\n'
              << "- Strike Price: " << option_single_threaded.K << '\n'
              << "- Time to maturity: " << option_single_threaded.T << " years\n"
              << "- Interest Rate: " << option_single_threaded.r << '\n'
              << "- Volatility: " << option_single_threaded.vol << '\n';
    std::cout << "\nOption (" << call_put << ") price (single-threaded): " << price << '\n';
    std::cout << "\nElapsed time (single-threaded): " <<
    std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " microseconds" << "\n---------------------\n";


    auto option_multi_threaded = generate_params();
    call_put = option_multi_threaded.OptionType ? "Call" : "Put";

    auto t3 = std::chrono::high_resolution_clock::now();
    double price2 = PriceOptionJR_MT(option_multi_threaded, N);
    auto t4 = std::chrono::high_resolution_clock::now();

    std::cout << "\nMULTI-THREADED\n---------------------\nOption Parameters:\n\n"
              << "- Option Type: " << call_put << '\n'
              << "- Stock Price: " << option_multi_threaded.S << '\n'
              << "- Strike Price: " << option_multi_threaded.K << '\n'
              << "- Time to maturity: " << option_multi_threaded.T << " years\n"
              << "- Interest Rate: " << option_multi_threaded.r << '\n'
              << "- Volatility: " << option_multi_threaded.vol << '\n';
    std::cout << "\nOption price (multi-threaded): " << price2 << '\n';
    std::cout << "\nElapsed time (multi-threaded): " <<
    std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << " microseconds" << "\n---------------------\n";

    return 0;
}
