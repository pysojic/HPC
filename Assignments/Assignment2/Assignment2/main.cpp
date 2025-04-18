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

#include <cmath>
#include <algorithm>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

double PriceOptionJR_MT(const OptionParams& params, size_t N) 
{
    double dt = params.T / static_cast<double>(N);
    
    double u = exp((params.r - 0.5 * params.vol * params.vol) * dt + params.vol * sqrt(dt));
    double d = exp((params.r - 0.5 * params.vol * params.vol) * dt - params.vol * sqrt(dt));
    
    double p = 0.5;
    double discount = exp(-params.r * dt);
    
    std::vector<double> prevRow(N + 1);
    std::vector<double> currRow(N);
    
    double log_d = log(d);
    double log_u = log(u);
    
    #pragma omp parallel for simd schedule(static)
    for (size_t j = 0; j <= N; j++) 
    {
        double exponent = (static_cast<double>(N) - j) * log_d + j * log_u;
        double s_val = params.S * exp(exponent);
        if (params.OptionType)
            prevRow[j] = std::max(s_val - params.K, 0.0);
        else
            prevRow[j] = std::max(params.K - s_val, 0.0);
    }
    
    for (int i = static_cast<int>(N) - 1; i >= 0; i--) 
    {
        #pragma omp parallel for simd schedule(static)
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
    OptionParams call1(90.0, 100.0, 1.0, 0.3, 0.03, true);
    OptionParams call2(95.0, 100.0, 1.0, 0.3, 0.03, true);
    OptionParams call3(100.0, 100.0, 1.0, 0.3, 0.03, true);
    OptionParams call4(105.0, 100.0, 1.0, 0.3, 0.03, true);
    OptionParams call5(110.0, 100.0, 1.0, 0.3, 0.03, true);

    double callPrice1 = PriceOptionJR(call1, N);
    double callPrice2 = PriceOptionJR(call2, N);
    double callPrice3 = PriceOptionJR(call3, N);
    double callPrice4 = PriceOptionJR(call4, N);
    double callPrice5 = PriceOptionJR(call5, N);

    OptionParams put1(90.0, 100.0, 1.0, 0.3, 0.03, false);
    OptionParams put2(95.0, 100.0, 1.0, 0.3, 0.03, false);
    OptionParams put3(100.0, 100.0, 1.0, 0.3, 0.03, false);
    OptionParams put4(105.0, 100.0, 1.0, 0.3, 0.03, false);
    OptionParams put5(110.0, 100.0, 1.0, 0.3, 0.03, false);

    double putPrice1 = PriceOptionJR(put1, N);
    double putPrice2 = PriceOptionJR(put2, N);
    double putPrice3 = PriceOptionJR(put3, N);
    double putPrice4 = PriceOptionJR(put4, N);
    double putPrice5 = PriceOptionJR(put5, N);

    std::cout << "Call1 price: " << callPrice1 << std::endl;
    std::cout << "Call2 price: " << callPrice2 << std::endl;
    std::cout << "Call3 price: " << callPrice3 << std::endl;
    std::cout << "Call4 price: " << callPrice4 << std::endl;
    std::cout << "Call5 price: " << callPrice5 << std::endl;

    std::cout << "\nPut1 price: " << putPrice1 << std::endl;
    std::cout << "Put2 price: " << putPrice2 << std::endl;
    std::cout << "Put3 price: " << putPrice3 << std::endl;
    std::cout << "Put4 price: " << putPrice4 << std::endl;
    std::cout << "Put5 price: " << putPrice5 << std::endl;


    std::cout << "\n--------Part 2--------\n";
    N = 1'000'000;

    auto option_single_threaded = generate_params();
    option_single_threaded.OptionType = 1;
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
    std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds" << "\n---------------------\n";


    auto option_multi_threaded = generate_params();
    option_multi_threaded.OptionType = 1;
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
    std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count() << " seconds" << "\n---------------------\n";

    return 0;
}



