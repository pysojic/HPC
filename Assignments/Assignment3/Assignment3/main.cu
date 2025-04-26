#include <iostream>
#include <iomanip>
#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

struct OptionData {
    float S, K, r, v, T;
};

struct OptionResults {
    float callPrice, putPrice;
    float callDelta, putDelta;
    float gamma, vega;
    float callRho, putRho;
    float callTheta, putTheta;
};

__host__ __device__ inline float norm_pdf(float x) {
    return expf(-0.5f*x*x) * rsqrtf(2.0f * M_PI);
}

__host__ __device__ inline float norm_cdf(float x) {
    return 0.5f * erfcf(-x * M_SQRT1_2);
}

__host__ __device__
OptionResults blackScholes(const OptionData& o) {
    float sqrtT = sqrtf(o.T), vsqrt = o.v * sqrtT;
    float d1 = (logf(o.S/o.K) + (o.r + 0.5f*o.v*o.v)*o.T)/vsqrt;
    float d2 = d1 - vsqrt;
    float cdf1 = norm_cdf(d1), cdf2 = norm_cdf(d2), pdf1 = norm_pdf(d1);
    float expRT = expf(-o.r * o.T);
    OptionResults r;
    r.callPrice = o.S*cdf1 - o.K*expRT*cdf2;
    r.putPrice  = o.K*expRT*(1.0f - cdf2) - o.S*(1.0f - cdf1);
    r.callDelta = cdf1;
    r.putDelta  = cdf1 - 1.0f;
    r.gamma     = pdf1/(o.S*o.v*sqrtT);
    r.vega      = o.S*sqrtT*pdf1*0.01f;
    r.callRho   = o.K*o.T*expRT*cdf2*0.01f;
    r.putRho    = -o.K*o.T*expRT*(1.0f - cdf2)*0.01f;
    r.callTheta = (-o.S*o.v*pdf1/(2.0f*sqrtT) - o.r*o.K*expRT*cdf2)/365.0f;
    r.putTheta  = (-o.S*o.v*pdf1/(2.0f*sqrtT) + o.r*o.K*expRT*(1.0f - cdf2))/365.0f;
    return r;
}

__global__
void bsKernel(const OptionData* opts, OptionResults* res, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N) res[idx] = blackScholes(opts[idx]);
}

int main() {
    OptionData presets[5] = {
        {  90,  90, 0.03f, 0.3f, 1.0f },
        {  95,  90, 0.03f, 0.3f, 1.0f },
        { 100, 100, 0.03f, 0.3f, 2.0f },
        { 105, 100, 0.03f, 0.3f, 2.0f },
        { 110, 100, 0.03f, 0.3f, 2.0f }
    };
    char labels[5] = {'a','b','c','d','e'};
    std::cout << "\nPreset Options:\n";
    std::cout << std::left
              << std::setw(6) << "Label"
              << std::setw(8) << "S"
              << std::setw(8) << "K"
              << std::setw(10) << "Call"
              << std::setw(10) << "Put"
              << std::setw(10) << "CΔ"
              << std::setw(10) << "PΔ"
              << std::setw(10) << "Gamma"
              << std::setw(10) << "Vega"
              << "\n"
              << std::string(72, '-') << "\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < 5; ++i) {
        auto r = blackScholes(presets[i]);
        std::cout << std::left
                  << std::setw(6) << labels[i]
                  << std::setw(8) << presets[i].S
                  << std::setw(8) << presets[i].K
                  << std::setw(10) << r.callPrice
                  << std::setw(10) << r.putPrice
                  << std::setw(10) << r.callDelta
                  << std::setw(10) << r.putDelta
                  << std::setw(10) << r.gamma
                  << std::setw(10) << r.vega
                  << "\n";
    }

    const int N = 1'000'000;
    OptionData* opts = nullptr;
    OptionResults* res = nullptr;
    cudaMallocManaged(&opts, N * sizeof*opts);
    cudaMallocManaged(&res,  N * sizeof*res);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dS(80,120), dR(0.01f,0.05f), dV(0.1f,0.5f), dT(0.25f,2);
    for (int i = 0; i < N; ++i)
        opts[i] = {dS(rng), dS(rng), dR(rng), dV(rng), dT(rng)};

    auto t0 = std::chrono::high_resolution_clock::now();
    int bs = 256, gb = (N+bs-1)/bs;
    bsKernel<<<gb,bs>>>(opts, res, N);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nGPU batch: computed "
              << N << " options in "
              << std::fixed << std::setprecision(3)
              << sec << " s\n\n";

    cudaFree(opts);
    cudaFree(res);
    return 0;
}

