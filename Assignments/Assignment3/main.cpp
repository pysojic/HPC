// black_scholes_simple.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Simple error‐check macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " at " << file << ":" << line << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// __device__ helper for the standard normal CDF
__device__ float cdf_normal(float x) {
    // inv√2 = 0.70710678f
    return 0.5f * erfcf(-x * 0.70710678f);
}

// One‐thread‐per‐option kernel: computes call price only
__global__ void BlackScholesKernel(
    const float* S,
    const float* K,
    const float* r,
    const float* v,
    const float* T,
    float*       C,
    int          N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float Si = S[idx], Ki = K[idx];
    float ri = r[idx], vi = v[idx], Ti = T[idx];
    float sqrtT = sqrtf(Ti);

    float d1 = (logf(Si/Ki) + (ri + 0.5f*vi*vi)*Ti) / (vi * sqrtT);
    float d2 = d1 - vi * sqrtT;

    float Nd1 = cdf_normal(d1);
    float Nd2 = cdf_normal(d2);
    float discK = Ki * expf(-ri * Ti);

    C[idx] = Si * Nd1 - discK * Nd2;
}

int main() {
    // ----------------------
    // Part 1: fixed 5 cases
    // ----------------------
    const int M = 5;
    float h_Sm[M] = { 90.f, 95.f, 100.f, 105.f, 110.f };
    float h_Km[M] = { 90.f, 90.f,  90.f, 100.f, 100.f };
    float h_rm[M] = { 0.03f,0.03f,0.03f,0.03f,0.03f };
    float h_vm[M] = { 0.30f,0.30f,0.30f,0.30f,0.30f };
    float h_Tm[M] = { 1.f,   1.f,   1.f,   2.f,   2.f   };
    float h_Cm[M];

    // Device buffers for small run
    float *d_Sm, *d_Km, *d_rm, *d_vm, *d_Tm, *d_Cm;
    CUDA_CHECK(cudaMalloc(&d_Sm, M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Km, M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rm, M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vm, M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Tm, M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Cm, M*sizeof(float)));

    // Copy fixed inputs
    CUDA_CHECK(cudaMemcpy(d_Sm, h_Sm, M*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Km, h_Km, M*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rm, h_rm, M*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vm, h_vm, M*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Tm, h_Tm, M*sizeof(float), cudaMemcpyHostToDevice));

    // Launch a single block of M threads
    BlackScholesKernel<<<1, M>>>(d_Sm, d_Km, d_rm, d_vm, d_Tm, d_Cm, M);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Retrieve results
    CUDA_CHECK(cudaMemcpy(h_Cm, d_Cm, M*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Part 1: Call prices for fixed scenarios\n";
    for (int i = 0; i < M; ++i) {
        std::cout << "S=" << h_Sm[i]
                  << "  K=" << h_Km[i]
                  << "  T=" << h_Tm[i]
                  << "  Call=" << h_Cm[i] << "\n";
    }

    // Cleanup small buffers
    cudaFree(d_Sm); cudaFree(d_Km);
    cudaFree(d_rm); cudaFree(d_vm);
    cudaFree(d_Tm); cudaFree(d_Cm);

    // -----------------------------------
    // Part 2: 1,000,000 random on the GPU
    // -----------------------------------
    const int N = 1'000'000;
    std::vector<float> h_S(N), h_K(N), h_r(N), h_v(N), h_T(N), h_C(N);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> distS(50.f,150.f),
                                       distr(0.f,0.10f),
                                       distv(0.1f,0.5f),
                                       distT(0.1f,3.0f);
    for (int i = 0; i < N; ++i) {
        h_S[i] = distS(rng);
        h_K[i] = distS(rng);
        h_r[i] = distr(rng);
        h_v[i] = distv(rng);
        h_T[i] = distT(rng);
    }

    // Device buffers
    float *d_S, *d_K, *d_r, *d_v, *d_T, *d_C;
    CUDA_CHECK(cudaMalloc(&d_S, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N*sizeof(float)));

    // Time from H2D through D2H (includes copies + kernel)
    auto t1 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(d_S, h_S.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T, h_T.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    // launch with 256 threads/block
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    BlackScholesKernel<<<blocks, threads>>>(d_S, d_K, d_r, d_v, d_T, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N*sizeof(float), cudaMemcpyDeviceToHost));

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;

    std::cout << "\nPart 2: 1,000,000 call prices computed on GPU\n"
              << "Elapsed time (H2D + kernel + D2H): "
              << elapsed.count() << " s\n";

    // Final cleanup
    cudaFree(d_S); cudaFree(d_K);
    cudaFree(d_r); cudaFree(d_v);
    cudaFree(d_T); cudaFree(d_C);

    return 0;
}
