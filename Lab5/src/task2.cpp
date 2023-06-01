//
// Created by Jeffrey on 6/1/2023.
//
#include "bits/stdc++.h"
#include "immintrin.h"

int N = (1 << 10);

inline void gemm_verify(float *A, float *B, float *C) {

}

inline void gemm_avx(float *A, float *B, float *C) {

}

int main() {
    // malloc A, B, C
    auto A = (float *) malloc(N * N * sizeof(float));
    auto B = (float *) malloc(N * N * sizeof(float));
    auto C = (float *) malloc(N * N * sizeof(float));
    // random initialize A, B
    std::uniform_real_distribution<float> distribution(-100, 100);
    std::default_random_engine generator;
    for (int i = 0; i < N * N; i++) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }
    // measure time
    auto start = std::chrono::high_resolution_clock::now();
    gemm_avx(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "elapsed time: " << elapsed.count() * 1000 << "ms\n";
    // use gemm_baseline verify gemm_avx
    gemm_verify(A, B, C);
    return 0;
}
