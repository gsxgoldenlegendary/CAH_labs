//
// Created by Jeffrey on 6/1/2023.
//
#include <cmath>

#include "bits/stdc++.h"
#include "immintrin.h"

int N = (1 << 11);
int M = (1 << 3);

inline void gemm_verify(const float *A, const float *B, float *C) {
    auto C_baseline = (float *) malloc(N * N * sizeof(float));
    memset(C_baseline, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C_baseline[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    for (int i = 0; i < N * N; i++) {
        if (std::fabs(C[i] - C_baseline[i]) > 1e-2) {
            std::cout << "gemm_avx wrong answer\n";
            return;
        }
    }
}

inline void gemm_block(const float *A, const float *B, float *C) {
    for (int i = 0; i < N; i += M) {
        for (int j = 0; j < N; j += M) {
            for (int k = 0; k < N; k++) {
                for (int ii = i; ii < i + M; ii++) {
                    auto a = _mm256_set1_ps(*(A + ii * N + k));
                    for (int jj = j; jj < j + M; jj += 8) {
                        auto b = _mm256_loadu_ps(B + k * N + jj);
                        auto c = _mm256_loadu_ps(C + ii * N + jj);
                        c = _mm256_fmadd_ps(a, b, c);
                        _mm256_storeu_ps(C + ii * N + jj, c);
                    }
                }
            }
        }
    }
}

int main() {
    // malloc A, B, C
    auto A = (float *) malloc(N * N * sizeof(float));
    auto B = (float *) malloc(N * N * sizeof(float));
    auto C = (float *) malloc(N * N * sizeof(float));
    // random initialize A, B
    std::uniform_real_distribution<float> distribution(-10, 10);
    std::default_random_engine generator;
    for (int i = 0; i < N * N; i++) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
        C[i] = 0;
    }
    // measure time
    auto start = std::chrono::high_resolution_clock::now();
    gemm_block(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "elapsed time: " << elapsed.count() * 1000 << "ms\n";
    // use gemm_baseline verify gemm_avx
    gemm_verify(A, B, C);
    return 0;
}
