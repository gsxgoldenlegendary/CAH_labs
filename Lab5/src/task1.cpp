//
// Created by Jeffrey on 26/5/2023.
//

#include <bits/stdc++.h>

int N = (1 << 10);

inline void gemm_baseline(const float *A, const float *B, float *C) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
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
    std::uniform_real_distribution<float> distribution(-100, 100);
    std::default_random_engine generator;
    for (int i = 0; i < N * N; i++) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }
    // measure time
    auto start = std::chrono::high_resolution_clock::now();
    gemm_baseline(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "elapsed time: " << elapsed.count() * 1000 << "ms\n";
    return 0;
}
