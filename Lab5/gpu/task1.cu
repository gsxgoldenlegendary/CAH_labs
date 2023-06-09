﻿#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>
#include <iostream>

#define N (1 << 11)
#define M (N>>0)

__global__ void gemm_baseline(const float* A,const float* B, float* C) {
    int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    int row = threadId / N;
    int col = threadId % N;

    float t = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
        t += A[row * N + i] * B[i * N + col];
    }
    C[threadId] = t;
}

void gemm_verify(const float* A, const float* B,const float* C) {
    auto C_baseline = (float*)malloc(N * N * sizeof(float));
    memset(C_baseline, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C_baseline[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    for (int i = 0; i < N * N; i++) {
        if (std::fabs(C[i] - C_baseline[i]) > 1) {
            std::cout << "gemm_avx wrong answer\n";
            exit(-1);
        }
    }
    free (C_baseline);
}

int main()
{
    // malloc A, B, C
    auto A = (float*)malloc(N * N * sizeof(float));
    auto B = (float*)malloc(N * N * sizeof(float));
    auto C = (float*)malloc(N * N * sizeof(float));
    memset(C, 0, N * N * sizeof(float));
    // random initialize A, B
    std::uniform_real_distribution<float> distribution(-10, 10);
    std::default_random_engine generator;
    for (int i = 0; i < N * N; i++) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }
    // cumalloc A, B, C
    float* cuda_a, * cuda_b, * cuda_c;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc((void**)&cuda_a, sizeof(float) * N * N);
    cudaMalloc((void**)&cuda_b, sizeof(float) * N * N);
    cudaMalloc((void**)&cuda_c, sizeof(float) * N * N);
    cudaMemcpy(cuda_a, A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    // define gridsize and blocksize
    dim3 grid(M, M, 1);
    dim3 block(1, 1, 1);
    // compute
    cudaEventRecord(start, 0);
    gemm_baseline << <grid, block >> > (cuda_a, cuda_b, cuda_c);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaMemcpy(C, cuda_c, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
    // free mem
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time cost:" << elapsedTime << "ms";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //gemm_verify(A, B, C);
    free(A);
    free(B);
    free(C);
}
