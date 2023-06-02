#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>
#include <iostream>

#define N (1 << 13)

__global__ void gemm_baseline(const float* A,const float* B, float* C) {
    int const tid = blockIdx.x * gridDim.x + threadIdx.x;
    int const row = tid / N;
    int const column = tid % N;
    float c = 0;
    for (int i = 0; i < N; i++) {
        c += A[row * N + i] * B[i * N + column];
    }
    C[tid] = c;
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
    /*for (int i = 0; i < N * N; i++) {
        std::cout << A[i] << " \n"[i % N == N - 1];
    }
    std::cout << "\n";
    for (int i = 0; i < N * N; i++) {
        std::cout << B[i] << " \n"[i % N == N - 1];
    }
    std::cout << "\n";*/
    // cumalloc A, B, C
    float* cuda_a, * cuda_b, * cuda_c;
    clock_t* time;
    cudaMalloc((void**)&cuda_a, sizeof(float) * N * N);
    cudaMalloc((void**)&cuda_b, sizeof(float) * N * N);
    cudaMalloc((void**)&cuda_c, sizeof(float) * N * N);
    cudaMemcpy(cuda_a, A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    // define gridsize and blocksize
    dim3 grid(N, 1, 1);
    dim3 block(N, 1, 1);
    // compute
    //clock_t start = clock();
    gemm_baseline << <8,1024>> > (cuda_a, cuda_b, cuda_c);
    //clock_t end = clock();
    cudaMemcpy(C, cuda_c, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
    /*for (int i = 0; i < N * N; i++) {
        std::cout << C[i] << " \n"[i % N == N - 1];
    }*/
    //
    // free mem
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
    //std::cout << start - end;
    //gemm_verify(A, B, C);
    free(A);
    free(B);
    free(C);
}
