#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>
#include <iostream>
#include <chrono>

#define N (1 << 11)
#define BLOCK_SIZE (1<<0)

__global__ void gemm_block(const float* A, const float* B, float* C) {
    // thread location
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    // blockDim.x == blockDim.y == BLOCK_SIZE here
    const int begin_a = block_y * blockDim.y * N;
    const int end_a = begin_a + N - 1;
    const int step_a = blockDim.x;

    const int begin_b = block_x * blockDim.x;
    const int step_b = blockDim.y * N;

    float result_temp = 0.0f;

    for (int index_a = begin_a, index_b = begin_b; index_a < end_a; index_a += step_a, index_b += step_b)
    {
        // shared memory
        __shared__ float SubMat_A[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float SubMat_B[BLOCK_SIZE][BLOCK_SIZE];

        // copy data to shared memory
        SubMat_A[thread_y][thread_x] = A[index_a + thread_y * N + thread_x];
        SubMat_B[thread_y][thread_x] = B[index_b + thread_y * N + thread_x];

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            result_temp += SubMat_A[thread_y][i] * SubMat_B[i][thread_x];
        }

        __syncthreads();
    }

    int begin_result = block_y * blockDim.y * N + begin_b;
    C[begin_result + thread_y * N + thread_x] = result_temp;
}

void gemm_verify(const float* A, const float* B, const float* C) {
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
    free(C_baseline);
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
    dim3 grid(N/BLOCK_SIZE ,N/BLOCK_SIZE, 1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE,1);
    // compute
    cudaEventRecord(start, 0);
    gemm_block << <grid, block >> > (cuda_a, cuda_b, cuda_c);
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
