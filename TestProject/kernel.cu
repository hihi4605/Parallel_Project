
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <random>
#include <vector>
#include <assert.h>
#include "Timer.hpp"
#include <cstdlib>
__global__ void matrixMultiply(int *a, int *b,int *c , int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int tmp = 0;
        for (int i = 0; i < N; i++)
        {
            tmp += a[row * N + i] * b[i * N + col];

        }
        c[row * N + col] = tmp;
    }
}



unsigned
getInput();

void
fillMatrix(int* m, unsigned N);

void verify_result(int* a, int* b, int* c, int N);

int main()
{
    while (1)
    {
        auto N = getInput();

        size_t num_bytes = N * N * sizeof(int);

        int* a, int* b, int* c;
        cudaMallocManaged(&a, num_bytes);
        cudaMallocManaged(&b, num_bytes);
        cudaMallocManaged(&c, num_bytes);

        fillMatrix(a, N);
        fillMatrix(b, N);
        // Threablock 
        int threads = 16;
        int blocks = (N + threads - 1) / threads;

        dim3 THREADS(threads, threads);
        dim3 BLOCKS(blocks, blocks);

        // Launch Kernal 
        Timer<> t;
        matrixMultiply <<< BLOCKS, THREADS >>> (a, b, c, N);
        t.stop();
        printf("GPU Done!\n");
        double matrix_time = t.getElapsedMs();
        printf("GPU Elapsed Time: %f ms\n", matrix_time);
        cudaDeviceSynchronize();





        printf("Starting CPU\n");
        t.start();
        verify_result(a, b, c, N);
        t.stop();
  
        double cpu_time = t.getElapsedMs();
        printf("CPU Elapsed Time: %f ms\n", cpu_time);
    }
}


unsigned
getInput()
{
    unsigned N;
    std::cout << "Matrix Deminsions N >> ";
    std::cin >> N;
    return N;
}

void
fillMatrix(int* m, unsigned N)
{
 

    for (size_t i = 0; i < N * N; ++i)
    {
        m[i] = rand() % 100;
    }
}

void verify_result(int* a, int* b, int* c, int N) {
    // For every row...
    for (int i = 0; i < N; i++) {
        // For every column...
        for (int j = 0; j < N; j++) {
            // For every element in the row-column pair
            int tmp = 0;
            for (int k = 0; k < N; k++) {
                // Accumulate the partial results
                tmp += a[i * N + k] * b[k * N + j];
            }

            // Check against the CPU result
            assert(tmp == c[i * N + j]);
        }
    }
    printf("CPU Done!\n");
    std::cout << "Matrix Correct!\n";
}