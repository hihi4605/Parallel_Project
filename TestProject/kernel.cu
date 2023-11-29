#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <random>
#include <vector>
#include <assert.h>
#include "Timer.hpp"
#include <cstdlib>
#include <string>

unsigned
getInput();

void
fillMatrix(int* m, unsigned N);

unsigned
calculateChecksum(int* matrix, unsigned N);

void verify_result(int* a, int* b, int* c, int N);

void
printIntro();

__global__ void matrixMultiply(int* a, int* b, int* c, int N)
{
    int d_col = blockIdx.x * blockDim.x + threadIdx.x;
    int d_row = blockIdx.y * blockDim.y + threadIdx.y;
 

    if (d_row < N && d_col < N)
    {
        int tmp = 0;
        for (int i = 0; i < N; i++)
        {
            tmp += a[d_row * N + i] * b[i * N + d_col];
        }
        c[d_row * N + d_col] = tmp;
    }
}

int main()
{
    // Initialize the CUDA device
    cudaFree(0);
    printIntro();
  
 
        auto N = getInput();

        // Get the number of bytes for our matrix
        size_t num_bytes = N * N * sizeof(int);
        // Allocate memory on the device
        // Cuda malloc managed will ensure that memory on the device is freed at program completion
        int* a, * b, * c;
        cudaMallocManaged(&a, num_bytes);
        cudaMallocManaged(&b, num_bytes);
        cudaMallocManaged(&c, num_bytes);

        fillMatrix(a, N);
        fillMatrix(b, N);

        // Allocate our threadblocks that will be used to launch our kernel
        int threads = 16;
        int blocks = (N + threads - 1) / threads;

        // A structure to contain our grid, block configuration for our kernel
        dim3 threads_per_block(threads, threads, 1);
        dim3 blocks_per_grid(blocks, blocks);

        // Time GPU
        Timer<> t;
        // Launch Kernel with a 2D grid of (block, block)
        // with each thread block containing a 2D grid of (thread, thread)
        matrixMultiply << < blocks_per_grid, threads_per_block >> > (a, b, c, N);
        t.stop();
        double matrix_time = t.getElapsedMs();
        printf("\nGPU Done!\n");
        printf("GPU Elapsed Time: %f ms\n", matrix_time);

        // Tell the CPU to wait until the GPU completes its task
        cudaDeviceSynchronize();

        printf("\nStarting CPU\n");
        t.start();
        verify_result(a, b, c, N);
        t.stop();
        double cpu_time = t.getElapsedMs();
        printf("CPU Elapsed Time: %f ms\n", cpu_time);

        // Verify CPU result with checksum
        unsigned checksumCPU = calculateChecksum(c, N);
        printf("CPU Checksum: %u\n", checksumCPU);

        double speedup = cpu_time / matrix_time;
        printf("\nSpeedup: %f\n", speedup);
 
    return 0;
}

unsigned getInput()
{
    unsigned N;
    std::string input;

    while (true)
    {
        std::cout << "Enter the Order of the square matrix (N x N) or 'q' to quit: ";
        std::cin >> input;

        if (input == "q") {
            std::cout << "Exiting the program.\n";
            std::exit(0); // Exit the program
        }

        // Try converting input to an unsigned integer
        try
        {
            N = std::stoul(input);
            if (N > 0)
                break; // Input is valid, exit the loop
            else
                std::cout << "Invalid input. Please enter a positive integer for N." << std::endl;
        }
        catch (const std::invalid_argument& e)
        {
            std::cout << "Invalid input. Please enter a positive integer for N or 'q' to quit." << std::endl;
            std::cin.clear(); // Clear the error flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
        }
        catch (const std::out_of_range& e)
        {
            std::cout << "Invalid input. Number out of range. Please enter a smaller value." << std::endl;
            std::cin.clear(); // Clear the error flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
        }
    }

    return N;
}


void fillMatrix(int* m, unsigned N) {
    // Use a consistent seed for reproducibility
    static std::minstd_rand engine(0);

    // Adjust the range as needed
    std::uniform_int_distribution<int> dis(0, 4);

    for (size_t i = 0; i < N * N; ++i) {
        m[i] = dis(engine);
    }
}

void printIntro() {
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "============================================" << std::endl;
    std::cout << "       Matrix Multiplication Program        " << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "This program performs matrix multiplication" << std::endl;
    std::cout << "on both the CPU and GPU using CUDA. It then" << std::endl;
    std::cout << "compares the results and calculates the" << std::endl;
    std::cout << "speedup achieved by the GPU computation." << std::endl;
    std::cout << "============================================" << std::endl;

    // Print device properties
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Number of Multi-Processors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "============================================" << std::endl;
}


void verify_result(int* a, int* b, int* c, int N) {
    unsigned global_sum = 0;
    // For every row...
    for (int i = 0; i < N; i++) {
        // For every column...
        for (int j = 0; j < N; j++) {
            // For every element in the row-column pair
            int tmp = 0;
            for (int k = 0; k < N; k++) {
                // Accumulate the partial results
                tmp += a[i * N + k] * b[k * N + j];
                global_sum += tmp;
            }

            // Check against the CPU result
            assert(tmp == c[i * N + j]);
        }
    }
    printf("CPU Done!\n");
    printf("Matrix Sum: %u\n", global_sum);

}

unsigned calculateChecksum(int* matrix, unsigned N) {
    unsigned checksum = 0;
    for (size_t i = 0; i < N * N; ++i) {
        checksum += matrix[i];
    }
    return checksum;
}
