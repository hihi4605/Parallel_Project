 
#include "Matrix.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <random>
#include <vector>
#include <assert.h>
#include "Timer.hpp"
#include <cstdlib>
#include <string>

unsigned
getInput();

void
cudaFillMatrix(int* m, unsigned N);

void
assert_correctness(int* a, int* b, int* c, int N);

void
printIntro();

void
KIJMatMul(std::vector< std::vector<int>> A, std::vector< std::vector<int>> B, unsigned N);

void
fillCPU(std::vector< std::vector<int>>& A, unsigned N);


__global__ void matrixMultiply(int* a, int* b, int* c, int N)
{
    int d_col = blockIdx.x * blockDim.x + threadIdx.x;
    int d_row = blockIdx.y * blockDim.y + threadIdx.y;

    // This will ensure we dont access threads outside our matrix dimension
    if (d_row < N && d_col < N)
    {
        unsigned prod = 0;
        for (int i = 0; i < N; i++)
        {
            prod += a[d_row * N + i] * b[i * N + d_col];
        }
        c[d_row * N + d_col] = prod;
    }
}

__global__ void warmupCall()
{

}

int main()
{
    // Initialize the CUDA Runtime with a kernal call 
    // else our initial CUDA calls will be very slow
    cudaSetDevice(0);
    printIntro();
    warmupCall <<< 1, 1 >> > ();

    auto N = getInput();

    // Get the number of bytes for our matrix
    size_t num_bytes = N * N * sizeof(int);
    // Allocate memory on the device

    int* a, * b, * c;
    cudaMallocManaged(&a, num_bytes);
    cudaMallocManaged(&b, num_bytes);
    cudaMallocManaged(&c, num_bytes);

    cudaFillMatrix(a, N);
    cudaFillMatrix(b, N);



    // Allocate our threadblocks that will be used to launch our kernel
    int threads = 32;
    int blocks = (ceil)(N + threads - 1 / threads);
    // Launch Kernel with a 2D grid of (block, block)
    // with each thread block containing a 2D grid of 1024 threads
    // Dim3 stores memory layout for a kernel launch
    dim3 threads_per_block(threads, threads);
    dim3 blocks_per_grid(blocks, blocks);

    // Time GPU using cudaEvent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMultiply << < blocks_per_grid, threads_per_block >> > (a, b, c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\nGPU Done!\n");
    printf("GPU Elapsed Time: %f ms\n", milliseconds);

    // Tell the CPU to wait until the GPU completes its task
    cudaDeviceSynchronize();

    printf("\nChecking GPU Correctness...\n");

    assert_correctness(a, b, c, N);

    printf("\nCorrect Assertion Passed!\n");

    std::vector<std::vector<int>> A(N, std::vector<int>(N));
    std::vector<std::vector<int>> B(N, std::vector<int>(N));

    // Timing CPU
    fillCPU(A, N);
    fillCPU(B, N);

   
    printf("\nStarting CPU IJK...\n");

    Timer t;
    KIJMatMul(A, B, N);
    t.stop();

    printf("CPU Done!\n");
    float cpu_time = t.getElapsedMs();
    printf("CPU Elapsed Time: %f ms\n", cpu_time);

    float speedup = cpu_time / milliseconds;
    printf("\nSpeedup: %f\n", speedup);

    return 0;
}



/**
 * @brief Fills our matrix with random integers.
 *
 * @param m A pointer to the begining of the matrix you
 * are filling.
 * @param N The order of the matrix.
 */
void cudaFillMatrix(int* m, unsigned N) {
    // Use a consistent seed for reproducibility
    static std::minstd_rand engine(0);

    static  std::uniform_int_distribution<int> dis(0, 4);

    for (size_t i = 0; i < N * N; ++i) {
        m[i] = dis(engine);
    }
}

void
KIJMatMul(std::vector< std::vector<int>> A, std::vector< std::vector<int>> B, unsigned N)
{
    std::vector< std::vector<int>> result(N, std::vector<int>(N, 0));
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}



/**
 * @brief Compares the result of our matrix row,col computation
 * against the cpu to test if the correct computation is done.
 * If not causes an assert failure.
 *
 * @param a Matrix A
 * @param b Matrix B
 * @param c Output matrix C
 * @param N The order of the matricies
 */
void assert_correctness(int* a, int* b, int* c, int N) {
    for (int i = 0; i < N; i++) {

        for (int j = 0; j < N; j++) {

            int tmp = 0;
            for (int k = 0; k < N; k++) {

                tmp += a[i * N + k] * b[k * N + j];
            }
            // Check against the CPU result
            assert(tmp == c[i * N + j]);

        }
    }



}

/**
 * @brief Generates consistent random numbers for our CPU matrix.
 *
 * @param A The matrix we want to fill
 * @param N The order of the matrix
 */
void
fillCPU(std::vector< std::vector<int>>& A, unsigned N)
{
    // Use a consistent seed for reproducibility
    static std::minstd_rand engine(0);

    static  std::uniform_int_distribution<int> dis(0, 4);

    auto lambda = [&]() {
        static std::uniform_int_distribution<int> dis(0, 4);
        return dis(engine);
        };

    std::generate_n(A.begin(), N, [&]() {
        std::generate_n(A.back().begin(), N, lambda);
        return A.back();
        });

}

/**
 * @brief Prints an intro as well as the device properties of the system.
 *
 */
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


    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Number of Multi-Processors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "============================================" << std::endl;
}


/**
 * @brief Get user input or quit the program.
 *
 * @return unsigned An unsigned order for the matrix.
 */
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
            std::exit(0);
        }


        else
        {
            N = std::stoul(input);
            if (N > 0)
            {
                break;
            }
            else
                std::cout << "Invalid input. Please enter a positive integer for N." << std::endl;
        }
    }

    return N;
}