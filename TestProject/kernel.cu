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
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

boost::numeric::ublas::matrix<int>
multiplyMatrices(const boost::numeric::ublas::matrix<int>& A, const boost::numeric::ublas::matrix<int>& B);

void 
initializeMatrixRandomly(boost::numeric::ublas::matrix<int>& matrix);

unsigned
getInput();

void
fillMatrix(int* m, unsigned N);

void verify_result(int* a, int* b, int* c, int N);

__global__ void matrixMultiply(int* a, int* b, int* c, int N)
{
    unsigned sum = 0;

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
        sum += tmp;
    }

}

int main()
{
    while (1)
    {
        auto N = getInput();
        // Initialize Matricies
  
        boost::numeric::ublas::matrix<int> A(N, N);
        boost::numeric::ublas::matrix<int> B(N, N);
        boost::numeric::ublas::matrix<int> C;

        initializeMatrixRandomly(A);
        initializeMatrixRandomly(B);
        Timer<> r;
        C = multiplyMatrices(A, B);
        r.stop();
        double boost = r.getElapsedMs();
        std::cout << "Boost Library: " << boost << "ms" << std::endl;
        
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
        unsigned out = 0;
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
        printf("\nDevice Matrix Sum: %d\n", out);
        double cpu_time = t.getElapsedMs();
        printf("CPU Elapsed Time: %f ms\n", cpu_time);
        double speedup = cpu_time / matrix_time;
        printf("Speedup: %f\n", speedup);
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

void initializeMatrixRandomly(boost::numeric::ublas::matrix<int>& matrix) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1.0, 10.0); // Adjust the range as needed

    for (size_t i = 0; i < matrix.size1(); ++i) {
        for (size_t j = 0; j < matrix.size2(); ++j) {
            matrix(i, j) = dis(gen);
        }
    }
}

// Function to perform matrix multiplication
boost::numeric::ublas::matrix<int>
multiplyMatrices(const boost::numeric::ublas::matrix<int>& A, const boost::numeric::ublas::matrix<int>& B) {
    // Check if matrices A and B can be multiplied
    if (A.size2() != B.size1()) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }

    // Perform matrix multiplication and return the result
    return boost::numeric::ublas::prod(A, B);
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
    std::cout << "Matrix Sum: " << global_sum;

}