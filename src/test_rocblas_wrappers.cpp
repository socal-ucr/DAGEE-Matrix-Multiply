#include <iostream>
#include "GPU_Utils.hpp"
#include "CPU_Utils.hpp"

float valA = 3;
float valB = 2;

template <typename T>
void verify_matrix_addition(const T *A, const T *B, const T *C, int m)
{
    for (auto i = 0; i < m * m; ++i)
    {
        if (C[i] != A[i] + B[i])
        {
            std::cerr << "Found error in vector Add at " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

template <typename T>
void verify_matrix_subtraction(const T *A, const T *B, const T *C, int m)
{
    for (auto i = 0; i < m * m; ++i)
    {
        if (C[i] != A[i] - B[i])
        {
            std::cerr << "Found error in vector Sub at " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

int main()
{
    int n = 8;
    float *arrA = hip_host_malloc<float>(n * n);
    float *arrB = hip_host_malloc<float>(n * n);

    for (auto i = 0; i < n * n; i++)
    {
        arrA[i] = valA;
        arrB[i] = valB;
    }

    rocblas_initialize();

    float *resultAdd = hip_host_malloc<float>(n * n);
    rocblas_add(arrA, arrB, resultAdd, n);
    hipDeviceSynchronize();
    verify_matrix_addition(arrA, arrB, resultAdd, n);

    float *resultSub = hip_host_malloc<float>(n * n);
    rocblas_sub(arrA, arrB, resultSub, n);
    hipDeviceSynchronize();
    verify_matrix_subtraction(arrA, arrB, resultSub, n);

    float *resultMul = hip_host_malloc<float>(n * n);
    rocblas_multiply(arrA, arrB, resultMul, n);
    hipDeviceSynchronize();
    verify_matrix_multiply(arrA, arrB, resultMul, n);

    hip_free<float>(arrA);
    hip_free<float>(arrB);
    hip_free<float>(resultSub);
    hip_free<float>(resultAdd);
    hip_free<float>(resultMul);
}
