#include <cstdlib>
#include <iostream>
#include <vector>
#include <rocblas.h>
#include <hip/hip_runtime.h>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(ERROR)                                                      \
    do                                                                              \
    {                                                                               \
        /* Use error__ in case ERROR contains "error" */                            \
        hipError_t error__ = (ERROR);                                               \
        if (error__ != hipSuccess)                                                  \
        {                                                                           \
            std::cerr << "error: " << hipGetErrorString(error__) << " (" << error__ \
                      << ") at " __FILE__ ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)
#endif

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(ERROR)                                      \
    do                                                                  \
    {                                                                   \
        /* Use error__ in case ERROR contains "error" */                \
        rocblas_status error__ = (ERROR);                               \
        if (error__ != rocblas_status_success)                          \
        {                                                               \
            std::cerr << "error: "                                      \
                      << " (" << error__                                \
                      << ") at " __FILE__ ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)
#endif

// TODO(kiran) : Verify if global handle could be used during concurrent execution.
rocblas_handle handle;

const float alpha = 1;
const float beta = 1;
const float negativeBeta = -1 * alpha;

// float *da = hip_host_malloc<float>(n)
template <typename T>
T *hip_host_malloc(size_t n)
{
    T *ptr;
    CHECK_HIP_ERROR(hipHostMalloc(&ptr, n * sizeof(T)));
    return ptr;
}

template <typename T>
T *hip_malloc(size_t n)
{
    T *ptr;
    CHECK_HIP_ERROR(hipMalloc(&ptr, n * sizeof(T)));
    return ptr;
}

template <typename T>
void hip_free(T *ptr)
{
    CHECK_HIP_ERROR(hipFree(ptr));
}

template <typename T>
void hip_memcpy(T *dstPtr, T *srcPtr, size_t n, hipMemcpyKind kind = hipMemcpyDefault)
{
    CHECK_HIP_ERROR(hipMemcpy(dstPtr, srcPtr, sizeof(T) * n, kind));
}

template <typename T>
void print_matrix(T *matrix, int length, std::string matrix_name = "Unknown Matrix")
{
    std::cout << std::endl
              << "Printing Matrix: " << matrix_name << std::endl;

    std::cout << "Size of Matrix: " << length << std::endl;

    for (int i = 0; i < length; i++)
    {
        if (fmod(i, sqrt(length)) == 0)
        {
            std::cout << std::endl;
        }
        std::cout << matrix[i] << "\t";
    }
    std::cout << std::endl;
}

// ----------------------------------------------------------- OPERATIONS
void rocblas_initialize()
{
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
}

template <typename T>
void rocblas_add(const T *A, const T *B, T *C, rocblas_int m)
{
    CHECK_ROCBLAS_ERROR(rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, m, &alpha, A, m, &beta, B, m, C, m));
}

template <typename T>
void rocblas_sub(const T *A, const T *B, T *C, rocblas_int m)
{
    CHECK_ROCBLAS_ERROR(rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, m, &alpha, A, m, &negativeBeta, B, m, C, m));
}

template <typename T>
void rocblas_multiply(const T *A, const T *B, T *C, rocblas_int m)
{
    CHECK_ROCBLAS_ERROR(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, m, m, &alpha, A, m, B, m, &beta, C, m));
}

template <typename T>
void verify_matrix_multiply(const T *A, const T *B, const T *C, int m)
{
    T *tempResult = hip_host_malloc<T>(m * m);
    for (auto i = 0; i < m; i++)
    {
        for (auto j = 0; j < m; j++)
        {
            for (auto k = 0; k < m; k++)
            {
                tempResult[i * m + k] += A[i * m + j] * B[j * m + k];
            }
        }
    }

    for (auto i = 0; i < m; i++)
    {
        if (std::abs(C[i] - tempResult[i]) >= 1)
        {
            std::cerr << "Found error in vector at " << i << std::endl;
            std::cerr << "Expected: " << C[i] << " Obtained: " << tempResult[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    hip_free(tempResult);
}
