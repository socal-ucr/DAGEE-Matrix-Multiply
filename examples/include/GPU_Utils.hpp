#ifndef GPU_UTILS_HPP
#define GPU_UTILS_HPP

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocblas.h>

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

constexpr float alpha = 1;
constexpr float beta = 1;
constexpr float negativeBeta = -1 * alpha;

constexpr size_t TILE_SIZE = 32;

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
__global__ void hip_add_kernel(const T *A, const T *B, T *C, size_t n)
{
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (i < n * n)
    {
        C[i] = A[i] + B[i];
    }
}

template <typename T>
__global__ void hip_subtract_kernel(const T *A, const T *B, T *C, size_t n)
{
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (i < n * n)
    {
        C[i] = A[i] - B[i];
    }
}

template <typename T>
__global__ void hip_multiply_kernel(const T *A, const T *B, T *C, size_t n)
{
    __shared__ T ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ T ds_B[TILE_SIZE][TILE_SIZE];

    auto row = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_y;
    auto col = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_x;

    T Cval = 0;

    for (auto i = 0; i < hipGridDim_x; i++)
    {
        if (row < n && (hipThreadIdx_x + (i * TILE_SIZE)) < n)
            ds_A[hipThreadIdx_y][hipThreadIdx_x] = A[(row * n) + hipThreadIdx_x + (i * TILE_SIZE)];
        else
            ds_A[hipThreadIdx_y][hipThreadIdx_x] = 0;

        if (col < n && (hipThreadIdx_y + i * TILE_SIZE) < n)
            ds_B[hipThreadIdx_y][hipThreadIdx_x] = B[(hipThreadIdx_y + i * TILE_SIZE) * n + col];
        else
            ds_B[hipThreadIdx_y][hipThreadIdx_x] = 0;

        __syncthreads();

        for (auto j = 0; j < TILE_SIZE; j++)
            Cval += ds_A[hipThreadIdx_y][j] * ds_B[j][hipThreadIdx_x];
        __syncthreads();
    }

    if (row < n && col < n)
    {
        C[row * n + col] = Cval;
    }
}

template <typename T>
void hip_add(const T *A, const T *B, T *C, size_t n)
{
    dim3 threadsPerBlock = (1024);
    dim3 blocks = (((n * n) + threadsPerBlock.x - 1) / threadsPerBlock.x);
    hipLaunchKernelGGL(hip_add_kernel,
                       dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       A, B, C, n * n);
    hipDeviceSynchronize();
}

template <typename T>
void hip_subtract(const T *A, const T *B, T *C, size_t n)
{
    dim3 threadsPerBlock = (1024);
    dim3 blocks = (((n * n) + threadsPerBlock.x - 1) / threadsPerBlock.x);
    hipLaunchKernelGGL(hip_subtract_kernel,
                       dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       A, B, C, n * n);
    hipDeviceSynchronize();
}

template <typename T>
void hip_multiply(const T *A, const T *B, T *C, size_t n)
{
    dim3 threadsPerBlock;
    dim3 blocks;

    threadsPerBlock.x = TILE_SIZE;
    threadsPerBlock.y = TILE_SIZE;

    blocks.x = (n + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocks.y = (n + threadsPerBlock.y - 1) / threadsPerBlock.y;

    hipLaunchKernelGGL(hip_multiply_kernel,
                       dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       A, B, C, n);
    hipDeviceSynchronize();
}

template <typename T>
void hip_memcpy(T *dstPtr, T *srcPtr, size_t n, hipMemcpyKind kind = hipMemcpyDefault)
{
    CHECK_HIP_ERROR(hipMemcpy(dstPtr, srcPtr, sizeof(T) * n, kind));
}

// ----------------------------------------------------------- rocBLAS OPERATIONS
void rocblas_initialize()
{
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
}

template <typename T>
void rocblas_add(const T *A, const T *B, T *C, rocblas_int m)
{
    CHECK_ROCBLAS_ERROR(rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, m, &alpha, A, m, &beta, B, m, C, m));
    // Function does not return until operation is completed.
    hipDeviceSynchronize();
}

template <typename T>
void rocblas_subtract(const T *A, const T *B, T *C, rocblas_int m)
{
    CHECK_ROCBLAS_ERROR(rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, m, &alpha, A, m, &negativeBeta, B, m, C, m));
    // Function does not return until operation is completed.
    hipDeviceSynchronize();
}

template <typename T>
void rocblas_multiply(const T *A, const T *B, T *C, rocblas_int m)
{
    CHECK_ROCBLAS_ERROR(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, m, m, &alpha, A, m, B, m, &beta, C, m));
    // Function does not return until operation is completed.
    hipDeviceSynchronize();
}

#endif // GPU_UTILS_HPP
