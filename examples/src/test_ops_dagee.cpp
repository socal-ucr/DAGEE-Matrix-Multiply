#include <iostream>

#include <GPU_Utils.hpp>
#include <verify.hpp>

#include <dagee/ATMIdagExecutor.h>
#include <dagee/ATMIalloc.h>

float valA = 3;
float valB = 2;

template <typename T>
void dagee_test(size_t n, bool sequentially = true)
{
    // unsigned numThreads = 1;
    dim3 threadsPerBlock = (1024);
    dim3 blocks = (((n * n) + threadsPerBlock.x - 1) / threadsPerBlock.x);

    dim3 mulThreadsPerBlock;
    dim3 mulBlocks;

    mulThreadsPerBlock.x = TILE_SIZE;
    mulThreadsPerBlock.y = TILE_SIZE;

    mulBlocks.x = (n + mulThreadsPerBlock.x - 1) / mulThreadsPerBlock.x;
    mulBlocks.y = (n + mulThreadsPerBlock.y - 1) / mulThreadsPerBlock.y;

    std::vector<T>
        A(n * n, valA);
    std::vector<T> B(n * n, valB);
    std::vector<T> Res(n * n);

    // Initialize executors

    // using CpuExec = dagee::CpuExecutorAtmi;
    using GpuExec = dagee::GpuExecutorAtmi;
    using DagExec = dagee::ATMIdagExecutor<GpuExec>;

    dagee::AllocManagerAtmi bufMgr;

    auto *arrA = bufMgr.makeSharedCopy(A);
    auto *arrB = bufMgr.makeSharedCopy(B);
    auto *resultAdd = bufMgr.makeSharedCopy(Res);
    auto *resultSub = bufMgr.makeSharedCopy(Res);
    auto *resultMul = bufMgr.makeSharedCopy(Res);

    // Register kernel

    // CpuExec cpuEx;
    GpuExec gpuEx;

    auto addFunc = gpuEx.registerKernel<T *, T *, T *, dim3>(&hip_add_kernel<T>);
    auto subFunc = gpuEx.registerKernel<T *, T *, T *, dim3>(&hip_subtract_kernel<T>);
    auto mulFunc = gpuEx.registerKernel<T *, T *, T *, dim3>(&hip_multiply_kernel<T>);

    // Define and run tasks

    if (sequentially)
    {
        // rocblas_initialize();
        auto addTask = gpuEx.launchTask(gpuEx.makeTask(blocks, threadsPerBlock, addFunc, arrA, arrB, resultAdd, n));

        auto subTask = gpuEx.launchTask(gpuEx.makeTask(blocks, threadsPerBlock, subFunc, arrA, arrB, resultSub, n), {addTask});

        auto mulTask = gpuEx.launchTask(gpuEx.makeTask(mulBlocks, mulThreadsPerBlock, mulFunc, arrA, arrB, resultMul, n), {subTask});

        gpuEx.waitOnTask(mulTask);
    }

    // Verify results

    verify_matrix_addition(arrA, arrB, resultAdd, n);
    verify_matrix_subtraction(arrA, arrB, resultSub, n);
    verify_matrix_multiply(arrA, arrB, resultMul, n);
}

int main(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 8;
    dagee_test<float>(n);

    std::cout << "Success" << std::endl;

    return 0;
}
