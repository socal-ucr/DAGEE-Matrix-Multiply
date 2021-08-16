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
    size_t threadsPerBlock = 1024;
    size_t blocks = ((n * n) + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<T> A(n * n, valA);
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

    auto addFunc = gpuEx.registerKernel<T *, T *, T *, size_t>(&hip_add<T>);
    auto subFunc = gpuEx.registerKernel<T *, T *, T *, size_t>(&hip_subtract<T>);
    auto mulFunc = gpuEx.registerKernel<T *, T *, T *, size_t>(&hip_multiply<T>);

    // Define and run tasks

    if (sequentially)
    {
        // rocblas_initialize();
        auto addTask = gpuEx.launchTask(gpuEx.makeTask(blocks, threadsPerBlock, addFunc, arrA, arrB, resultAdd, n));

        auto subTask = gpuEx.launchTask(gpuEx.makeTask(blocks, threadsPerBlock, subFunc, arrA, arrB, resultSub, n), {addTask});

        auto mulTask = gpuEx.launchTask(gpuEx.makeTask(blocks, threadsPerBlock, mulFunc, arrA, arrB, resultMul, n), {subTask});

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
