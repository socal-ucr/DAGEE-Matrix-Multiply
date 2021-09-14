#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <GPU_Utils.hpp>
#include <verify.hpp>

#include <dagee/ATMIalloc.h>
#include <dagee/ATMIdagExecutor.h>

#if POWER == 1
#include <rocm_smi/rocm_smi.h>
#endif

template <typename T, class BufMgr = dagee::AllocManagerAtmi>
void gpu_strassen_mul_dagee(const T *A, const T *B, T *C, size_t dim, BufMgr &bufMgr)
{
    if (dim == 1)
    {
        C[0] = A[0] * B[0];
    }
    else if (dim == 2)
    {
        auto S_1 = A[2] + A[3];
        auto S_2 = S_1 - A[0];
        auto S_3 = A[0] - A[2];
        auto S_4 = A[1] - S_2;
        auto S_5 = B[1] - B[0];
        auto S_6 = B[3] - S_5;
        auto S_7 = B[3] - B[1];
        auto S_8 = S_6 - B[2];

        auto M_1 = S_2 * S_6;
        auto M_2 = A[0] * B[0];
        auto M_3 = A[1] * B[2];
        auto M_4 = S_3 * S_7;
        auto M_5 = S_1 * S_5;
        auto M_6 = S_4 * B[3];
        auto M_7 = A[3] * S_8;

        auto V_1 = M_1 + M_2;
        auto V_2 = V_1 + M_4;
        auto V_3 = M_5 + M_6;

        C[0] = M_2 + M_3;
        C[1] = V_1 + V_3;
        C[2] = V_2 - M_7;
        C[3] = V_2 + M_5;
    }
    else
    {
        auto m = dim / 2;

        dim3 threadsPerBlock = (1024);
        dim3 blocks = (((m * m) + threadsPerBlock.x - 1) / threadsPerBlock.x);

        dim3 mulThreadsPerBlock;
        dim3 mulBlocks;

        mulThreadsPerBlock.x = TILE_SIZE;
        mulThreadsPerBlock.y = TILE_SIZE;

        mulBlocks.x = (m + mulThreadsPerBlock.x - 1) / mulThreadsPerBlock.x;
        mulBlocks.y = (m + mulThreadsPerBlock.y - 1) / mulThreadsPerBlock.y;

        // rocblas_sub_matrices
        std::vector<float> emptyVec(m * m);
        T *A_11 = bufMgr.makeSharedCopy(emptyVec);
        T *A_12 = bufMgr.makeSharedCopy(emptyVec);
        T *A_21 = bufMgr.makeSharedCopy(emptyVec);
        T *A_22 = bufMgr.makeSharedCopy(emptyVec);
        T *B_11 = bufMgr.makeSharedCopy(emptyVec);
        T *B_12 = bufMgr.makeSharedCopy(emptyVec);
        T *B_21 = bufMgr.makeSharedCopy(emptyVec);
        T *B_22 = bufMgr.makeSharedCopy(emptyVec);

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                if (i != 0)
                {
                    A_11[(i * m) + j] = A[(i * dim) + j];
                    A_12[(i * m) + j] = A[(i * dim) + j + m];
                    A_21[(i * m) + j] = A[(i * dim) + j + (dim * m)];
                    A_22[(i * m) + j] = A[(i * dim) + j + m + (dim * m)];

                    B_11[(i * m) + j] = B[(i * dim) + j];
                    B_12[(i * m) + j] = B[(i * dim) + j + m];
                    B_21[(i * m) + j] = B[(i * dim) + j + (dim * m)];
                    B_22[(i * m) + j] = B[(i * dim) + j + m + (dim * m)];
                }
                else
                {
                    A_11[(i * dim) + j] = A[i + j];
                    A_12[(i * dim) + j] = A[i + j + m];
                    A_21[(i * dim) + j] = A[i + j + (dim * m)];
                    A_22[(i * dim) + j] = A[i + j + m + (dim * m)];

                    B_11[(i * dim) + j] = B[i + j];
                    B_12[(i * dim) + j] = B[i + j + m];
                    B_21[(i * dim) + j] = B[i + j + (dim * m)];
                    B_22[(i * dim) + j] = B[i + j + m + (dim * m)];
                }
            }
        }
        auto *S_1 = bufMgr.makeSharedCopy(emptyVec);
        auto *S_2 = bufMgr.makeSharedCopy(emptyVec);
        auto *S_3 = bufMgr.makeSharedCopy(emptyVec);
        auto *S_4 = bufMgr.makeSharedCopy(emptyVec);
        auto *S_5 = bufMgr.makeSharedCopy(emptyVec);
        auto *S_6 = bufMgr.makeSharedCopy(emptyVec);
        auto *S_7 = bufMgr.makeSharedCopy(emptyVec);
        auto *S_8 = bufMgr.makeSharedCopy(emptyVec);

        auto *M_1 = bufMgr.makeSharedCopy(emptyVec);
        auto *M_2 = bufMgr.makeSharedCopy(emptyVec);
        auto *M_3 = bufMgr.makeSharedCopy(emptyVec);
        auto *M_4 = bufMgr.makeSharedCopy(emptyVec);
        auto *M_5 = bufMgr.makeSharedCopy(emptyVec);
        auto *M_6 = bufMgr.makeSharedCopy(emptyVec);
        auto *M_7 = bufMgr.makeSharedCopy(emptyVec);

        auto *V_1 = bufMgr.makeSharedCopy(emptyVec);
        auto *V_2 = bufMgr.makeSharedCopy(emptyVec);
        auto *V_3 = bufMgr.makeSharedCopy(emptyVec);

        auto *C_11 = bufMgr.makeSharedCopy(emptyVec);
        auto *C_12 = bufMgr.makeSharedCopy(emptyVec);
        auto *C_21 = bufMgr.makeSharedCopy(emptyVec);
        auto *C_22 = bufMgr.makeSharedCopy(emptyVec);

        // Initilialize Executors

        using GpuExec = dagee::GpuExecutorAtmi;
        using DagExec = dagee::ATMIdagExecutor<GpuExec>;

        GpuExec gpuEx;
        DagExec dagEx(gpuEx);

        auto *dag = dagEx.makeDAG();

        // Register kernel functions

        auto addFunc = gpuEx.registerKernel<T *, T *, T *, dim3>(&hip_add_kernel<T>);
        auto subFunc = gpuEx.registerKernel<T *, T *, T *, dim3>(&hip_subtract_kernel<T>);
        auto mulFunc = gpuEx.registerKernel<T *, T *, T *, dim3>(&hip_multiply_kernel<T>);

        // Create Tasks

        auto S_1Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, addFunc, A_21, A_22, S_1, m));
        auto S_2Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, subFunc, S_1, A_11, S_2, m));
        auto S_3Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, subFunc, A_11, A_21, S_3, m));
        auto S_4Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, subFunc, A_12, S_2, S_4, m));
        auto S_5Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, subFunc, B_12, B_11, S_5, m));
        auto S_6Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, subFunc, B_22, S_5, S_6, m));
        auto S_7Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, subFunc, B_22, B_12, S_7, m));
        auto S_8Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, subFunc, S_6, B_21, S_8, m));

        auto M_1Task = dag->addNode(gpuEx.makeTask(mulBlocks, mulThreadsPerBlock, mulFunc, S_6, S_2, M_1, m));
        auto M_2Task = dag->addNode(gpuEx.makeTask(mulBlocks, mulThreadsPerBlock, mulFunc, B_11, A_11, M_2, m));
        auto M_3Task = dag->addNode(gpuEx.makeTask(mulBlocks, mulThreadsPerBlock, mulFunc, B_21, A_12, M_3, m));
        auto M_4Task = dag->addNode(gpuEx.makeTask(mulBlocks, mulThreadsPerBlock, mulFunc, S_7, S_3, M_4, m));
        auto M_5Task = dag->addNode(gpuEx.makeTask(mulBlocks, mulThreadsPerBlock, mulFunc, S_5, S_1, M_5, m));
        auto M_6Task = dag->addNode(gpuEx.makeTask(mulBlocks, mulThreadsPerBlock, mulFunc, B_22, S_4, M_6, m));
        auto M_7Task = dag->addNode(gpuEx.makeTask(mulBlocks, mulThreadsPerBlock, mulFunc, S_8, A_22, M_7, m));

        auto V_1Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, addFunc, M_1, M_2, V_1, m));
        auto V_2Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, addFunc, V_1, M_4, V_2, m));
        auto V_3Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, addFunc, M_5, M_6, V_3, m));

        auto C_11Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, addFunc, M_2, M_3, C_11, m));
        auto C_12Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, addFunc, V_1, V_3, C_12, m));
        auto C_21Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, subFunc, V_2, M_7, C_21, m));
        auto C_22Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, addFunc, V_2, M_5, C_22, m));

        // Build dependencies

        dag->addEdge(S_1Task, S_2Task);
        dag->addEdge(S_2Task, S_4Task);
        dag->addEdge(S_5Task, S_6Task);
        dag->addEdge(S_6Task, S_8Task);

        dag->addFanInEdges({S_6Task, S_2Task}, M_1Task);
        dag->addFanInEdges({S_7Task, S_3Task}, M_4Task);
        dag->addFanInEdges({S_5Task, S_1Task}, M_5Task);
        dag->addEdge(S_4Task, M_6Task);
        dag->addEdge(S_8Task, M_7Task);

        dag->addFanInEdges({M_1Task, M_2Task}, V_1Task);
        dag->addFanInEdges({V_1Task, M_4Task}, V_2Task);
        dag->addFanInEdges({M_5Task, M_6Task}, V_3Task);

        dag->addFanInEdges({M_2Task, M_3Task}, C_11Task);
        dag->addFanInEdges({V_1Task, V_3Task}, C_12Task);
        dag->addFanInEdges({V_2Task, M_7Task}, C_21Task);
        dag->addFanInEdges({V_2Task, M_5Task}, C_22Task);

        // Execute DAG

        dagEx.execute(dag);

    #if POWER == 1
        uint64_t avgPower = 0;
        rsmi_dev_power_ave_get(0,0, &avgPower);
        std::cout << "Avg Power (mW):" << avgPower << std::endl;
    #endif


        // Move data to *C

        auto *temp = hip_host_malloc<T>(dim * dim);

        for (auto i = 0; i < m; ++i)
        {
            for (auto j = 0; j < m; ++j)
            {
                temp[(i * dim) + j] = C_11[j + (i * m)];
            }
            for (auto j = 0; j < m; ++j)
            {
                temp[(i * dim) + j + m] = C_12[j + (i * m)];
            }
        }

        for (auto i = 0; i < m; ++i)
        {
            for (auto j = 0; j < m; ++j)
            {
                temp[(i * dim) + j + (dim * m)] = C_21[j + (i * m)];
            }
            for (auto j = 0; j < m; ++j)
            {
                temp[(i * dim) + j + m + (dim * m)] = C_22[j + (i * m)];
            }
        }

        for (auto i = 0; i < dim * dim; ++i)
        {
            C[i] = temp[i];
        }
    }
}

int main(int argc, char **argv)
{
    int n = (argc < 2) ? 8 : atoi(argv[1]);

    std::vector<float> vecA(n * n);
    std::vector<float> vecB(n * n);
    std::vector<float> vecC(n * n);

    for (int i = 0; i < n * n; ++i)
    {
        auto val = rand() % 10;
        vecA[i] = val;
        vecB[i] = val;
    }

    dagee::AllocManagerAtmi bufMgr;
    auto *A = bufMgr.makeSharedCopy(vecA);
    auto *B = bufMgr.makeSharedCopy(vecB);
    auto *C = bufMgr.makeSharedCopy(vecC);
#if POWER == 1
    rsmi_init(0);
#endif

#if TIME == 1
    auto start = std::chrono::high_resolution_clock::now();
#endif
    gpu_strassen_mul_dagee(A, B, C, n, bufMgr);
#if TIME == 1
    auto stop = std::chrono::high_resolution_clock::now();
#endif

#if DEBUG == 1
    print_matrix(C, n * n, "matrix_C");
    verify_matrix_multiply(A, B, C, n);
#endif

    std::cout << "Done" << std::endl;

#if TIME == 1
    auto duration_s = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Runtime(microseconds)" << duration_s.count() << std::endl;
#endif

    return 0;
}
