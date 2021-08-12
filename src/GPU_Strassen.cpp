#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cstdlib>

#include "GPU_Utils.hpp"
#include "CPU_Utils.hpp"

#define DEBUG 0

template <typename T>
void gpu_strassen_mul(T A, T B, T &C, int dim)
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

        print_matrix(C, dim * dim, "Strassen-Winograd Matrix");
    }
    else
    {
        int m = dim / 2;

        // rocblas_sub_matrices
        float *A_11 = hip_host_malloc<float>(m * m);
        float *A_12 = hip_host_malloc<float>(m * m);
        float *A_21 = hip_host_malloc<float>(m * m);
        float *A_22 = hip_host_malloc<float>(m * m);
        float *B_11 = hip_host_malloc<float>(m * m);
        float *B_12 = hip_host_malloc<float>(m * m);
        float *B_21 = hip_host_malloc<float>(m * m);
        float *B_22 = hip_host_malloc<float>(m * m);

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

        // S_1 = A_21 + A_22
        float *S_1 = hip_host_malloc<float>(m * m);
        rocblas_add(A_21, A_22, S_1, m);

        // S_2 = S_1 - A_11
        float *S_2 = hip_host_malloc<float>(m * m);
        rocblas_sub(S_1, A_11, S_2, m);

        // S_3 = A_11 - A_21
        float *S_3 = hip_host_malloc<float>(m * m);
        rocblas_sub(A_11, A_21, S_3, m);

        // S_4 = A_12 - S_2
        float *S_4 = hip_host_malloc<float>(m * m);
        rocblas_sub(A_12, S_2, S_4, m);

        // S_5 = B_12 - B_11
        float *S_5 = hip_host_malloc<float>(m * m);
        rocblas_sub(B_12, B_11, S_5, m);

        // S_6 = B_22 - S_5
        float *S_6 = hip_host_malloc<float>(m * m);
        rocblas_sub(B_22, S_5, S_6, m);

        // S_7 = B_22 - B_12
        float *S_7 = hip_host_malloc<float>(m * m);
        rocblas_sub(B_22, B_12, S_7, m);

        // S_8 = S_6 - B_21
        float *S_8 = hip_host_malloc<float>(m * m);
        rocblas_sub(S_6, B_21, S_8, m);

        // ----------------------------------------------

        // M_1 = S_2 x S_6
        float *M_1 = hip_host_malloc<float>(m * m);
        rocblas_multiply(S_6, S_2, M_1, m);

        // M_2 = A_11 x B_11
        float *M_2 = hip_host_malloc<float>(m * m);
        rocblas_multiply(B_11, A_11, M_2, m);

        // M_3 = A_12 x B_21
        float *M_3 = hip_host_malloc<float>(m * m);
        rocblas_multiply(B_21, A_12, M_3, m);

        // M_4 = S_3 x S_7
        float *M_4 = hip_host_malloc<float>(m * m);
        rocblas_multiply(S_7, S_3, M_4, m);

        // // M_5 = S_1 x S_5
        float *M_5 = hip_host_malloc<float>(m * m);
        rocblas_multiply(S_5, S_1, M_5, m);

        // // M_6 = S_4 x B_22
        float *M_6 = hip_host_malloc<float>(m * m);
        rocblas_multiply(B_22, S_4, M_6, m);

        // M_7 = A_22 x S_8
        float *M_7 = hip_host_malloc<float>(m * m);
        rocblas_multiply(S_8, A_22, M_7, m);

        // ----------------------------------------------

        // V_1 = M_1 + M_2
        float *V_1 = hip_host_malloc<float>(m * m);
        rocblas_add(M_1, M_2, V_1, m);

        // V_2 = V_1 + M_4
        float *V_2 = hip_host_malloc<float>(m * m);
        rocblas_add(V_1, M_4, V_2, m);

        // V_3 = M_5 + M_6
        float *V_3 = hip_host_malloc<float>(m * m);
        rocblas_add(M_5, M_6, V_3, m);

        // ----------------------------------------------

        // C_11 = M_2 + M_3
        float *C_11 = hip_host_malloc<float>(m * m);
        rocblas_add(M_2, M_3, C_11, m);

        // C_12 = V_1 + V_3
        float *C_12 = hip_host_malloc<float>(m * m);
        rocblas_add(V_1, V_3, C_12, m);

        // C_21 = V_2 - M_7
        float *C_21 = hip_host_malloc<float>(m * m);
        rocblas_sub(V_2, M_7, C_21, m);

        // C_22 = V_2 + M_5
        float *C_22 = hip_host_malloc<float>(m * m);
        rocblas_add(V_2, M_5, C_22, m);

        hipDeviceSynchronize();

        // ----------- POPULATING C-MATRIX ---------------

        float *temp = hip_host_malloc<float>(dim * dim);

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                temp[(i * dim) + j] = C_11[j + (i * m)];
            }
            for (int j = 0; j < m; ++j)
            {
                temp[(i * dim) + j + m] = C_12[j + (i * m)];
            }
        }

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                temp[(i * dim) + j + (dim * m)] = C_21[j + (i * m)];
            }
            for (int j = 0; j < m; ++j)
            {
                temp[(i * dim) + j + m + (dim * m)] = C_22[j + (i * m)];
            }
        }

        for (int i = 0; i < dim * dim; ++i)
        {
            C[i] = temp[i];
        }
    }
}

int main(int argc, char **argv)
{
    int n = (argc < 2) ? 8 : atoi(argv[1]);

    float *A = hip_host_malloc<float>(n * n);
    float *B = hip_host_malloc<float>(n * n);
    float *C = hip_host_malloc<float>(n * n);
    float *C_verify = hip_host_malloc<float>(n * n);

    for (int i = 0; i < n * n; ++i)
    {
        auto val = rand() % 10;
        A[i] = val;
        B[i] = val;
    }

    rocblas_initialize();

    gpu_strassen_mul(A, B, C, n);

#if DEBUG == 1
    print_matrix(C, n * n, "matrix_C");
#endif
    verify_matrix_multiply(A, B, C, n);
#if DEBUG == 1
    print_matrix(C_verify, n * n, "matrix_C_verify");
#endif

    std::cout << "Success" << std::endl;

    return 0;
}