#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <GPU_Utils.hpp>
#include <verify.hpp>

template <typename T>
void gpu_strassen_mul(const T *A, const T *B, T *C, size_t dim,
                      std::function<void(const T *A, const T *B, T *C, size_t n)> addFunc = hip_add<T>,
                      std::function<void(const T *A, const T *B, T *C, size_t n)> subtractFunc = hip_subtract<T>,
                      std::function<void(const T *A, const T *B, T *C, size_t n)> multiplyFunc = hip_multiply<T>)
{
    rocblas_initialize();

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
        auto m = dim / 2;

        // rocblas_sub_matrices
        auto *A_11 = hip_host_malloc<T>(m * m);
        auto *A_12 = hip_host_malloc<T>(m * m);
        auto *A_21 = hip_host_malloc<T>(m * m);
        auto *A_22 = hip_host_malloc<T>(m * m);
        auto *B_11 = hip_host_malloc<T>(m * m);
        auto *B_12 = hip_host_malloc<T>(m * m);
        auto *B_21 = hip_host_malloc<T>(m * m);
        auto *B_22 = hip_host_malloc<T>(m * m);

        for (auto i = 0; i < m; ++i)
        {
            for (auto j = 0; j < m; ++j)
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
        auto *S_1 = hip_host_malloc<T>(m * m);
        addFunc(A_21, A_22, S_1, m);
#if DEBUG == 1
        auto step = 0;
        std::cout << "Step: " << step++ << std::endl; // 0
        verify_matrix_addition(A_21, A_22, S_1, m);
#endif

        // S_2 = S_1 - A_11
        auto *S_2 = hip_host_malloc<T>(m * m);
        subtractFunc(S_1, A_11, S_2, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 1
        verify_matrix_subtraction(S_1, A_11, S_2, m);
#endif

        // S_3 = A_11 - A_21
        auto *S_3 = hip_host_malloc<T>(m * m);
        subtractFunc(A_11, A_21, S_3, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 2
        verify_matrix_subtraction(A_11, A_21, S_3, m);
#endif

        // S_4 = A_12 - S_2
        auto *S_4 = hip_host_malloc<T>(m * m);
        subtractFunc(A_12, S_2, S_4, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 3
        verify_matrix_subtraction(A_12, S_2, S_4, m);
#endif

        // S_5 = B_12 - B_11
        auto *S_5 = hip_host_malloc<T>(m * m);
        subtractFunc(B_12, B_11, S_5, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 4
        verify_matrix_subtraction(B_12, B_11, S_5, m);
#endif

        // S_6 = B_22 - S_5
        auto *S_6 = hip_host_malloc<T>(m * m);
        subtractFunc(B_22, S_5, S_6, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 5
        verify_matrix_subtraction(B_22, S_5, S_6, m);
#endif

        // S_7 = B_22 - B_12
        auto *S_7 = hip_host_malloc<T>(m * m);
        subtractFunc(B_22, B_12, S_7, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 6
        verify_matrix_subtraction(B_22, B_12, S_7, m);
#endif

        // S_8 = S_6 - B_21
        auto *S_8 = hip_host_malloc<T>(m * m);
        subtractFunc(S_6, B_21, S_8, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 7
        verify_matrix_subtraction(S_6, B_21, S_8, m);
#endif

        // ----------------------------------------------

        // M_1 = S_2 x S_6
        auto *M_1 = hip_host_malloc<T>(m * m);
        multiplyFunc(S_6, S_2, M_1, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 8
        verify_matrix_multiply(S_6, S_2, M_1, m);
#endif

        // M_2 = A_11 x B_11
        auto *M_2 = hip_host_malloc<T>(m * m);
        multiplyFunc(B_11, A_11, M_2, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 9
        verify_matrix_multiply(B_11, A_11, M_2, m);
#endif

        // M_3 = A_12 x B_21
        auto *M_3 = hip_host_malloc<T>(m * m);
        multiplyFunc(B_21, A_12, M_3, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 10
        verify_matrix_multiply(B_21, A_12, M_3, m);
#endif

        // M_4 = S_3 x S_7
        auto *M_4 = hip_host_malloc<T>(m * m);
        multiplyFunc(S_7, S_3, M_4, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 11
        verify_matrix_multiply(S_7, S_3, M_4, m);
#endif

        // // M_5 = S_1 x S_5
        auto *M_5 = hip_host_malloc<T>(m * m);
        multiplyFunc(S_5, S_1, M_5, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 12
        verify_matrix_multiply(S_5, S_1, M_5, m);
#endif

        // // M_6 = S_4 x B_22
        auto *M_6 = hip_host_malloc<T>(m * m);
        multiplyFunc(B_22, S_4, M_6, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 13
        verify_matrix_multiply(B_22, S_4, M_6, m);
#endif

        // M_7 = A_22 x S_8
        auto *M_7 = hip_host_malloc<T>(m * m);
        multiplyFunc(S_8, A_22, M_7, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 14
        verify_matrix_multiply(S_8, A_22, M_7, m);
#endif

        // ----------------------------------------------

        // V_1 = M_1 + M_2
        auto *V_1 = hip_host_malloc<T>(m * m);
        addFunc(M_1, M_2, V_1, m); // 15
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl;
        verify_matrix_addition(M_1, M_2, V_1, m);
#endif

        // V_2 = V_1 + M_4
        auto *V_2 = hip_host_malloc<T>(m * m);
        addFunc(V_1, M_4, V_2, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 16
        verify_matrix_addition(V_1, M_4, V_2, m);
#endif

        // V_3 = M_5 + M_6
        auto *V_3 = hip_host_malloc<T>(m * m);
        addFunc(M_5, M_6, V_3, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 17
        verify_matrix_addition(M_5, M_6, V_3, m);
#endif

        // ----------------------------------------------

        // C_11 = M_2 + M_3
        auto *C_11 = hip_host_malloc<T>(m * m);
        addFunc(M_2, M_3, C_11, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 18
        verify_matrix_addition(M_2, M_3, C_11, m);
#endif

        // C_12 = V_1 + V_3
        auto *C_12 = hip_host_malloc<T>(m * m);
        addFunc(V_1, V_3, C_12, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 19
        verify_matrix_addition(V_1, V_3, C_12, m);
#endif

        // C_21 = V_2 - M_7
        auto *C_21 = hip_host_malloc<T>(m * m);
        subtractFunc(V_2, M_7, C_21, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 20
        verify_matrix_subtraction(V_2, M_7, C_21, m);
#endif

        // C_22 = V_2 + M_5
        auto *C_22 = hip_host_malloc<T>(m * m);
        addFunc(V_2, M_5, C_22, m);
#if DEBUG == 1
        std::cout << "Step: " << step++ << std::endl; // 21
        verify_matrix_addition(V_2, M_5, C_22, m);
        std::cout << "Completed computation" << std::endl; // 22
#endif

        hipDeviceSynchronize();

        // ----------- POPULATING C-MATRIX ---------------

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

    float *A = hip_host_malloc<float>(n * n);
    float *B = hip_host_malloc<float>(n * n);
    float *C = hip_host_malloc<float>(n * n);

    for (int i = 0; i < n * n; ++i)
    {
        auto val = rand() % 10;
        A[i] = val;
        B[i] = val;
    }

#if TIME == 1
    auto start = std::chrono::high_resolution_clock::now();
#endif
    gpu_strassen_mul(A, B, C, n);
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
