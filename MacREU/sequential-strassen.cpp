#include <rocblas.h>
#include <vector>
#include <iostream>
#include <hip/hip_runtime.h>

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

void strassen(rocblas_handle handle,    // (?) handle to the rocblas lib context queue
              rocblas_operation transA, // (?)
              rocblas_operation transB, // (?)
              rocblas_int m,            // matrix dimension m
              rocblas_int n,            // matrix dimension n
              rocblas_int k,            // matrix dimension k
              const float *alpha,       // (?) scalar value
              const float *A,           // (?) device pointer to the first matrix A_0 on the GPU
              rocblas_int lda,          // (?) specifies the leading dimension of A. Each A_i is of dimension ( lda, n ).
              const float *beta,        // scalar value
              const float *B,           // device pointer to the first matrix B_0 on the GPU
              rocblas_int ldb,          // specifies the leading dimension of B. Each B_i is of dimension ( ldb, n ).
              float *C,                 // device pointer to the first matrix C_0 on the GPU. Each C_i is of dimension ( ldc, n ).
              rocblas_int ldc,          // specifies the increment between values of C
              float *S_1,
              rocblas_int ld_s1,
              float *S_2,
              rocblas_int ld_s2,
              float *S_3,
              rocblas_int ld_s3,
              float *S_4,
              rocblas_int ld_s4,
              float *S_5,
              rocblas_int ld_s5,
              float *S_6,
              rocblas_int ld_s6,
              float *S_7,
              rocblas_int ld_s7,
              float *S_8,
              rocblas_int ld_s8,
              float *M_1,
              rocblas_int ld_m1,
              float *M_2,
              rocblas_int ld_m2,
              float *M_3,
              rocblas_int ld_m3,
              float *M_4,
              rocblas_int ld_m4,
              float *M_5,
              rocblas_int ld_m5,
              float *M_6,
              rocblas_int ld_m6,
              float *M_7,
              rocblas_int ld_m7,
              float *V_1,
              rocblas_int ld_v1,
              float *V_2,
              rocblas_int ld_v2,
              float *V_3,
              rocblas_int ld_v3)
{

    std::cout << "STRASSEN-ALGORITHM: " << std::endl
              << std::endl;

    std::cout << std::endl;
    std::cout << "MATRIX A: " << std::endl
              << std::endl;
    std::cout << *A << " " << *(A + 1) << std::endl;
    std::cout << *(A + n) << " " << *(A + n + 1) << std::endl
              << std::endl;

    std::cout << std::endl;
    std::cout << "MATRIX B: " << std::endl
              << std::endl;
    std::cout << *B << " " << *(B + 1) << std::endl;
    std::cout << *(B + n) << " " << *(B + n + 1) << std::endl
              << std::endl;

    const float negativeBeta = -1 * *beta;

    std::cout << "S_1 BEFORE: " << *S_1 << std::endl;

    // S_1 = A_21 + A_22
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, (A + n), lda, beta, (A + n + 1), lda, S_1, ld_s1);

    std::cout << "S_1 AFTER: " << *(A + n) << " + " << *(A + n + 1) << " = " << *S_1 << std::endl;
    std::cout << "m = " << m << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "alpha = " << *alpha << std::endl;
    std::cout << "lda = " << lda << std::endl;
    std::cout << "beta = " << *beta << std::endl;
    std::cout << "ld_s1 = " << ld_s1 << std::endl
              << std::endl;

    // S_2 = S_1 - A_11
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, S_1, ld_s1, &negativeBeta, A, lda, S_2, ld_s2);
    std::cout << "S_2: " << *S_1 << " - " << *A << " = " << *S_2 << std::endl;

    // S_3 = A_11 - A_21
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, A, lda, &negativeBeta, (A + n), lda, S_3, ld_s3);

    // S_4 = A_12 - S_2
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, (A + 1), lda, &negativeBeta, S_2, lda, S_4, ld_s4);

    // S_5 = B_12 - B_11
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, (B + 1), ldb, &negativeBeta, (B + n + 1), ldb, S_5, ld_s5);

    // S_6 = B_22 - S_5
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, (B + n + 1), ldb, &negativeBeta, S_5, ldb, S_6, ld_s6);

    // S_7 = B_22 - B_12
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, (B + n + 1), ldb, &negativeBeta, (B + 1), ldb, S_7, ld_s7);

    // S_8 = S_6 - B_21
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, S_6, ldb, &negativeBeta, (B + n), ldb, S_8, ld_s8);

    // ----------------------------------------------

    // M_1 = S_2 x S_6
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, alpha, S_2, ld_s2, S_6, ld_s6, beta, M_1, ld_m1);

    // M_2 = A_11 x B_11
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, alpha, A, lda, B, ldb, beta, M_2, ld_m2);

    // M_3 = A_12 x B_21
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, alpha, (A + n), lda, (B + n), ldb, beta, M_3, ld_m3);

    // M_4 = S_3 x S_7
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, alpha, S_3, ld_s3, S_7, ld_s7, beta, M_4, ld_m4);

    // M_5 = S_1 x S_5
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, alpha, S_1, ld_s1, S_5, ld_s5, beta, M_5, ld_m5);

    // M_6 = S_4 x B_22
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, alpha, S_4, ld_s4, (B + n + 1), ldb, beta, M_6, ld_m6);

    // M_7 = A_22 x S_8
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, alpha, (A + n + 1), lda, S_8, ld_s8, beta, M_7, ld_m7);

    // ----------------------------------------------

    // V_1 = M_1 + M_2
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, M_1, ld_m1, beta, M_2, ld_m2, V_1, ld_v1);

    // V_2 = V_1 + M_4
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, V_1, ld_v1, beta, M_4, ld_m4, V_2, ld_v2);

    // V_3 = M_5 + M_6
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, M_5, ld_m5, beta, M_6, ld_m6, V_3, ld_v3);

    // ----------------------------------------------

    // C_11 = M_2 + M_3
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, M_2, ld_m2, beta, M_3, ld_m3, C, ldc);

    // C_12 = V_1 + V_3
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, V_1, ld_v1, beta, V_3, ld_v3, (C + 1), ldc);

    // C_21 = V_2 - M_7
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, V_2, ld_v2, &negativeBeta, M_7, ld_m7, (C + n), ldc);

    // C_22 = V_2 + M_5
    rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, alpha, V_2, ld_v2, beta, M_5, ld_m5, (C + n + 1), ldc);

    // ---------------------------------------------- OUTPUT RESULT
    std::cout << std::endl;
    std::cout << "STRASSEN-ALGORITHM RESULTS: " << std::endl
              << std::endl;
    std::cout << *C << " " << *(C + 1) << std::endl;
    std::cout << *(C + n) << " " << *(C + n + 1) << std::endl
              << std::endl;
}

#define DIM1 2
#define DIM2 2
#define DIM3 2

int N = 2;

void verify(int matrix_A[2][2],
            int matrix_B[2][2],
            int ans[2][2])
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            ans[i][j] = 0;
            for (k = 0; k < N; k++)
            {
                ans[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }
}

int main()
{
    // int i, j;
    // int answer[4][4]; // To store result
    // int matrix_A[4][4] = {{1, 1, 1, 1},
    //                       {2, 2, 2, 2},
    //                       {3, 3, 3, 3},
    //                       {4, 4, 4, 4}};

    // int matrix_B[4][4] = {{1, 1, 1, 1},
    //                       {2, 2, 2, 2},
    //                       {3, 3, 3, 3},
    //                       {4, 4, 4, 4}};

    int i, j;
    int answer[2][2]; // To store result
    int matrix_A[2][2] = {{1, 2},
                          {3, 4}};

    int matrix_B[2][2] = {{1, 2},
                          {3, 4}};

    verify(matrix_A, matrix_B, answer);
    // std::cout << std::endl
    //           << std::endl;
    // std::cout << "RESULTS MATRIX: \n \n";
    // for (i = 0; i < 2; i++)
    // {
    //     for (j = 0; j < 2; j++)
    //     {
    //         std::cout << answer[i][j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    std::cout << std::endl;

    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
    float alpha = 1, beta = 1;

    rocblas_int m = DIM1, n = DIM2, k = DIM3;
    rocblas_int lda, ldb, ldc, size_a, size_b, size_c;
    rocblas_int ld_s1 = DIM1, ld_s2 = DIM1, ld_s3 = DIM1, ld_s4 = DIM1, ld_s5 = DIM1, ld_s6 = DIM1, ld_s7 = DIM1, ld_s8 = DIM1, ld_m1 = DIM1, ld_m2 = DIM1, ld_m3 = DIM1, ld_m4 = DIM1, ld_m5 = DIM1, ld_m6 = DIM1, ld_m7 = DIM1, ld_v1 = DIM1, ld_v2 = DIM1, ld_v3 = DIM1;
    int a_stride_1, a_stride_2, b_stride_1, b_stride_2;

    std::cout << "sgemm example" << std::endl;

    if (transa == rocblas_operation_none)
    {
        lda = m;
        size_a = k * lda;
        a_stride_1 = 1;
        a_stride_2 = lda;
        std::cout << "N";
    }
    else
    {
        lda = k;
        size_a = m * lda;
        a_stride_1 = lda;
        a_stride_2 = 1;
        std::cout << "T";
    }
    if (transb == rocblas_operation_none)
    {
        ldb = k;
        size_b = n * ldb;
        b_stride_1 = 1;
        b_stride_2 = ldb;
        std::cout << "N: ";
    }
    else
    {
        ldb = n;
        size_b = k * ldb;
        b_stride_1 = ldb;
        b_stride_2 = 1;
        std::cout << "T: ";
    }
    ldc = m;
    size_c = n * ldc;

    std::cout << "M value: " << m << std::endl;
    std::cout << "b_stride_1 value: " << b_stride_1 << std::endl
              << std::endl;

    std::cout << "CHECK 1" << std::endl;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<float> ha(size_a);
    std::vector<float> hb(size_b);
    std::vector<float> hc(size_c);
    std::vector<float> hc_gold(size_c);

    // initial data on host

    std::vector<int> numbers = {1, 2, 3, 4};
    std::cout << "check a" << std::endl;

    ha[0] = 1;
    ha[1] = 2;
    ha[2] = 3;
    ha[3] = 4;

    hb[0] = 1;
    hb[1] = 2;
    hb[2] = 3;
    hb[3] = 4;

    hc[0] = 1;
    hc[1] = 2;
    hc[2] = 3;
    hc[3] = 4;

    // srand(1);
    // for (int i = 0; i < size_a; ++i)
    // {
    //     ha[i] = rand() % 17;
    // }
    // for (int i = 0; i < size_b; ++i)
    // {
    //     hb[i] = rand() % 17;
    // }
    // for (int i = 0; i < size_c; ++i)
    // {
    //     hc[i] = rand() % 17;
    // }
    // hc_gold = hc;

    std::cout << "CHECK 2" << std::endl;

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipHostMalloc(&da, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipHostMalloc(&db, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipHostMalloc(&dc, size_c * sizeof(float)));

    std::cout << "CHECK 3" << std::endl;

    // allocating temporary matrices on device
    float *S_1, *S_2, *S_3, *S_4, *S_5, *S_6, *S_7, *S_8;
    float *M_1, *M_2, *M_3, *M_4, *M_5, *M_6, *M_7;
    float *V_1, *V_2, *V_3;
    float *C_11, *C_12, *C_21, *C_22;

    CHECK_HIP_ERROR(hipMalloc(&S_1, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&S_2, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&S_3, size_c * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&S_4, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&S_5, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&S_6, size_c * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&S_7, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&S_8, size_b * sizeof(float)));

    CHECK_HIP_ERROR(hipMalloc(&M_1, size_c * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&M_2, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&M_3, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&M_4, size_c * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&M_5, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&M_6, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&M_7, size_c * sizeof(float)));

    CHECK_HIP_ERROR(hipMalloc(&V_1, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&V_2, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&V_3, size_c * sizeof(float)));

    // CHECK_HIP_ERROR(hipMalloc(&C_11, size_a * sizeof(float)));
    // CHECK_HIP_ERROR(hipMalloc(&C_12, size_b * sizeof(float)));
    // CHECK_HIP_ERROR(hipMalloc(&C_21, size_c * sizeof(float)));
    // CHECK_HIP_ERROR(hipMalloc(&C_22, size_a * sizeof(float)));

    std::cout << "CHECK 4" << std::endl;

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));

    std::cout << "CHECK 5" << std::endl;

    rocblas_handle handle;
    // CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
    rocblas_create_handle(&handle);

    rocblas_sgemm(handle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc);

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));

    std::cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << lda
              << ", " << ldb << ", " << ldc << std::endl;

    float max_relative_error = std::numeric_limits<float>::min();

    std::cout << "CHECK 6" << std::endl;

    /* 
    ---------------------------------------------------------------------------------------------
    
    NOW THAT MATRICES HAVE BEEN INITIALIZED, CALL STRASSEN ALGORITHM TO COMPARE RUNTIME TO SGEMM  
    
    ---------------------------------------------------------------------------------------------
    */

    std::cout << std::endl
              << std::endl;
    std::cout << "CORRECT MATRIX: \n \n";

    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            std::cout << answer[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    strassen(handle, // (?) handle to the rocblas lib context queue
             transa, // (?)
             transb, // (?)
             m,      // matrix dimension m
             n,      // matrix dimension n
             k,      // matrix dimension k
             &alpha, // (?) scalar value
             da,     // (?) device pointer to the first matrix A_0 on the GPU
             lda,    // (?) specifies the leading dimension of A. Each A_i is of dimension ( lda, n ).
             &beta,  // scalar value
             db,     // device pointer to the first matrix B_0 on the GPU
             ldb,    // specifies the leading dimension of B. Each B_i is of dimension ( ldb, n ).
             dc,     // device pointer to the first matrix C_0 on the GPU. Each C_i is of dimension ( ldc, n ).
             ldc,    // specifies the increment between values of C
             S_1,
             ld_s1,
             S_2,
             ld_s2,
             S_3,
             ld_s3,
             S_4,
             ld_s4,
             S_5,
             ld_s5,
             S_6,
             ld_s6,
             S_7,
             ld_s7,
             S_8,
             ld_s8,
             M_1,
             ld_m1,
             M_2,
             ld_m2,
             M_3,
             ld_m3,
             M_4,
             ld_m4,
             M_5,
             ld_m5,
             M_6,
             ld_m6,
             M_7,
             ld_m7,
             V_1,
             ld_v1,
             V_2,
             ld_v2,
             V_3,
             ld_v3);

    std::cout << "CHECK 7" << std::endl;

    return 0;
}
