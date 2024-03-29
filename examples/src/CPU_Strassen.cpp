#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <CPU_Utils.hpp>
#include <verify.hpp>

template <typename T>
void strassenMul(std::vector<T> A, std::vector<T> B, std::vector<double> &C, int dim)
{

    if (dim == 1)
    {
        C.push_back(A.at(0) * B.at(0));
    }
    else if (dim == 2)
    {
        auto S_1 = A.at(2) + A.at(3);
        auto S_2 = S_1 - A.at(0);
        auto S_3 = A.at(0) - A.at(2);
        auto S_4 = A.at(1) - S_2;
        auto S_5 = B.at(1) - B.at(0);
        auto S_6 = B.at(3) - S_5;
        auto S_7 = B.at(3) - B.at(1);
        auto S_8 = S_6 - B.at(2);

        auto M_1 = S_2 * S_6;
        auto M_2 = A.at(0) * B.at(0);
        auto M_3 = A.at(1) * B.at(2);
        auto M_4 = S_3 * S_7;
        auto M_5 = S_1 * S_5;
        auto M_6 = S_4 * B.at(3);
        auto M_7 = A.at(3) * S_8;

        auto V_1 = M_1 + M_2;
        auto V_2 = V_1 + M_4;
        auto V_3 = M_5 + M_6;

        C.at(0) = M_2 + M_3;
        C.at(1) = V_1 + V_3;
        C.at(2) = V_2 - M_7;
        C.at(3) = V_2 + M_5;

        print_matrix(C, "Strassen-Winograd Matrix");
    }
    else
    {
        int m = dim / 2;

        // sub_matrices
        std::vector<double> A_11;
        std::vector<double> A_12;
        std::vector<double> A_21;
        std::vector<double> A_22;
        std::vector<double> B_11;
        std::vector<double> B_12;
        std::vector<double> B_21;
        std::vector<double> B_22;

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                if (i != 0)
                {
                    A_11.push_back(A.at((i * dim) + j));
                    A_12.push_back(A.at((i * dim) + j + m));
                    A_21.push_back(A.at((i * dim) + j + (dim * m)));
                    A_22.push_back(A.at((i * dim) + j + m + (dim * m)));

                    B_11.push_back(B.at((i * dim) + j));
                    B_12.push_back(B.at((i * dim) + j + m));
                    B_21.push_back(B.at((i * dim) + j + (dim * m)));
                    B_22.push_back(B.at((i * dim) + j + m + (dim * m)));
                }
                else
                {
                    A_11.push_back(A.at(i + j));
                    A_12.push_back(A.at(i + j + m));
                    A_21.push_back(A.at(i + j + (dim * m)));
                    A_22.push_back(A.at(i + j + m + (dim * m)));

                    B_11.push_back(B.at(i + j));
                    B_12.push_back(B.at(i + j + m));
                    B_21.push_back(B.at(i + j + (dim * m)));
                    B_22.push_back(B.at(i + j + m + (dim * m)));
                }
            }
        }

        // S_1 = A_21 + A_22
        std::vector<double> S_1 = add(A_21, A_22);

        // S_2 = S_1 - A_11
        std::vector<double> S_2 = sub(S_1, A_11);

        // S_3 = A_11 - A_21
        std::vector<double> S_3 = sub(A_11, A_21);

        // S_4 = A_12 - S_2
        std::vector<double> S_4 = sub(A_12, S_2);

        // S_5 = B_12 - B_11
        std::vector<double> S_5 = sub(B_12, B_11);

        // S_6 = B_22 - S_5
        std::vector<double> S_6 = sub(B_22, S_5);

        // S_7 = B_22 - B_12
        std::vector<double> S_7 = sub(B_22, B_12);

        // S_8 = S_6 - B_21
        std::vector<double> S_8 = sub(S_6, B_21);

        // ----------------------------------------------

        // M_1 = S_2 x S_6
        std::vector<double> M_1(m * m);
        matrixMul(S_2, S_6, M_1, m);

        // M_2 = A_11 x B_11
        std::vector<double> M_2(m * m);
        matrixMul(A_11, B_11, M_2, m);

        // M_3 = A_12 x B_21
        std::vector<double> M_3(m * m);
        matrixMul(A_12, B_21, M_3, m);

        // M_4 = S_3 x S_7
        std::vector<double> M_4(m * m);
        matrixMul(S_3, S_7, M_4, m);

        // // M_5 = S_1 x S_5
        std::vector<double> M_5(m * m);
        matrixMul(S_1, S_5, M_5, m);

        // // M_6 = S_4 x B_22
        std::vector<double> M_6(m * m);
        matrixMul(S_4, B_22, M_6, m);

        // M_7 = A_22 x S_8
        std::vector<double> M_7(m * m);
        matrixMul(A_22, S_8, M_7, m);

        // ----------------------------------------------

        // V_1 = M_1 + M_2
        std::vector<double> V_1 = add(M_1, M_2);

        // V_2 = V_1 + M_4
        std::vector<double> V_2 = add(V_1, M_4);

        // V_3 = M_5 + M_6
        std::vector<double> V_3 = add(M_5, M_6);

        // ----------------------------------------------

        // C_11 = M_2 + M_3
        std::vector<double> C_11 = add(M_2, M_3);

        // C_12 = V_1 + V_3
        std::vector<double> C_12 = add(V_1, V_3);

        // C_21 = V_2 - M_7
        std::vector<double> C_21 = sub(V_2, M_7);

        // C_22 = V_2 + M_5
        std::vector<double> C_22 = add(V_2, M_5);

        // ----------- POPULATING C-MATRIX ---------------

        std::vector<T> temp;

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                temp.push_back(C_11.at(j + (i * m)));
            }
            for (int j = 0; j < m; ++j)
            {
                temp.push_back(C_12.at(j + (i * m)));
            }
        }
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                temp.push_back(C_21.at(j + (i * m)));
            }
            for (int j = 0; j < m; ++j)
            {
                temp.push_back(C_22.at(j + (i * m)));
            }
        }

        for (int i = 0; i < temp.size(); ++i)
        {
            C.at(i) = temp.at(i);
        }
    }
}

int main(int argc, char **argv)
{
    int n = (argc < 2) ? 8 : atoi(argv[1]);

    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> C(n * n);

    for (double i = 0; i < n * n; ++i)
    {
        auto val = rand() % 10;
        A[i] = val;
        B[i] = val;
    }

#if TIME == 1
    auto start = std::chrono::high_resolution_clock::now();
#endif
    strassenMul(A, B, C, n);
#if TIME == 1
    auto stop = std::chrono::high_resolution_clock::now();
#endif

#if DEBUG == 1
    print_matrix(C, "matrix_C");
    verify_matrix_multiply(&A[0], &B[0], &C[0], n);
#endif

    std::cout << "Done" << std::endl;

#if TIME == 1
    auto duration_s = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Runtime(microseconds): " << duration_s.count() << std::endl;
#endif

    return 0;
}
