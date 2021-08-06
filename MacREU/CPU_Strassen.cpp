#include <iostream>
#include <vector>
#include <cmath>
#include <string>

// Class Matrix matrix
// matrix.submatrices(size_t sz)
// {
//     return  std::container_typestd
// }

// NOTE: Matrices have to be square.
template <typename T>
void print_matrix(std::vector<T> matrix, std::string matrix_name = "Unknown Matrix")
{
    std::cout << std::endl
              << "Printing Matrix: " << matrix_name << std::endl;
    for (T i = 0; i < matrix.size(); i++)
    {
        if (fmod(i, sqrt(matrix.size())) == 0)
        {
            std::cout << std::endl;
        }
        std::cout << matrix.at(i) << "\t";
    }
    std::cout << std::endl;
}

template <typename T>
void add(std::vector<T> A, std::vector<T> B, std::vector<T> &C)
{
    for (int i = 0; i < A.size(); ++i)
    {
        // std::cout << "CHECK ADD 1" << std::endl;
        for (int j = 0; j < A.size(); ++j)
        {
            // std::cout << "CHECK ADD 2" << std::endl;
            C.push_back(A.at(i + j) + B.at(i + j));
            // std::cout << "i: " << i << " " << std::endl;
            // std::cout << "j: " << j << " " << std::endl;
            // std::cout << "boom da boom: " << i + j << " " << std::endl;
        }
        print_matrix(C, "C Matrix");
    }
    // print_matrix(C, "C Matrix");
}

template <typename T>
void sub(std::vector<T> A, std::vector<T> B, std::vector<T> &C)
{

    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A.size(); j++)
        {
            C.push_back(A.at(i + j) - B.at(i + j));
        }
    }
}

template <typename T>
void strassen_winograd(std::vector<T> A, std::vector<T> B, std::vector<double> &C, int dim)
{

    if (dim == 1)
    {
        C.push_back(A.at(0) * B.at(0));
        std::cout << "Result: " << C.at(0) << std::endl;
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

        C.push_back(M_2 + M_3);
        C.push_back(V_1 + V_3);
        C.push_back(V_2 - M_7);
        C.push_back(V_2 + M_5);

        std::cout << "STRASSEN-ALGORITHM RESULTS: " << std::endl
                  << std::endl;
        std::cout << C.at(0) << " " << C.at(1) << std::endl;
        std::cout << C.at(2) << " " << C.at(3) << std::endl
                  << std::endl;
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
                    B_11.push_back(B.at((i * dim) + j + m));
                    B_12.push_back(B.at((i * dim) + j + m + m));
                    B_21.push_back(B.at((i * dim) + j + m + (dim * 2)));
                    B_22.push_back(B.at((i * dim) + j + m + m + (dim * 2)));
                }
                else
                {
                    A_11.push_back(A.at(i + j));
                    A_12.push_back(A.at(i + j + m));
                    A_21.push_back(A.at(i + j + (dim * m)));
                    A_22.push_back(A.at(i + j + m + (dim * m)));
                    B_11.push_back(B.at(i + j + m + dim + dim));
                    B_12.push_back(B.at(i + j + m + m + dim + dim));
                    B_21.push_back(B.at(i + j + m + (dim * 2) + dim + dim));
                    B_22.push_back(B.at(i + j + m + m + (dim * 2) + dim + dim));
                }
            }
        }

        print_matrix(A_11, "A_11");
        print_matrix(A_12, "A_12");
        print_matrix(A_21, "A_21");
        print_matrix(A_22, "A_22");

        // print_matrix(B_11, "B_11");
        // print_matrix(B_12, "B_12");
        // print_matrix(B_21, "B_21");
        // print_matrix(B_22, "B_22");

        // std::cout << "OUT OF BOUNDS CHECK 1" << std::endl;

        // S_1 = A_21 + A_22
        std::vector<double> S_1;
        add(A_21, A_22, S_1);

        // S_2 = S_1 - A_11
        std::vector<double> S_2;
        sub(S_1, A_11, S_2);

        // S_3 = A_11 - A_21
        std::vector<double> S_3;
        sub(A_11, A_21, S_3);

        // S_4 = A_12 - S_2
        std::vector<double> S_4;
        sub(A_12, S_2, S_4);

        // S_5 = B_12 - B_11
        std::vector<double> S_5;
        sub(B_12, B_11, S_5);

        // S_6 = B_22 - S_5
        std::vector<double> S_6;
        sub(B_22, S_5, S_6);

        // S_7 = B_22 - B_12
        std::vector<double> S_7;
        sub(B_22, B_12, S_7);

        // S_8 = S_6 - B_21
        std::vector<double> S_8;
        sub(S_6, B_21, S_8);

        // ----------------------------------------------

        std::cout << "OUT OF BOUNDS CHECK 1" << std::endl;

        // M_1 = S_2 x S_6
        std::vector<double> M_1;
        strassen_winograd(S_6, S_6, M_1, m);

        // M_2 = A_11 x B_11
        std::vector<double> M_2;
        strassen_winograd(A_11, B_11, M_2, m);

        // M_3 = A_12 x B_21
        std::vector<double> M_3;
        strassen_winograd(A_12, B_21, M_3, m);

        // M_4 = S_3 x S_7
        std::vector<double> M_4;
        strassen_winograd(S_3, S_7, M_4, m);

        // M_5 = S_1 x S_5
        std::vector<double> M_5;
        strassen_winograd(S_1, S_5, M_5, m);

        // M_6 = S_4 x B_22
        std::vector<double> M_6;
        strassen_winograd(S_4, B_22, M_6, m);

        // M_7 = A_22 x S_8
        std::vector<double> M_7;
        strassen_winograd(A_22, S_8, M_7, m);

        // ----------------------------------------------

        // V_1 = M_1 + M_2
        std::vector<double> V_1;
        add(M_1, M_2, V_1);

        // V_2 = V_1 + M_4
        std::vector<double> V_2;
        add(V_1, M_4, V_2);

        // V_3 = M_5 + M_6
        std::vector<double> V_3;
        add(M_5, M_6, V_3);

        // ----------------------------------------------

        // C_11 = M_2 + M_3
        std::vector<double> C_11;
        add(M_2, M_3, C_11);

        // C_12 = V_1 + V_3
        std::vector<double> C_12;
        add(V_1, V_3, C_12);

        // C_21 = V_2 - M_7
        std::vector<double> C_21;
        sub(V_2, M_7, C_21);

        // C_22 = V_2 + M_5
        std::vector<double> C_22;
        add(V_2, M_5, C_22);
    }
}

int main()
{
    int n = 8;
    std::vector<double> A;
    std::vector<double> B;
    std::vector<double> C;

    for (double i = 0; i < n * n; ++i)
    {
        A.push_back(i);
        B.push_back(i);
    }

    print_matrix(A, "matrix A");
    print_matrix(B, "matrix B");

    strassen_winograd(A, B, C, n);

    return 0;
}
