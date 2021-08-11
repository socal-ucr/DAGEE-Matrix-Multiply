#include <iostream>

int main()
{

    int A[2][2] = {{1, 2},
                   {3, 4}};

    int B[2][2] = {{1, 2},
                   {3, 4}};
    int n = 2;

    std::cout << B[0][0] << std::endl;
    std::cout << B[0][1] << std::endl;
    std::cout << B[1][0] << std::endl;
    std::cout << B[1][1] << std::endl;

    auto S_1 = A[1][0] + A[1][1];
    auto S_2 = S_1 - A[0][0];
    auto S_3 = A[0][0] - A[1][0];
    auto S_4 = A[0][1] - S_2;
    auto S_5 = B[0][1] - B[0][0];
    auto S_6 = B[1][1] - S_5;
    auto S_7 = B[1][1] - B[0][1];
    auto S_8 = S_6 - B[1][0];

    auto M_1 = S_2 * S_6;
    auto M_2 = A[0][0] * B[0][0];
    auto M_3 = A[0][1] * B[1][0];
    auto M_4 = S_3 * S_7;
    auto M_5 = S_1 * S_5;
    auto M_6 = S_4 * B[1][1];
    auto M_7 = A[1][1] * S_8;

    auto V_1 = M_1 + M_2;
    auto V_2 = V_1 + M_4;
    auto V_3 = M_5 + M_6;
    auto C_11 = M_2 + M_3;
    auto C_12 = V_1 + V_3;
    auto C_21 = V_2 - M_7;
    auto C_22 = V_2 + M_5;

    std::cout << "RESULT MATRIX: " << std::endl
              << std::endl;

    std::cout << C_11 << " " << C_12 << std::endl;
    std::cout << C_21 << " " << C_22 << std::endl
              << std::endl;

    return 0;
}
