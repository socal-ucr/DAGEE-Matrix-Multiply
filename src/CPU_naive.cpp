#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <CPU_Utils.hpp>
#include <verify.hpp>

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
    matrixMul(A, B, C, n);
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
