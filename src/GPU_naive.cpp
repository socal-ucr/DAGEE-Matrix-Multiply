#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <GPU_Utils.hpp>
#include <verify.hpp>

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
    hip_multiply(A, B, C, n);
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
