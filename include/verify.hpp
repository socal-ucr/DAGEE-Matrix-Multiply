#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <vector>

template <typename T>
void verify_matrix_multiply(const T *A, const T *B, const T *C, int m)
{
  std::vector<T> tempResult(m * m);
  for (auto i = 0; i < m; i++)
  {
    for (auto j = 0; j < m; j++)
    {
      for (auto k = 0; k < m; k++)
      {
        tempResult[i * m + k] += A[i * m + j] * B[j * m + k];
      }
    }
  }

  for (auto i = 0; i < m; i++)
  {
    if (std::abs(C[i] - tempResult[i]) >= 1)
    {
      std::cerr << "Found error in vector at " << i << std::endl;
      std::cerr << "Expected: " << C[i] << " Obtained: " << tempResult[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

template <typename T>
void verify_matrix_addition(const T *A, const T *B, const T *C, int m)
{
  for (auto i = 0; i < m * m; ++i)
  {
    if (C[i] != A[i] + B[i])
    {
      std::cerr << "Found error in vector Add at " << i << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

template <typename T>
void verify_matrix_subtraction(const T *A, const T *B, const T *C, int m)
{
  for (auto i = 0; i < m * m; ++i)
  {
    if (C[i] != A[i] - B[i])
    {
      std::cerr << "Found error in vector Sub at " << i << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

#endif // COMMON_HPP
