#ifndef CPU_UTILS_HPP
#define CPU_UTILS_HPP

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// NOTE: Matrices have to be square.
template <typename T>
void print_matrix(const std::vector<T> &matrix, std::string matrix_name = "Unknown Matrix")
{
  std::cout << std::endl
            << "Printing Matrix: " << matrix_name << std::endl;

  std::cout << "Size of Matrix: " << matrix.size() << std::endl;
  for (double i = 0; i < matrix.size(); i++)
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
std::vector<T> add(const std::vector<T> &A, const std::vector<T> &B)
{
  std::vector<T> C;
  for (T j = 0; j < A.size(); ++j)
  {
    C.push_back(A.at(j) + B.at(j));
  }
  return C;
}

template <typename T>
void matrixMul(const std::vector<T> &A, const std::vector<T> &B, std::vector<T> &C, size_t n)
{
  for (auto i = 0; i < n; i++)
  {
    for (auto j = 0; j < n; j++)
    {
      for (auto k = 0; k < n; k++)
      {
        C[i * n + k] += A[i * n + j] * B[j * n + k];
      }
    }
  }
}

template <typename T>
std::vector<T> sub(const std::vector<T> &A, const std::vector<T> &B)
{
  std::vector<T> C;
  for (T j = 0; j < A.size(); ++j)
  {
    C.push_back(A.at(j) - B.at(j));
  }
  return C;
}

#endif // CPU_UTILS_HPP
