#ifndef CPU_UTILS_HPP
#define CPU_UTILS_HPP

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

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
void add_kernel(const T *A, const T *B, T *C, size_t n)
{
  for (auto i = 0; i < n; i++)
  {
    C[i] = A[i] + B[i];
  }
}

template <typename T>
void matrixMul_kernel(const T *A, const T *B, T *C, size_t n)
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
void matrixMul(const std::vector<T> &A, const std::vector<T> &B, std::vector<T> &C, size_t n)
{
  matrixMul_kernel(&A[0], &B[0], &C[0], n);
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

template <typename T>
void sub_kernel(const T *A, const T *B, T *C, size_t n)
{
  for (auto i = 0; i < n; i++)
  {
    C[i] = A[i] - B[i];
  }
}

#endif // CPU_UTILS_HPP
