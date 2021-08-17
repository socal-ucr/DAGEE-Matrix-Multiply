#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <vector>

// NOTE: Matrices have to be square.
template <typename T>
void print_matrix(T *matrix, size_t length, std::string matrix_name = "Unknown Matrix")
{
  std::cout << std::endl
            << "Printing Matrix: " << matrix_name << std::endl;

  std::cout << "Size of Matrix: " << length << std::endl;

  for (int i = 0; i < length; i++)
  {
    if (fmod(i, sqrt(length)) == 0)
    {
      std::cout << std::endl;
    }
    std::cout << matrix[i] << "\t";
  }
  std::cout << std::endl;
}

template <typename T>
void print_matrix(const std::vector<T> &matrix, std::string matrix_name = "Unknown Matrix")
{
  print_matrix(&matrix[0], matrix.size(), matrix_name);
}

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
      std::cerr << "Found error in Matrix Mul at " << i << std::endl;
      std::cerr << "Expected: " << tempResult[i] << " Obtained: " << C[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::cout << "Success" << std::endl;
}

template <typename T>
void verify_matrix_addition(const T *A, const T *B, const T *C, int m)
{
  for (auto i = 0; i < m * m; ++i)
  {
    if (C[i] != A[i] + B[i])
    {
      std::cerr << "Found error in vector Add at " << i << std::endl;
      std::cerr << "Expected: " << A[i] + B[i] << " Obtained: " << C[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::cout << "Success" << std::endl;
}

template <typename T>
void verify_matrix_subtraction(const T *A, const T *B, const T *C, int m)
{
  for (auto i = 0; i < m * m; ++i)
  {
    if (C[i] != A[i] - B[i])
    {
      std::cerr << "Found error in vector Sub at " << i << std::endl;
      std::cerr << "Expected: " << A[i] - B[i] << " Obtained: " << C[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::cout << "Success" << std::endl;
}

#endif // COMMON_HPP
