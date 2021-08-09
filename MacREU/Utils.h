#include <iostream>
#include <vector>
#include <cmath>
#include <string>

// NOTE: Matrices have to be square.
template <typename T>
void print_matrix(std::vector<T> &matrix, std::string matrix_name = "Unknown Matrix")
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
void add(std::vector<T> A, std::vector<T> B, std::vector<T> &C)
{
  for (T j = 0; j < A.size(); ++j)
  {
    C.push_back(A.at(j) + B.at(j));
  }
  print_matrix(C, "C Matrix");
}

template <typename T>
void matrixMul(std::vector<T> &A, std::vector<T> &B, std::vector<T> &C, size_t n)
{
  // auto n = static_cast<size_t>(std::sqrt(n));

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
void sub(std::vector<T> A, std::vector<T> B, std::vector<T> &C)
{

  for (int j = 0; j < A.size(); ++j)
  {
    C.push_back(A.at(j) - B.at(j));
  }
  print_matrix(C, "C Matrix");
}
