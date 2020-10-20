//
// Created by kellerberrin on 18/10/20.
//

#ifndef KEG_CUDA_CUBLAS_H
#define KEG_CUDA_CUBLAS_H

#include "kel_exec_env.h"
#include "keg_cuda_device.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <string>
#include <memory>


namespace kellerberrin::gpu {   //  organization::project level namespace


class CheckCublasCode {

public:

  CheckCublasCode() = delete;
  ~CheckCublasCode() = delete;

  static bool check(cublasStatus_t status_code, const std::string &module_text = "");

private:

  static std::pair<bool, std::string> checkCublasError(cublasStatus_t status_code);

};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// A simple matrix allocated on the GPU device.
template<class T> class CuBlasHostMatrix; //forward.

template<class T>
class CuBlasGPUMatrix {

public:

  CuBlasGPUMatrix();
  CuBlasGPUMatrix(CUdeviceptr gpu_address, size_t Rows, size_t columns);
  CuBlasGPUMatrix(size_t Rows, size_t columns);
  ~CuBlasGPUMatrix();

  CuBlasGPUMatrix<T>& operator=(const CuBlasGPUMatrix<T>& copy) = default;

  bool transferToHost(CuBlasHostMatrix<T>& host_matrix) const;
  // Create the host matrix and transfer.
  CuBlasHostMatrix<T> transferToHost() const;

  [[nodiscard]] size_t rows() const { return rows_; }
  [[nodiscard]] size_t columns() const { return columns_; }
  [[nodiscard]] T* data() const { return device_ptr_; }

private:

  T* device_ptr_;
  size_t rows_;
  size_t columns_;

};


template<class T>
CuBlasGPUMatrix<T>::CuBlasGPUMatrix() {

  device_ptr_ = nullptr;
  rows_ = 0;
  columns_ = 0;

}


template<class T>
CuBlasGPUMatrix<T>::CuBlasGPUMatrix(CUdeviceptr gpu_address, size_t rows, size_t columns) {

  device_ptr_ = reinterpret_cast<T*>(gpu_address);
  rows_ = rows;
  columns_ = columns;

}


template<class T>
CuBlasGPUMatrix<T>::CuBlasGPUMatrix(size_t rows, size_t columns) {

  const size_t mem_size_bytes = rows * columns * sizeof(T);
  device_ptr_ = nullptr;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_ptr_), mem_size_bytes));
  if (device_ptr_ == nullptr) {

    rows_ = 0;
    columns_ = 0;

  } else {

    rows_ = rows;
    columns_ = columns;

  }

}


template<class T>
CuBlasGPUMatrix<T>::~CuBlasGPUMatrix() {

  if (data() != nullptr) {

    checkCudaErrors(cudaFree(static_cast<void *>(device_ptr_)));

  }

}

/*
template<class T>
CuBlasGPUMatrix<T>& CuBlasGPUMatrix<T>::operator=(const CuBlasGPUMatrix<T>& copy) {

  device_ptr_ = copy.data();
  rows_ = copy.rows();
  columns_ = copy.columns();
  return *this;

}
*/

template<class T>
bool CuBlasGPUMatrix<T>::transferToHost(CuBlasHostMatrix<T>& host_matrix) const {

  // Check dimensions.
  if (host_matrix.rows() != rows() or host_matrix.columns() != columns()) {

    ExecEnv::log().warn("CuBlasGPUMatrix::transferToHost; GPU Matrix({},{}) != Host Matrix({},{}), dimensional mismatch",
                        rows(), columns(), host_matrix.rows(), host_matrix.columns());
    return  false;

  }

  size_t memory_size = rows() * columns() * sizeof(T);
  checkCudaErrors(cudaMemcpy(static_cast<void*>(data()), static_cast<void*>(host_matrix.data()), memory_size, cudaMemcpyDeviceToHost));

  return true;

}

template<class T>
CuBlasHostMatrix<T> CuBlasGPUMatrix<T>::transferToHost() const {

  CuBlasHostMatrix<T> host_matrix(rows(), columns());
  if (not transferToHost(host_matrix)) {

    CuBlasHostMatrix<T>(0, 0); // return the null matrix.

  }

  return host_matrix;

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// A simple host matrix malloced on the CPU.

template<class T>
class CuBlasHostMatrix {

public:

  CuBlasHostMatrix(size_t Rows, size_t columns);
  ~CuBlasHostMatrix() = default;

  // All elements are in zero offset row major (C/C++) format.
  T operator()(size_t row, size_t column) { assert(row < rows_); assert(column < columns); return matrix_array_[rowMajorIndex(row, column)]; }
  void operator()(size_t row, size_t column,const T value) { assert(row < rows_); assert(column < columns); matrix_array_[rowMajorIndex(row, column)] = value; }
  // Convert to column major format.
  bool transpose();

  [[nodiscard]] size_t rows() const { return rows_; }
  [[nodiscard]] size_t columns() const { return columns_; }
  [[nodiscard]] T* data() const { return matrix_array_.get(); }

  // Implemented on the CPU.
  [[nodiscard]] bool operator==(const CuBlasHostMatrix& matrix) const;
  CuBlasHostMatrix<T> operator*(const CuBlasHostMatrix<T>& B) const;
  CuBlasHostMatrix<T> operator+(const CuBlasHostMatrix<T>& B) const;

  // Transfer to GPU device.
  bool transferToGPU(CuBlasGPUMatrix<T>& gpu_matrix) const;
  // Allocate GPU matrix and transfer
  CuBlasGPUMatrix<T> transferToGPU() const;

private:

  size_t rows_;
  size_t columns_;
  std::unique_ptr<T[]> matrix_array_;

  size_t columnMajorIndex(size_t row, size_t column) { return (column * columns_) + row; }
  size_t rowMajorIndex(size_t row, size_t column) { return (row * rows_) + column; }

};


template<class T>
CuBlasHostMatrix<T>::CuBlasHostMatrix(size_t rows, size_t columns) {

  rows_ = rows;
  columns_ = columns;
  matrix_array_ = std::make_unique<T[]>(rows_ * columns_);
  if (not matrix_array_) {

    ExecEnv::log().error("CuBlasHostMatrix::CuBlasHostMatrix; failed to allocate storage of a Matrix({}, {})", rows_, columns_);
    rows_ = 0;
    columns_ = 0;

  }

}

template<class T>
bool CuBlasHostMatrix<T>::transpose() {

  std::unique_ptr<T[]> transpose_ptr = std::make_unique<T[]>(rows_ * columns_);
  if (not transpose_ptr) {

    ExecEnv::log().error("CuBlasHostMatrix::transpose; failed to allocate storage for a tranpose a Matrix({}, {})", columns_, rows_);
    return false;

  }

  for (size_t row_index = 0; row_index < rows_; ++row_index) {

    for (size_t column_index = 0; column_index < columns_; ++column_index) {

      transpose_ptr[columnMajorIndex(row_index, column_index)] = matrix_array_[rowMajorIndex(column_index, row_index)];

    }

  }

  matrix_array_ = std::move(transpose_ptr);
  std::swap(rows_, columns_);

  return true;

}

template<class T>
bool CuBlasHostMatrix<T>::operator==(const CuBlasHostMatrix& matrix) const {

  const T accuracy = 1.0e-08;
  if (rows() != matrix.rows() or columns() != matrix.columns()) {

    ExecEnv::log().warn("CuBlasHostMatrix::operator==; LH Matrix({},{}) != RH Matrix({}, {}), dimensional mismatch",
                        rows(), columns(), matrix.rows(), matrix.columns());
    return  false;

  }

  for (size_t row_index = 0; row_index < rows_; ++row_index) {

    for (size_t column_index = 0; column_index < columns_; ++column_index) {

      if (std::fabs(matrix_array_[rowMajorIndex(row_index, column_index)] - matrix.matrix_array_[rowMajorIndex(column_index, row_index)]) > accuracy) {

        return false;

      }

    }

  }

  return true;

}


template<class T>
CuBlasHostMatrix<T> CuBlasHostMatrix<T>::operator*(const CuBlasHostMatrix<T>& matrix_B) const
{

  if (columns() != matrix_B.rows()) {

    ExecEnv::log().error("CuBlasHostMatrix::operator*; LH Matrix columns {} != RH Matrix rows {}, dimensional mismatch",
                         columns(), matrix_B.rows());
    return CuBlasHostMatrix<T>(0, 0); // return the empty matrix.

  }

  CuBlasHostMatrix<T> result_C (rows(), matrix_B.columns());

  if (result_C.rows() == 0) {

    ExecEnv::log().error("CuBlasHostMatrix::operator*; could not allocate result matrix({}, {})", rows(), matrix_B.columns());
    return CuBlasHostMatrix<T>(0, 0); // return the empty matrix.

  }

  for (size_t i = 0; i < rows(); ++i) {

    for (size_t j = 0; j < matrix_B.columns(); ++j) {
      T sum = 0;

      for (size_t k = 0; k < columns(); ++k) {

        T a = matrix_array_[(i * columns()) + k];
        T b = matrix_array_[(k * matrix_B.columns()) + j];
        sum += a * b;

      }

      result_C.matrix_array_[(i * matrix_B.columns()) + j] = sum;

    }

  }

  return result_C;

}

template<class T>
CuBlasHostMatrix<T> CuBlasHostMatrix<T>::operator+(const CuBlasHostMatrix<T>& matrix_B) const
{

  if (rows() != matrix_B.rows() or columns() != matrix_B.columns()) {

    ExecEnv::log().error("CuBlasHostMatrix::operator+; LH Matrix({},{}) != RH Matrix({},{}), dimensional mismatch",
                         rows(), columns(), matrix_B.rows(), matrix_B.columns());
    return CuBlasHostMatrix<T>(0, 0); // return the empty matrix.

  }

  CuBlasHostMatrix<T> result_C (rows(), columns());

  if (result_C.rows() == 0) {

    ExecEnv::log().error("CuBlasHostMatrix::operator+; could not allocate result matrix({}, {})", rows(), matrix_B.columns());
    return CuBlasHostMatrix<T>(0, 0); // return the empty matrix.

  }

  for (size_t i = 0; i < rows_; ++i) {

    for (size_t j = 0; j < columns_; ++j) {

      T a = matrix_array_[rowMajorIndex(i, j)];
      T b = matrix_array_[rowMajorIndex(i, j)];
      result_C.matrix_array_[rowMajorIndex(i, j)] = a + b;

    }

  }

  return result_C;

}


template<class T>
bool CuBlasHostMatrix<T>::transferToGPU(CuBlasGPUMatrix<T>& gpu_matrix) const {

  size_t memory_size = rows_ * columns_ * sizeof(T);
  checkCudaErrors(cudaMemcpy(reinterpret_cast<void*>(gpu_matrix.data()), reinterpret_cast<void*>(matrix_array_.get()), memory_size, cudaMemcpyHostToDevice));
  return true;

}

template<class T>
CuBlasGPUMatrix<T> CuBlasHostMatrix<T>::transferToGPU() const {

  CuBlasGPUMatrix<T> gpu_matrix(rows_, columns_);
  transferToGPU(gpu_matrix);
  return gpu_matrix;

}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//


class CuBlasSessionImpl {

public:

  CuBlasSessionImpl();
  ~CuBlasSessionImpl();

  bool setTensorCore();

  // transpose(C) = A * B
  // cuBlas expects data in column major (fortran) form
  // A and B are presented as transposes in standard C/C++ row major form
  // Therefore result vector (C) is stored (in row major form) as the transpose of the result of the multiplication.
  // Thus CuBlasGPUMatrix<T>& C_matrix must have dimensions rows = B.column() and columns = A.rows()
  template<class T>
  bool multiplyMatrix(const CuBlasGPUMatrix<T>& A_matrix,
                      const CuBlasGPUMatrix<T>& B_matrix,
                      CuBlasGPUMatrix<T>& C_matrix,
                      T alpha = 1.0,
                      T beta = 0.0);
  // A convenience routine that does all the allocations and conversions.
  template<class T>
  CuBlasHostMatrix<T> multiplyMatrix( const CuBlasHostMatrix<T>& A_matrix,
                                      const CuBlasHostMatrix<T>& B_matrix,
                                      T alpha = 1.0,
                                      T beta = 0.0);

private:

  cublasHandle_t cuda_cublas_handle_;

};


template<class T>
CuBlasHostMatrix<T> CuBlasSessionImpl::multiplyMatrix( const CuBlasHostMatrix<T>& A_matrix,
                                                       const CuBlasHostMatrix<T>& B_matrix,
                                                       T alpha,
                                                       T beta) {

  auto A_gpu_matrix = A_matrix.transferToGPU();
  auto B_gpu_matrix = B_matrix.transferToGPU();
  CuBlasGPUMatrix<T> C_matrix(B_matrix.column(), A_matrix.rows());
  if (not multiplyMatrix<T>(A_matrix, B_matrix, C_matrix, alpha, beta)) {

    // return the empty matrix.
    return CuBlasHostMatrix<T>(0, 0);

  }

  auto C_transpose_matrix = C_matrix.transferToHost();
  auto C_host_matrix = C_transpose_matrix.transpose();

  // Check the dim of C
  if (C_host_matrix.rows() != A_matrix.rows() or C_host_matrix.columns() != B_matrix.columns()) {

    ExecEnv::log().error("CuBlasSessionImpl::multiplyMatrix; 'C' matrix has dim({},{}) should be dim({},{}), the result transpose",
                         C_matrix.rows(), C_matrix.columns(), A_matrix.rows(), B_matrix.columns());
    // return the empty matrix.
    return CuBlasHostMatrix<T>(0, 0);

  }

  return C_host_matrix;

}


} // namespace

#endif //GPU_LIBRARY_KEG_CUDA_CUBLAS_H
