//
// Created by kellerberrin on 20/10/20.
//

#ifndef KEG_CUDA_CUBLAS_MATRIX_H
#define KEG_CUDA_CUBLAS_MATRIX_H


#include "kel_exec_env.h"
#include "keg_cuda_device.h"
#include "keg_gpu_mem.h"


#include <string>
#include <memory>
#include <cmath>


namespace kellerberrin::gpu {   //  organization::project level namespace


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// A simple matrix allocated on the GPU device.
template<class T> class HostMatrix; //forward.

template<class T>
class GPUMatrix {

  friend HostMatrix<T>;

public:

  GPUMatrix() noexcept;
  GPUMatrix(size_t Rows, size_t columns) noexcept;
  GPUMatrix(const GPUMatrix<T>& copy) = delete;
  GPUMatrix(GPUMatrix<T>&& copy) noexcept;
  ~GPUMatrix() noexcept = default;

  GPUMatrix<T> &operator=(GPUMatrix<T>&& copy);
  GPUMatrix<T> &operator=(const GPUMatrix<T>& copy) = delete;

  bool transferToHost(HostMatrix<T> &host_matrix) const;

  // Create the host matrix and transfer.
  HostMatrix<T> transferToHost() const;

  [[nodiscard]] size_t rows() const { return rows_; }

  [[nodiscard]] size_t columns() const { return columns_; }

  [[nodiscard]] T* data() const { return reinterpret_cast<T*>(gpu_memory_.memInfo().address()); }

private:

  GPUMemory gpu_memory_;
  size_t rows_;
  size_t columns_;

};


template<class T>
GPUMatrix<T>::GPUMatrix() noexcept {

  rows_ = 0;
  columns_ = 0;

}


template<class T>
GPUMatrix<T>::GPUMatrix(GPUMatrix<T>&& copy) noexcept {

  gpu_memory_ = std::move(copy.gpu_memory_);
  rows_ = copy.rows_;
  columns_ = copy.columns_;
  copy.rows_ = 0;
  copy.columns_ = 0;

}

template<class T>
GPUMatrix<T>::GPUMatrix(size_t rows, size_t columns) noexcept  {

  const size_t mem_size_bytes = rows * columns * sizeof(T);
  if (not gpu_memory_.reallocateMemory(mem_size_bytes)) {

    rows_ = 0;
    columns_ = 0;

  } else {

    rows_ = rows;
    columns_ = columns;

  }

}


template<class T>
GPUMatrix<T>& GPUMatrix<T>::operator=(GPUMatrix<T>&& copy) {

  gpu_memory_ = std::move(copy.gpu_memory_);
  rows_ = copy.rows_;
  columns_ = copy.columns_;
  copy.rows_ = 0;
  copy.columns_ = 0;

  return *this;

}



template<class T>
bool GPUMatrix<T>::transferToHost(HostMatrix<T> &host_matrix) const {

  // Check dimensions.
  if (not gpu_memory_.transferToHost(host_matrix.host_memory_)) {

    ExecEnv::log().warn("GPUMatrix::transferToHost; GPU Matrix({},{}) cannot transfer to host matrix", rows(), columns());
    return false;

  }
  host_matrix.rows_ = rows();
  host_matrix.columns_ = columns();

  return true;

}

template<class T>
HostMatrix<T> GPUMatrix<T>::transferToHost() const {

  HostMatrix<T> host_matrix(rows(), columns());
  if (not transferToHost(host_matrix)) {

    HostMatrix<T>(0, 0); // return the null matrix.

  }

  return std::move(host_matrix);

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// A simple host matrix malloced on the CPU.

template<class T>
class HostMatrix {

 friend GPUMatrix<T>;

public:

  HostMatrix(size_t Rows, size_t columns);
  ~HostMatrix() = default;

  // All elements are in zero offset row major (C/C++) format.
  T operator()(size_t row, size_t column) const;
  void operator()(size_t row, size_t column, const T value);

  // Convert to column major format.
  bool transpose();

  [[nodiscard]] size_t rows() const { return rows_; }

  [[nodiscard]] size_t columns() const { return columns_; }

  [[nodiscard]] T *data() const { return reinterpret_cast<T*>(matrix_array_.memInfo().address()); }

  // Implemented on the CPU.
  [[nodiscard]] bool operator==(const HostMatrix &matrix) const;

  HostMatrix<T> operator*(const HostMatrix<T> &B) const;

  HostMatrix<T> operator+(const HostMatrix<T> &B) const;

  // Transfer to GPU device.
  bool transferToGPU(GPUMatrix<T> &gpu_matrix) const;

  // Allocate GPU matrix and transfer
  GPUMatrix<T> transferToGPU() const;

private:

  size_t rows_;
  size_t columns_;
  HostMemory matrix_array_;

  size_t columnMajorIndex(size_t row, size_t column) { return (column * columns_) + row; }

  size_t rowMajorIndex(size_t row, size_t column) { return (row * rows_) + column; }

};


template<class T>
HostMatrix<T>::HostMatrix(size_t rows, size_t columns) {

  rows_ = rows;
  columns_ = columns;
  size_t matrix_size = sizeof(T) * rows * columns;
  if (not matrix_array_.reallocateMemory(matrix_size)) {

    ExecEnv::log().error("HostMatrix::HostMatrix; failed to allocate storage of a Matrix({}, {})", rows_, columns_);
    rows_ = 0;
    columns_ = 0;

  }

}

// All elements are in zero offset row major (C/C++) format.
template<class T>
T HostMatrix<T>::operator()(size_t row, size_t column) const {

  if (row >= rows_) {

    ExecEnv::log().error("HostMatrix<T>::operator(row, column); row: {} index is invalid, matrix rows: {}", row, rows_);
    return 0.0;

  }
  if (column >= columns_) {

    ExecEnv::log().error("HostMatrix<T>::operator(row, column); column: {} index is invalid, matrix columns: {}", column, columns_);
    return 0.0;

  }

  return data()[rowMajorIndex(row, column)];

}

template<class T>
void HostMatrix<T>::operator()(size_t row, size_t column, const T value) {

  if (row >= rows_) {

    ExecEnv::log().error("HostMatrix<T>::operator(row, column, value); row: {} index is invalid, matrix rows: {}", row, rows_);
    return;

  }
  if (column >= columns_) {

    ExecEnv::log().error("HostMatrix<T>::operator(row, column, value); column: {} index is invalid, matrix columns: {}", column, columns_);
    return;

  }

  data()[rowMajorIndex(row, column)] = value;

}


template<class T>
bool HostMatrix<T>::transpose() {

  size_t matrix_size = sizeof(T) * rows() * columns();
  HostMemory matrix_transpose;
  if (not matrix_transpose.reallocateMemory(matrix_size)) {

    ExecEnv::log().error("HostMatrix::transpose; failed to allocate storage for a tranpose a Matrix({}, {})", columns_, rows_);
    return false;

  }

  T* transpose_address = data();
  T* matrix_address = static_cast<T*>(matrix_transpose.memInfo().address());

  for (size_t row_index = 0; row_index < rows_; ++row_index) {

    for (size_t column_index = 0; column_index < columns_; ++column_index) {

      transpose_address[columnMajorIndex(row_index, column_index)] = matrix_address[rowMajorIndex(column_index, row_index)];

    }

  }

  matrix_array_ = std::move(matrix_transpose);
  std::swap(rows_, columns_);

  return true;

}

template<class T>
bool HostMatrix<T>::operator==(const HostMatrix &matrix) const {

  const T accuracy = 1.0e-08;
  if (rows() != matrix.rows() or columns() != matrix.columns()) {

    ExecEnv::log().warn("HostMatrix::operator==; LH Matrix({},{}) != RH Matrix({}, {}), dimensional mismatch",
                        rows(), columns(), matrix.rows(), matrix.columns());
    return false;

  }

  for (size_t row_index = 0; row_index < rows_; ++row_index) {

    for (size_t column_index = 0; column_index < columns_; ++column_index) {

      if (std::fabs(data()[rowMajorIndex(row_index, column_index)] - matrix.data()[rowMajorIndex(column_index, row_index)]) > accuracy) {

        return false;

      }

    }

  }

  return true;

}


template<class T>
HostMatrix<T> HostMatrix<T>::operator*(const HostMatrix<T> &matrix_B) const {

  if (columns() != matrix_B.rows()) {

    ExecEnv::log().error("HostMatrix::operator*; LH Matrix columns {} != RH Matrix rows {}, dimensional mismatch",
                         columns(), matrix_B.rows());
    return HostMatrix<T>(0, 0); // return the empty matrix.

  }

  HostMatrix<T> result_C(rows(), matrix_B.columns());

  if (result_C.rows() == 0) {

    ExecEnv::log().error("HostMatrix::operator*; could not allocate result matrix({}, {})", rows(), matrix_B.columns());
    return HostMatrix<T>(0, 0); // return the empty matrix.

  }

  for (size_t i = 0; i < rows(); ++i) {

    for (size_t j = 0; j < matrix_B.columns(); ++j) {
      T sum = 0;

      for (size_t k = 0; k < columns(); ++k) {

        T a = data()[(i * columns()) + k];
        T b = matrix_B.data()[(k * matrix_B.columns()) + j];
        sum += a * b;

      }

      result_C.data()[(i * matrix_B.columns()) + j] = sum;

    }

  }

  return result_C;

}

template<class T>
HostMatrix<T> HostMatrix<T>::operator+(const HostMatrix<T> &matrix_B) const {

  if (rows() != matrix_B.rows() or columns() != matrix_B.columns()) {

    ExecEnv::log().error("HostMatrix::operator+; LH Matrix({},{}) != RH Matrix({},{}), dimensional mismatch",
                         rows(), columns(), matrix_B.rows(), matrix_B.columns());
    return HostMatrix<T>(0, 0); // return the empty matrix.

  }

  HostMatrix<T> result_C(rows(), columns());

  if (result_C.rows() == 0) {

    ExecEnv::log().error("HostMatrix::operator+; could not allocate result matrix({}, {})", rows(), matrix_B.columns());
    return HostMatrix<T>(0, 0); // return the empty matrix.

  }

  for (size_t i = 0; i < rows_; ++i) {

    for (size_t j = 0; j < columns_; ++j) {

      T a = data()[rowMajorIndex(i, j)];
      T b = matrix_B.data()[rowMajorIndex(i, j)];
      result_C.matrix_array_[rowMajorIndex(i, j)] = a + b;

    }

  }

  return result_C;

}


template<class T>
bool HostMatrix<T>::transferToGPU(GPUMatrix<T> &gpu_matrix) const {

  matrix_array_.transferToGPU(gpu_matrix.gpu_memory_);
  gpu_matrix.rows_ = rows_;
  gpu_matrix.columns_ = columns_;
  return true;

}

template<class T>
GPUMatrix<T> HostMatrix<T>::transferToGPU() const {

  GPUMatrix<T> gpu_matrix(rows_, columns_);
  transferToGPU(gpu_matrix);
  return gpu_matrix;

}


} //namespace

#endif //GPU_LIBRARY_KEG_CUDA_CUBLAS_MATRIX_H
