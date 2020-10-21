//
// Created by kellerberrin on 21/10/20.
//

#ifndef KEG_GPU_MATRIX3_H
#define KEG_GPU_MATRIX3_H

#include "keg_gpu_matrix.h"

namespace kellerberrin::gpu {   //  organization::project level namespace

// Implements simple GPU-side indexed matrices to store the repeated results of matrix computations.

template<class T>
class GPUMatrix3 {

public:

  GPUMatrix3() noexcept;
  GPUMatrix3(size_t rows, size_t columns, size_t matrices) noexcept;
  GPUMatrix3(const GPUMatrix<T>& copy) = delete;
  GPUMatrix3(GPUMatrix3<T>&& copy) noexcept;
  ~GPUMatrix3() noexcept = default;

  GPUMatrix3<T> &operator=(GPUMatrix3<T>&& copy) noexcept;
  GPUMatrix3<T> &operator=(const GPUMatrix3<T>& copy) = delete;

  bool reallocateMatrices(size_t rows, size_t columns, size_t matrices);

  bool transferToHost(HostMatrix<T> &host_matrix, size_t matrix_index) const;
  HostMatrix<T> transferToHost(size_t matrix_index) const;

  [[nodiscard]] T* data(size_t matrix_index) const;
  [[nodiscard]] void** getPointerAddress() { return gpu_memory_.getPointerAddress(); };

  [[nodiscard]] size_t rows() const { return rows_; }
  [[nodiscard]] size_t columns() const { return columns_; }
  [[nodiscard]] size_t matrices() const { return matrices_; }

private:

  size_t rows_;
  size_t columns_;
  size_t matrices_;
  GPUMemory gpu_memory_;

};



template<class T>
GPUMatrix3<T>::GPUMatrix3() noexcept {

  rows_ = 0;
  columns_ = 0;
  matrices_ = 0;

}


template<class T>
GPUMatrix3<T>::GPUMatrix3(GPUMatrix3<T>&& copy) noexcept {

  gpu_memory_ = std::move(copy.gpu_memory_);
  rows_ = copy.rows_;
  columns_ = copy.columns_;
  matrices_ = copy.matrices_;
  copy.rows_ = 0;
  copy.columns_ = 0;
  copy.matrices_ = 0;

}

template<class T>
GPUMatrix3<T>::GPUMatrix3(size_t rows, size_t columns, size_t matrices) noexcept  {

  const size_t mem_size_bytes = matrices * rows * columns * sizeof(T);
  if (not gpu_memory_.reallocateMemory(mem_size_bytes)) {

    rows_ = 0;
    columns_ = 0;
    matrices_ = 0;

  } else {

    rows_ = rows;
    columns_ = columns;
    matrices_ = matrices;

  }

}


template<class T>
GPUMatrix3<T>& GPUMatrix3<T>::operator=(GPUMatrix3<T>&& copy) noexcept {

  gpu_memory_ = std::move(copy.gpu_memory_);
  rows_ = copy.rows_;
  columns_ = copy.columns_;
  matrices_ = copy.matrices_;
  copy.device_ptr_ = nullptr;
  copy.rows_ = 0;
  copy.columns_ = 0;
  copy.matrices_ = 0;

  return *this;

}


template<class T>
bool GPUMatrix3<T>::reallocateMatrices(size_t rows, size_t columns, size_t matrices)  {

  const size_t mem_size_bytes = matrices * rows * columns * sizeof(T);
  if (not gpu_memory_.reallocateMemory(mem_size_bytes)) {

    rows_ = 0;
    columns_ = 0;
    matrices_ = 0;
    return false;

  } else {

    rows_ = rows;
    columns_ = columns;
    matrices_ = matrices;

  }

  return true;

}

template<class T>
T* GPUMatrix3<T>::data(size_t matrix_index) const {

  // Check dimensions.
  if (matrix_index >= matrices_) {

    ExecEnv::log().warn("GPUMatrix3::transferToHost; matrix index: {} >= number of GPU matrices: {}", matrix_index, matrices_);
    return reinterpret_cast<T*>(gpu_memory_.memInfo().address());

  }

  size_t mem_offset = sizeof(T) * rows() * columns() * matrix_index;
  return reinterpret_cast<T*>(gpu_memory_.memInfo().address() + mem_offset);

}


template<class T>
bool GPUMatrix3<T>::transferToHost(HostMatrix<T> &host_matrix, size_t matrix_index) const {

  // Check dimensions.
  if (matrix_index >= matrices_) {

    ExecEnv::log().warn("GPUMatrix3::transferToHost; matrix index: {} >= number of GPU matrices: {}", matrix_index, matrices_);
    return false;

  }

  size_t mem_offset = sizeof(T) * rows() * columns() * matrix_index;
  size_t mem_size = sizeof(T) * rows() * columns();
  if (not gpu_memory_.transferSubBlockToHost(mem_offset, mem_size, host_matrix.host_memory_)) {

    ExecEnv::log().warn("GPUMatrix3::transferToHost; GPU Matrix({},{}) cannot transfer to host matrix", rows(), columns());
    return false;

  }

  host_matrix.rows_ = rows();
  host_matrix.columns_ = columns();

  return true;

}

template<class T>
HostMatrix<T> GPUMatrix3<T>::transferToHost(size_t matrix_index) const {

  HostMatrix<T> host_matrix(rows(), columns());
  if (not transferToHost(host_matrix, matrix_index)) {

    HostMatrix<T>(0, 0); // return the null matrix.

  }

  return std::move(host_matrix);

}





} //

#endif //KEG_GPU_3MATRIX_H
