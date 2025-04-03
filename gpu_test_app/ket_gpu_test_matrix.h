//
// Created by kellerberrin on 7/10/20.
//

#ifndef GPU_MATRIX_TEST_H
#define GPU_MATRIX_TEST_H

#include <string>
#include <map>
#include <vector>
#include <fstream>

#include "keg_gpu_device.h"
#include "keg_cuda_cublas.h"

namespace kellerberrin::gpu::test {   //  organization::project level namespace


// Pure virtual class for different GPU tests.
class GPUTest {

public:

  GPUTest() = default;
  virtual ~GPUTest() = default;

  virtual double flopsPerOp() = 0;
  // Operations performed .first, operational errors .second.
  virtual std::pair<size_t, size_t> testGPU() = 0;

};


// Test the GPU using cublas matrix multiplication.
class GPUMatrixTest : public GPUTest {

public:

  GPUMatrixTest(DriverDevice device, bool double_flag, bool tensor_flag) :
      device_(device), context_(device), double_flag_(double_flag), tensor_flag_(tensor_flag) {

    initialize();

  }

  ~GPUMatrixTest() override;

  double flopsPerOp() override { return static_cast<double>(FLOPS_PER_MULTIPLICATION); }


  // Operations performed .first, operational errors .second.
  std::pair<size_t, size_t> testGPU() override;

private:

  // Input parameters
  DriverDevice device_; // Identifies which Cuda device
  ThreadContext context_; // The thread Cuda context.
  bool double_flag_;
  bool tensor_flag_;

  // Matrix size.
  constexpr static const size_t MATRIX_SIZE_ = 2048; // Matrix size N
  constexpr static const size_t MATRIX_ELEMENTS_ = MATRIX_SIZE_ * MATRIX_SIZE_; // N x N matrix elements
  // Floating point operations using the naive algorithm (MATRIX_SIZE^3 * 2) for a single matrix multiplication A * B.
  constexpr static const size_t FLOPS_PER_MULTIPLICATION = MATRIX_SIZE_ * MATRIX_SIZE_ * MATRIX_SIZE_ * 2;

  // Kernel launch parameters.
  constexpr static const size_t BLOCK_SIZE_ = 16;
  constexpr static const dim3 BLOCK_XY_{BLOCK_SIZE_, BLOCK_SIZE_}; // 256 threads per block.
  constexpr static const size_t GRID_SIZE_ = ((MATRIX_SIZE_ + BLOCK_SIZE_ - 1) / BLOCK_SIZE_);
  constexpr static const dim3 GRID_XY_{GRID_SIZE_, GRID_SIZE_};

  // Memory usage.
  constexpr static const double DEVICE_MEMORY_USAGE_ = 0.9;  // Approximate proportion of the device memory to allocate.
  size_t iterations_;

  // Double matrices.
  GPUMatrix<double> matrix_Ad_;
  GPUMatrix<double> matrix_Bd_;
  GPUMatrix3<double> matrix_Cd_;
  // Float matrices.
  GPUMatrix<float> matrix_Af_;
  GPUMatrix<float> matrix_Bf_;
  GPUMatrix3<float> matrix_Cf_;

  // Matrix mult and error count.
  size_t error_count_{0};
  CuBlasSessionImpl cublas_session_;

  void initialize();
  void initialize_memory();
  void compute();
  void compare();

};


} // namespace


#endif //GPU_MATRIX_TEST_H
