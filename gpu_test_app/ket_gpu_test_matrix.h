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

#include <cuda.h>
#include "cublas_v2.h"

#include "kel_exec_env.h"

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

  constexpr static const size_t MATRIX_SIZE_ = 2048; // Matrix size N
  constexpr static const size_t MATRIX_ELEMENTS_ = MATRIX_SIZE_ * MATRIX_SIZE_; // N x N matrix elements
  // Floating point operations using the naive algorithm (MATRIX_SIZE^3 * 2) for a single matrix multiplication A * B.
  constexpr static const size_t FLOPS_PER_MULTIPLICATION = MATRIX_SIZE_ * MATRIX_SIZE_ * MATRIX_SIZE_ * 2;
  const double DEVICE_MEMORY_USAGE_ = 0.9;  // Approximate proportion of the device memory to allocate.

  const size_t BLOCK_SIZE_ = 32;
  size_t iterations_;

  // Cuda context information.
  CUmodule cuda_module_struct_;
  CUfunction cuda_function_struct_;

  // Matrices.
  CuBlasGPUMatrix<double> matrix_Ad_;
  CuBlasGPUMatrix<double> matrix_Bd_;
  CuBlasGPUMatrix<float> matrix_Af_;
  CuBlasGPUMatrix<float> matrix_Bf_;

  CUdeviceptr cuda_Cdata_ptr_;
  CUdeviceptr cuda_Adata_ptr_;
  CUdeviceptr cuda_Bdata_ptr_;
  CUdeviceptr cuda_faulty_data_ptr_;

  size_t error_count_;
  int *faultyElemsHost_ptr_;

  CuBlasSessionImpl cublas_session_;

  void initialize();
  // .first is the total device memory, .second is the available (free) memory
  std::pair<size_t, size_t> memoryInfo();
  void initialize_memory();
  void initCompareKernel();
  size_t getErrors();
  size_t iterations() const { return iterations_; }
  void compute();
  void compare();

};


} // namespace


#endif //GPU_LIBRARY_GPU_MATRIX_TEST_H
