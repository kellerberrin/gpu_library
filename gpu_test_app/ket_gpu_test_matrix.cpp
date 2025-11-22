//
// Created by kellerberrin on 7/10/20.
//

#include "ket_gpu_test_matrix.h"
#include "ket_gpu_test_check.h"
#include "kel_distribution.h"
#include "keg_cuda_cublas.h"

#include "../kel_lib/kel_exec_env.h"

namespace ket = kellerberrin::gpu::test;
namespace kel = kellerberrin;


ket::GPUMatrixTest::~GPUMatrixTest() {}


void ket::GPUMatrixTest::initialize() {

  context_.bindToThread();

  if (tensor_flag_) {

    cublas_session_.setTensorCore();

  }

  initialize_memory();

}


std::pair<size_t, size_t> ket::GPUMatrixTest::testGPU() {

  compute();
  compare();

  return { iterations_,  error_count_ };

}


void ket::GPUMatrixTest::compute() {


  for (size_t result_index = 0; result_index < iterations_; ++result_index) {

    if (double_flag_) {

      cublas_session_.multiplyMatrix(matrix_Ad_, matrix_Bd_, matrix_Cd_, result_index);

    }
    else {

      cublas_session_.multiplyMatrix(matrix_Af_, matrix_Bf_, matrix_Cf_, result_index);

    } // if double

  } // for iteraions

}


void ket::GPUMatrixTest::compare() {

  size_t errors{0};
  if (double_flag_) {

    kel::doubleMatrixCheck(GRID_XY_, BLOCK_XY_, errors, matrix_Cd_.data(0), iterations_);

  } else {

    kel::floatMatrixCheck(GRID_XY_, BLOCK_XY_, errors, matrix_Cf_.data(0), iterations_);

  }

  error_count_ += errors;

}

void ket::GPUMatrixTest::initialize_memory() {

  // Allocate device memory
  auto [total_memory, free_memory] = device_.memoryInfo();
  auto allocate_device_memory = static_cast<size_t>(static_cast<double>(free_memory) * DEVICE_MEMORY_USAGE_);
  size_t actual_allocation;
  // Initialize the A & B matrices with random numbers.
  RandomEntropySource entropy;
  UniformUnitDistribution random_unit_real;

  if (double_flag_) {

    HostMatrix<double> A(MATRIX_SIZE_, MATRIX_SIZE_);
    HostMatrix<double> B(MATRIX_SIZE_, MATRIX_SIZE_);

    for (size_t row = 0; row < A.rows(); ++row) {

      for (size_t column = 0; column < A.columns(); ++column) {

        A(row, column, random_unit_real.random(entropy.generator()));
        B(row, column, random_unit_real.random(entropy.generator()));

      }

    }

    matrix_Ad_ = A.transferToGPU();
    matrix_Bd_ = B.transferToGPU();
    iterations_ = allocate_device_memory  / (MATRIX_ELEMENTS_ * sizeof(double));
    matrix_Cd_.reallocateMatrices(MATRIX_SIZE_, MATRIX_SIZE_, iterations_);
    actual_allocation = iterations_ * MATRIX_ELEMENTS_ * sizeof(double);

  } else {

    HostMatrix<float> A(MATRIX_SIZE_, MATRIX_SIZE_);
    HostMatrix<float> B(MATRIX_SIZE_, MATRIX_SIZE_);

    for (size_t row = 0; row < A.rows(); ++row) {

      for (size_t column = 0; column < A.columns(); ++column) {

        A(row, column, static_cast<float>(random_unit_real.random(entropy.generator())));
        B(row, column, static_cast<float>(random_unit_real.random(entropy.generator())));

      }

    }

    matrix_Af_ = A.transferToGPU();
    matrix_Bf_ = B.transferToGPU();
    iterations_ = allocate_device_memory  / (MATRIX_ELEMENTS_ * sizeof(float));
    matrix_Cf_.reallocateMatrices(MATRIX_SIZE_, MATRIX_SIZE_, iterations_);
    actual_allocation = iterations_ * MATRIX_ELEMENTS_ * sizeof(float);

  }

  ExecEnv::log().info("GPU {}, {}, total device memory {}MB, available (free) {}MB, allocated for test {}MB",
                      device_.getDeviceIdent(), device_.getDeviceName(),
                      total_memory / (1024 * 1024), free_memory / (1024 * 1024), actual_allocation / (1024 * 1024));

  ExecEnv::log().info("GPU {}, {}, Multiplying square A x B with rows: {} and columns: {}, {}, {}",
                      device_.getDeviceIdent(), device_.getDeviceName(), MATRIX_SIZE_,  MATRIX_SIZE_,
                      (double_flag_ ? "using double precision" : "using single precision (float)"),
                      (tensor_flag_ ? ", using Tensor Cores" : ""));

}



