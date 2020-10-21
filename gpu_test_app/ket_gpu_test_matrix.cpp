//
// Created by kellerberrin on 7/10/20.
//

#include "ket_gpu_test_matrix.h"
#include "kel_distribution.h"
#include "keg_cuda_device.h"
#include "keg_cuda_cublas.h"


namespace ket = kellerberrin::gpu::test;




ket::GPUMatrixTest::~GPUMatrixTest() {

  CheckCode::check(cuMemFreeHost(faultyElemsHost_ptr_));

}


void ket::GPUMatrixTest::initialize() {

  context_.bindToThread();

  if (tensor_flag_) {

    cublas_session_.setTensorCore();

  }

  CheckCode::check(cuMemAllocHost((void **) &faultyElemsHost_ptr_, sizeof(int)));
  error_count_ = 0;

  initialize_memory();

}


size_t ket::GPUMatrixTest::getErrors() {

  if (*faultyElemsHost_ptr_ > 0) {

    error_count_ += *faultyElemsHost_ptr_;

  }
  size_t tempErrs = error_count_;
  error_count_ = 0;

  return tempErrs;

}


std::pair<size_t, size_t> ket::GPUMatrixTest::testGPU() {

  compute();
  compare();

  return { iterations(),  getErrors() };

}


std::pair<size_t, size_t>  ket::GPUMatrixTest::memoryInfo() {

  size_t total_memory, free_memory;
  CheckCode::check(cuMemGetInfo(&free_memory, &total_memory));

  return { total_memory, free_memory};

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

  CheckCode::check(cuMemsetD32Async(cuda_faulty_data_ptr_, 0, 1, 0), "memset");
//  CheckCode::check(cuLaunchGridAsync(cuda_function_struct_, MATRIX_SIZE_ / BLOCK_SIZE_, MATRIX_SIZE_ / BLOCK_SIZE_, 0), "cuLaunchGridAsync");
//  CheckCode::check(cuMemcpyDtoHAsync(faultyElemsHost_ptr_, cuda_faulty_data_ptr_, sizeof(int), 0), "cuMemcpyDtoHAsync");

}

void ket::GPUMatrixTest::initialize_memory() {

  // Allocate device memory
  auto [total_memory, free_memory] = memoryInfo();
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

  initCompareKernel();

}


void ket::GPUMatrixTest::initCompareKernel() {

  const char *kernelFile = "cuda_kernel.ptx";

  {

    std::ifstream f(kernelFile);
    CheckCode::check(f.good() ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND, std::string("couldn't find file \"") + kernelFile + "\" from working directory");

  }

  CheckCode::check(cuModuleLoad(&cuda_module_struct_, kernelFile), "load module");
  CheckCode::check(cuModuleGetFunction(&cuda_function_struct_, cuda_module_struct_,
                                 double_flag_ ? "compareD" : "compare"), "get func");

  CheckCode::check(cuFuncSetCacheConfig(cuda_function_struct_, CU_FUNC_CACHE_PREFER_L1), "L1 config");

  if (double_flag_) {

    CheckCode::check(cuParamSetSize(cuda_function_struct_, __alignof(double *) + __alignof(int *) + __alignof(size_t)), "set param size");
    cuda_Cdata_ptr_ = reinterpret_cast<CUdeviceptr>(*matrix_Cd_.getPointerAddress());
    CheckCode::check(cuParamSetv(cuda_function_struct_, 0, &cuda_Cdata_ptr_, sizeof(double *)), "set param");
//    CheckCode::check(cuParamSetv(cuda_function_struct_, 0, matrix_Cd_.getPointerAddress(), sizeof(double *)), "set param");
    CheckCode::check(cuParamSetv(cuda_function_struct_, __alignof(double *), &cuda_faulty_data_ptr_, sizeof(double *)), "set param");
    CheckCode::check(cuParamSetv(cuda_function_struct_, __alignof(double *) + __alignof(int *), &iterations_, sizeof(size_t)), "set param");

  } else {

    CheckCode::check(cuParamSetSize(cuda_function_struct_, __alignof(float *) + __alignof(int *) + __alignof(size_t)), "set param size");
    CheckCode::check(cuParamSetv(cuda_function_struct_, 0, &cuda_Cdata_ptr_, sizeof(float *)), "set param");
    CheckCode::check(cuParamSetv(cuda_function_struct_, __alignof(float *), &cuda_faulty_data_ptr_, sizeof(float *)), "set param");
    CheckCode::check(cuParamSetv(cuda_function_struct_, __alignof(float *) + __alignof(int *), &iterations_, sizeof(size_t)), "set param");

  }

  CheckCode::check(cuFuncSetBlockShape(cuda_function_struct_, BLOCK_SIZE_, BLOCK_SIZE_, 1), "set block size");

}


