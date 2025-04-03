
// Provide logging to the cuda code.
#include "ket_gpu_test_check.h"
#include "kel_exec_env.h"

// CUDA runtime
#include "keg_cuda_error.h"

#include <cuda_runtime.h>


namespace kel = kellerberrin;
namespace keg = kellerberrin::gpu;


__global__ static void compareFloat(float *matrix_base, int *error_count, size_t matrix_count) {

  constexpr const static double FLOAT_ACCURACY = 0.001f;

  size_t matrix_elements = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

  size_t thread_index = (((blockIdx.y * blockDim.y) + threadIdx.y)
                        * (gridDim.x * blockDim.x))
                        + ((blockIdx.x * blockDim.x) + threadIdx.x);

  int element_error = 0;
  for (size_t idx = 1; idx < matrix_count; ++idx) {

    size_t matrix_offset = thread_index + (idx * matrix_elements);
    if (fabsf(matrix_base[thread_index] - matrix_base[matrix_offset]) > FLOAT_ACCURACY) {

      ++element_error;

    }

  }

  atomicAdd(error_count, element_error);

}


__global__ static void compareDouble(double *matrix_base, int *error_count, size_t matrix_count) {

  constexpr const static double DOUBLE_ACCURACY = 0.0000001;

  size_t matrix_elements = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

  size_t thread_index = (((blockIdx.y * blockDim.y) + threadIdx.y)
                         * (gridDim.x * blockDim.x))
                        + ((blockIdx.x * blockDim.x) + threadIdx.x);

  int element_error = 0;
  for (size_t idx = 1; idx < matrix_count; ++idx) {

    size_t matrix_offset = thread_index + (idx * matrix_elements);
    if (fabs(matrix_base[thread_index] - matrix_base[matrix_offset]) > DOUBLE_ACCURACY) {

      ++element_error;

    }

  }

  atomicAdd(error_count, element_error);

}


bool kel::floatMatrixCheck( const dim3& grid_size,
                            const dim3& block_size,
                            size_t& error_count,
                            float* matrix_address_base,
                            const size_t iterations) {

  int *dev_error{nullptr};
  int host_error{0};

  cudaError return_code = cudaMalloc((void **) &dev_error, sizeof(int));
  if (not keg::CudaErrorCode::validCudaCode(return_code)) {

    ExecEnv::log().error("Failed Float Matrix Check, Reason: {}", keg::CudaErrorCode::CudaCodeText(return_code));

  }

  return_code = cudaMemcpy(dev_error, &host_error, sizeof(int), cudaMemcpyHostToDevice);
  if (not keg::CudaErrorCode::validCudaCode(return_code)) {

    ExecEnv::log().error("Failed Float Matrix Check, Reason: {}", keg::CudaErrorCode::CudaCodeText(return_code));

  }

  compareFloat<<<grid_size, block_size>>>(matrix_address_base, dev_error, iterations);

  return_code = cudaMemcpy(&host_error, dev_error, sizeof(int), cudaMemcpyDeviceToHost);
  if (not keg::CudaErrorCode::validCudaCode(return_code)) {

    ExecEnv::log().error("Failed Float Matrix Check, Reason: {}", keg::CudaErrorCode::CudaCodeText(return_code));

  }

  error_count = host_error;

  return_code = cudaFree(dev_error);
  if (not keg::CudaErrorCode::validCudaCode(return_code)) {

    ExecEnv::log().error("Failed Float Matrix Check, Reason: {}", keg::CudaErrorCode::CudaCodeText(return_code));

  }

  return true;

}


bool kel::doubleMatrixCheck( const dim3& grid_size,
                             const dim3& block_size,
                             size_t& error_count,
                             double* matrix_address_base,
                             const size_t iterations) {

  int *dev_error{nullptr};
  int host_error{0};

  cudaError return_code = cudaMalloc((void **) &dev_error, sizeof(int));
  if (not keg::CudaErrorCode::validCudaCode(return_code)) {

    ExecEnv::log().error("Failed Double Matrix Check, Reason: {}", keg::CudaErrorCode::CudaCodeText(return_code));

  }

  return_code =  cudaMemcpy(dev_error, &host_error, sizeof(int), cudaMemcpyHostToDevice);
  if (not keg::CudaErrorCode::validCudaCode(return_code)) {

    ExecEnv::log().error("Failed Double Matrix Check, Reason: {}", keg::CudaErrorCode::CudaCodeText(return_code));

  }

  compareDouble<<<grid_size, block_size>>>(matrix_address_base, dev_error, iterations);

  return_code = cudaMemcpy(&host_error, dev_error, sizeof(int), cudaMemcpyDeviceToHost);
  if (not keg::CudaErrorCode::validCudaCode(return_code)) {

    ExecEnv::log().error("Failed Double Matrix Check, Reason: {}", keg::CudaErrorCode::CudaCodeText(return_code));

  }
  error_count = host_error;

  return_code = cudaFree(dev_error);
  if (not keg::CudaErrorCode::validCudaCode(return_code)) {

    ExecEnv::log().error("Failed Double Matrix Check, Reason: {}", keg::CudaErrorCode::CudaCodeText(return_code));

  }

  return true;

}


