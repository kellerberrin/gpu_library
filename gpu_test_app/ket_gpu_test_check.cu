
// Provide logging to the cuda code.
#include "ket_gpu_test_check.h"
#include "kel_exec_env.h"

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

namespace kel = kellerberrin;


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

  checkCudaErrors(cudaMalloc((void **) &dev_error, sizeof(int)));

  checkCudaErrors(cudaMemcpy(dev_error, &host_error, sizeof(int), cudaMemcpyHostToDevice));

  compareFloat<<<grid_size, block_size>>>(matrix_address_base, dev_error, iterations);

  checkCudaErrors(cudaMemcpy(&host_error, dev_error, sizeof(int), cudaMemcpyDeviceToHost));
  error_count = host_error;

  checkCudaErrors(cudaFree(dev_error));

  return true;

}


bool kel::doubleMatrixCheck( const dim3& grid_size,
                             const dim3& block_size,
                             size_t& error_count,
                             double* matrix_address_base,
                             const size_t iterations) {

  int *dev_error{nullptr};
  int host_error{0};

  checkCudaErrors(cudaMalloc((void **) &dev_error, sizeof(int)));
  checkCudaErrors(cudaMemcpy(dev_error, &host_error, sizeof(int), cudaMemcpyHostToDevice));

  compareDouble<<<grid_size, block_size>>>(matrix_address_base, dev_error, iterations);

  checkCudaErrors(cudaMemcpy(&host_error, dev_error, sizeof(int), cudaMemcpyDeviceToHost));
  error_count = host_error;

  checkCudaErrors(cudaFree(dev_error));

  return true;

}


