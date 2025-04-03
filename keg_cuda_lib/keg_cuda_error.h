//
// Created by kellerberrin on 1/04/25.
//

#ifndef KEG_CUDA_ERROR_H
#define KEG_CUDA_ERROR_H


#include <cuda.h>
#include <driver_types.h>

#include <string>

namespace kellerberrin::gpu {   //  organization::project level namespace



class CudaErrorCode {

public:

  CudaErrorCode() = delete;
  ~CudaErrorCode() = delete;

  [[nodiscard]] static bool validCudaCode(cudaError cuda_code) { return cuda_code == cudaSuccess; };
  [[nodiscard]] static std::string CudaCodeText(cudaError cuda_code);

private:

};



} //namespace


#endif // KEG_CUDA_ERROR_H
