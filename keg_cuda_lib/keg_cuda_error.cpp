//
// Created by kellerberrin on 1/04/25.
//

#include "keg_cuda_error.h"
#include "kel_exec_env.h"
#include <cuda_runtime_api.h>

namespace kel = kellerberrin;
namespace keg = kellerberrin::gpu;


std::string keg::CudaErrorCode::CudaCodeText(cudaError cuda_code) {

  const char* error_name_ptr = cudaGetErrorName(cuda_code);

  if (not error_name_ptr) {

    kel::ExecEnv::log().error("Invalid (unrecognized) error code: {} returned", static_cast<size_t>(cuda_code));
    return "Invalid Error Code";

  }

  const char* error_description_ptr = cudaGetErrorString (cuda_code);

  if (not error_description_ptr) {

    kel::ExecEnv::log().error("Invalid (unrecognized) error code: {} returned", static_cast<size_t>(cuda_code));
    return "Invalid Error Code";

  }

  std::string error_string = std::string(error_name_ptr) + std::string("; ") + std::string(error_description_ptr);

  return error_string;

}

