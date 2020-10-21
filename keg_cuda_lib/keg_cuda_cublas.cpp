//
// Created by kellerberrin on 18/10/20.
//

#include "keg_cuda_cublas.h"

#include "kel_exec_env.h"

#include <map>

namespace kel = kellerberrin;
namespace keg = kellerberrin::gpu;



std::pair<bool, std::string> keg::CheckCublasCode::checkCublasError(cublasStatus_t status_code) {

  static std::map<cublasStatus_t, std::pair<std::string, std::string>> error_description = {
      { CUBLAS_STATUS_SUCCESS, { "CUBLAS_STATUS_SUCCESS", "The operation was successful."}},
      { CUBLAS_STATUS_NOT_INITIALIZED, { "CUBLAS_STATUS_NOT_INITIALIZED", "The cuBLAS library was not initialized."}},
      { CUBLAS_STATUS_ALLOC_FAILED, { "CUBLAS_STATUS_ALLOC_FAILED", "Resource allocation failed inside the cuBLAS library."}},
      { CUBLAS_STATUS_INVALID_VALUE, { "CUBLAS_STATUS_INVALID_VALUE", "An unsupported value or parameter was passed to the function."}},
      { CUBLAS_STATUS_ARCH_MISMATCH, { "CUBLAS_STATUS_ARCH_MISMATCH",
                                   "The function requires a feature absent from the device architecture (no double precision)."}},
      { CUBLAS_STATUS_MAPPING_ERROR, { "CUBLAS_STATUS_MAPPING_ERROR", "An access to GPU memory space failed (failure to bind a texture)."}},
      { CUBLAS_STATUS_EXECUTION_FAILED, { "CUBLAS_STATUS_EXECUTION_FAILED", "The GPU program failed to execute."}},
      { CUBLAS_STATUS_INTERNAL_ERROR, { "CUBLAS_STATUS_INTERNAL_ERROR", "An internal cuBLAS operation failed."}},
      { CUBLAS_STATUS_NOT_SUPPORTED, { "CUBLAS_STATUS_NOT_SUPPORTED", "The functionality requested is not supported" }},
      { CUBLAS_STATUS_LICENSE_ERROR, { "CUBLAS_STATUS_LICENSE_ERROR","The functionality requested is not licensed or permitted"}}};

  if (status_code == CUBLAS_STATUS_SUCCESS) {

    return { true, "The Cublas operation was successful."};

  }

  auto error_record = error_description.find(status_code);
  if (error_record == error_description.end()) {

    kel::ExecEnv::log().error("CheckCode::checkCublasError; invalid (unrecognized) error code: {} returned", static_cast<size_t>(status_code));
    return { false, "CheckCode::checkCublasError; invalid (unrecognized) error code returned"};

  }

  auto& [cublas_code, cublas_error] = *error_record;
  std::string error_string = std::string(cublas_error.first) + std::string("; ") + std::string(cublas_error.second);

  return { false, error_string };

}

bool keg::CheckCublasCode::check(cublasStatus_t cublas_status_code, const std::string& module_text) {

  auto text_return_code = checkCublasError(cublas_status_code);

  if (not text_return_code.first) {

    if (not module_text.empty()) {

      kel::ExecEnv::log().error("cuBlas module: {} failed; reason: '{}'", module_text, text_return_code.second);

    } else {

      kel::ExecEnv::log().error("cuBlas API call failed; reason: '{}'", text_return_code.second);

    }

    return false;

  }

  return true;

}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////



keg::CuBlasSessionImpl::CuBlasSessionImpl() {

  CheckCublasCode::check(cublasCreate(&cuda_cublas_handle_), "cublasCreate");

}

keg::CuBlasSessionImpl::~CuBlasSessionImpl() {

  CheckCublasCode::check(cublasDestroy(cuda_cublas_handle_),"cublasDestroy");

}

bool keg::CuBlasSessionImpl::setTensorCore() {

  return CheckCublasCode::check(cublasSetMathMode(cuda_cublas_handle_, CUBLAS_TENSOR_OP_MATH));

}



// double precision.
template<>
bool keg::CuBlasSessionImpl::multiplyMatrix<double>(const GPUMatrix<double>& A_matrix,
                                                    const GPUMatrix<double>& B_matrix,
                                                    GPUMatrix3<double>& C_matrix,
                                                    size_t c_matrix_index,
                                                    double alpha,
                                                    double beta) {

  // Check dimensions
  if (A_matrix.rows() == 0 or A_matrix.columns() == 0) {

    ExecEnv::log().error("CuBlasSessionImpl::multiplyMatrix; 'A' matrix is zero sized");
    return false;

  }

  if (B_matrix.rows() == 0 or B_matrix.columns() == 0) {

    ExecEnv::log().error("CuBlasSessionImpl::multiplyMatrix; 'B' matrix is zero sized");
    return false;

  }

  if (C_matrix.rows() != B_matrix.columns() or C_matrix.columns() != A_matrix.rows()) {

    ExecEnv::log().error("CuBlasSessionImpl::multiplyMatrix; 'C' matrix has dim({},{}) should be dim({},{}), the result transpose",
                         C_matrix.rows(), C_matrix.columns(), B_matrix.columns(), A_matrix.rows());
    return false;

  }

  // Note cublas is column major
  // Assume the device vector are row major
  // Present them as Transpose(B) * Transpose(A) = Transpose(C)
  // Thus C should be dim(B.column(), A.row())
  return CheckCublasCode::check(cublasDgemm(cuda_cublas_handle_,
                                            CUBLAS_OP_N,     // No transpose or conjugate
                                            CUBLAS_OP_N,     // No transpose or conjugate
                                            B_matrix.columns(),
                                            A_matrix.rows(),
                                            A_matrix.columns(),
                                            &alpha,
                                            B_matrix.data(),
                                            B_matrix.columns(),
                                            A_matrix.data(),
                                            A_matrix.columns(),
                                            &beta,
                                            C_matrix.data(c_matrix_index),
                                            C_matrix.columns()), "cublasDgemm");

}





// single precision.
template<>
bool keg::CuBlasSessionImpl::multiplyMatrix<float>(const GPUMatrix<float>& A_matrix,
                                                   const GPUMatrix<float>& B_matrix,
                                                   GPUMatrix3<float>& C_matrix,
                                                   size_t c_matrix_index,
                                                   float alpha,
                                                   float beta) {

  // Check dimensions
  if (A_matrix.rows() == 0 or A_matrix.columns() == 0) {

    ExecEnv::log().error("CuBlasSessionImpl::multiplyMatrix; 'A' matrix is zero sized");
    return false;

  }

  if (B_matrix.rows() == 0 or B_matrix.columns() == 0) {

    ExecEnv::log().error("CuBlasSessionImpl::multiplyMatrix; 'B' matrix is zero sized");
    return false;

  }

  if (C_matrix.rows() != B_matrix.columns() or C_matrix.columns() != A_matrix.rows()) {

    ExecEnv::log().error("CuBlasSessionImpl::multiplyMatrix; 'C' matrix has dim({},{}) should be dim({},{}), the result transpose",
                         C_matrix.rows(), C_matrix.columns(), B_matrix.columns(), A_matrix.rows());
    return false;

  }

  // Note cublas is column major
  // Assume the device vector are row major
  // Present them as Transpose(B) * Transpose(A) = Transpose(C)
  // Thus C should be dim(B.column(), A.row())
  return CheckCublasCode::check(cublasSgemm(cuda_cublas_handle_,
                                            CUBLAS_OP_N,     // No transpose or conjugate
                                            CUBLAS_OP_N,     // No transpose or conjugate
                                            B_matrix.columns(),
                                            A_matrix.rows(),
                                            A_matrix.columns(),
                                            &alpha,
                                            B_matrix.data(),
                                            B_matrix.columns(),
                                            A_matrix.data(),
                                            A_matrix.columns(),
                                            &beta,
                                            C_matrix.data(c_matrix_index),
                                            C_matrix.columns()), "cublasSgemm");

}



