//
// Created by kellerberrin on 18/10/20.
//

#ifndef KEG_CUDA_CUBLAS_H
#define KEG_CUDA_CUBLAS_H

#include "kel_exec_env.h"
#include "keg_cuda_device.h"
#include "keg_gpu_matrix.h"
#include "keg_gpu_matrix3.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>


#include <string>
#include <memory>


namespace kellerberrin::gpu {   //  organization::project level namespace


class CheckCublasCode {

public:

  CheckCublasCode() = delete;
  ~CheckCublasCode() = delete;

  static bool check(cublasStatus_t status_code, const std::string &module_text = "");

private:

  static std::pair<bool, std::string> checkCublasError(cublasStatus_t status_code);

};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//


class CuBlasSessionImpl {

public:

  CuBlasSessionImpl();
  ~CuBlasSessionImpl();

  bool setTensorCore();

  // transpose(C) = T(B) * T(A)
  // cuBlas expects data in column major (fortran) form
  // A and B are presented as transposes in standard C/C++ row major form
  // Therefore result vector (C) is stored (in row major form) as the transpose of the result of the multiplication.
  // Thus GPUMatrix<T>& C_matrix must have dimensions rows = B.column() and columns = A.rows()
  template<class T>
  bool multiplyMatrix(const GPUMatrix<T>& A_matrix,
                      const GPUMatrix<T>& B_matrix,
                      GPUMatrix3<T>& C_matrix,
                      size_t c_matrix_index = 0,
                      T alpha = 1.0,
                      T beta = 0.0);
  // A convenience routine that does all the allocations and conversions.
  template<class T>
  HostMatrix<T> multiplyMatrix(const HostMatrix<T>& A_matrix,
                               const HostMatrix<T>& B_matrix,
                               T alpha = 1.0,
                               T beta = 0.0);

private:

  cublasHandle_t cuda_cublas_handle_;

};




template<class T>
HostMatrix<T> CuBlasSessionImpl::multiplyMatrix(const HostMatrix<T>& A_matrix,
                                                const HostMatrix<T>& B_matrix,
                                                T alpha,
                                                T beta) {

  auto A_gpu_matrix = A_matrix.transferToGPU();
  auto B_gpu_matrix = B_matrix.transferToGPU();
  // Only allocate 1 result matrix.
  GPUMatrix3<T> C_matrix(B_matrix.column(), A_matrix.rows(), 1);
  if (not multiplyMatrix<T>(A_matrix, B_matrix, C_matrix, 0, alpha, beta)) {

    // return the empty matrix.
    return HostMatrix<T>(0, 0);

  }

  auto C_transpose_matrix = C_matrix.transferToHost(0);
  auto C_host_matrix = C_transpose_matrix.transpose();

  // Check the dim of C
  if (C_host_matrix.rows() != A_matrix.rows() or C_host_matrix.columns() != B_matrix.columns()) {

    ExecEnv::log().error("CuBlasSessionImpl::multiplyMatrix; 'C' matrix has dim({},{}) should be dim({},{}), the result transpose",
                         C_matrix.rows(), C_matrix.columns(), A_matrix.rows(), B_matrix.columns());
    // return the empty matrix.
    return HostMatrix<T>(0, 0);

  }

  return C_host_matrix;

}




} // namespace

#endif //GPU_LIBRARY_KEG_CUDA_CUBLAS_H
