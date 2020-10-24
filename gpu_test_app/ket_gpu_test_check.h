//
// Created by kellerberrin on 22/10/20.
//

#ifndef KET_GPU_TEST_CHECK_H
#define KET_GPU_TEST_CHECK_H


namespace kellerberrin {   //  organization::project level namespace


  bool doubleMatrixCheck( const dim3& block_size,
                          const dim3& grid_size,
                          size_t& error_count,
                          double* matrix_address_base,
                          size_t iterations);

  bool floatMatrixCheck( const dim3& block_size,
                         const dim3& grid_size,
                         size_t& error_count,
                         float* matrix_address_base,
                         size_t iterations);

}


#endif //KET_GPU_TEST_CHECK_H
