//
// Created by kellerberrin on 6/10/20.
//

#include "ket_gpu_test_app.h"
#include "kel_exec_env_app.h"


int main(int argc, const char **argv) {

  namespace kel = kellerberrin;
  namespace ket = kellerberrin::gpu::test;

  return kel::ExecEnv::runApplication<ket::GPUTestExecEnv>(argc, argv);

}


