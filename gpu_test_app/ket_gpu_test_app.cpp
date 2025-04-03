//
// Created by kellerberrin on 6/10/20.
//

#include <iostream>
#include "ket_gpu_test_app.h"

namespace kel = kellerberrin;
namespace ket = kellerberrin::gpu::test;


std::unique_ptr<kel::ExecEnvLogger> ket::GPUTestExecEnv::createLogger() {

  // Setup the Logger.
  return ExecEnv::createLogger(MODULE_NAME, getArgs().log_file_, getArgs().max_error_count_, getArgs().max_warn_count_);

}


bool ket::GPUTestExecEnv::parseCommandLine(int argc, char const ** argv) {

  int thisParam = 0;

  std::cerr <<  "Usage: " << "<-d> (double) <-t> (tensor cores) <120> (seconds - default 60)" << std::endl;

  std::vector<std::string> args(argv, argv + argc);
  for (size_t i = 1; i < args.size(); ++i) {

    if (argc >= 2 and std::string(argv[i]).find("-d") != std::string::npos) {

      args_.use_double_precision_ = true;
      thisParam++;

    }

    if (argc >= 2 and std::string(argv[i]).find("-t") != std::string::npos) {

      args_.use_tensor_cores_ = true;
      thisParam++;

    }

  }

  if (argc - thisParam >= 2) {

    args_.test_run_time_ = atoi(argv[1 + thisParam]);

  }


  return true;

}


