//
// Created by kellerberrin on 6/10/20.
//

#include "ket_gpu_test_app.h"

namespace ket = kellerberrin::gpu::test;


bool ket::GPUTestExecEnv::parseCommandLine(int argc, char const ** argv) {

  int thisParam = 0;

  // Setup the Logger.
  ExecEnv::createLogger(MODULE_NAME, getArgs().log_file_, getArgs().max_error_count_, getArgs().max_warn_count_);

  std::vector<std::string> args(argv, argv + argc);
  for (size_t i = 1; i < args.size(); ++i) {

    if (argc >= 2 and std::string(argv[i]).find("-d") != std::string::npos) {

      args_.use_double_precision_ = true;
      thisParam++;

    }

    if (argc >= 2 and std::string(argv[i]).find("-tc") != std::string::npos) {

      args_.use_tensor_cores_ = true;
      thisParam++;

    }

  }

  if (argc - thisParam >= 2) {

    args_.test_run_time_ = atoi(argv[1 + thisParam]);

  }

  ExecEnv::log().info("Testing GPU(s) for {} seconds.", args_.test_run_time_);

  return true;

}


