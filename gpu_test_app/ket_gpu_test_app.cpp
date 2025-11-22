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

  if (argc == 1) {

    args_.help_ = false;
    return true;

  }

  std::vector<std::string> args(argv, argv + argc);
  for (size_t i = 1; i < args.size(); ++i) {

    if (argc >= 2 and std::string(argv[i]).find("-h") != std::string::npos) {

      args_.help_ = true;
      thisParam++;

    }

    if (argc >= 2 and std::string(argv[i]).find("--help") != std::string::npos) {

      args_.help_ = true;
      thisParam++;

    }

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

void ket::GPUTestExecEnv::executeApp() {

  if (args_.help_) {
    ExecEnv::log().info("Usage: '<-h/--help> <-d> <-t> <secs>'");
    ExecEnv::log().info("'<-h/--help> this usage message'");
    ExecEnv::log().info("'<-d> test using double precision (defaults to float)'");
    ExecEnv::log().info("'<-t> test using tensor cores (ignored if not available)'");
    ExecEnv::log().info("'<secs> approximate time to run the test (defaults to 60 seconds)'");
    return;
  }

  startGPUTest(args_.test_run_time_, args_.use_double_precision_, args_.use_tensor_cores_);

}

