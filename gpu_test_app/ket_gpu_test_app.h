//
// Created by kellerberrin on 6/10/20.
//

#ifndef KET_GPU_TEST_APP_H
#define KET_GPU_TEST_APP_H

#include "kel_exec_env.h"
#include <string>

namespace kellerberrin::gpu::test {   //  organization::project level namespace

// Holds the application parameters.
struct CmdLineArgs {

  std::string work_directory_{"./"};
  std::string log_file_{"gpu_test.log"};
  size_t test_run_time_{60};
  bool use_double_precision_{false};
  bool use_tensor_cores_{false};
  int max_error_count_{1000};
  int max_warn_count_{1000};
  bool verbose_{false};
  bool help_{false};

};

void startGPUTest(size_t run_length, bool use_double_precision, bool use_tensor_cores);

// The Runtime environment.
class GPUTestExecEnv {

public:

  GPUTestExecEnv()=default;
  ~GPUTestExecEnv()=default;

  [[nodiscard]] static const CmdLineArgs& getArgs() { return args_; }

  // The following 4 static members are required for all applications.
  inline static constexpr const char* VERSION = "0.1";
  inline static constexpr const char* MODULE_NAME = "GPU Test";
  // Logger is active when executeApp() is called.
  static void executeApp();
  // Logger is inactive when these functions are called.
  [[nodiscard]] static bool parseCommandLine(int argc, char const ** argv);  // Parse command line arguments.
  [[nodiscard]] static std::unique_ptr<ExecEnvLogger> createLogger(); // Create application logger.

private:

  inline static CmdLineArgs args_;


};


} //  end namespace



#endif //KEC_GPU_APP_H
