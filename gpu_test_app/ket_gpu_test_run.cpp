/* 
 * Copyright (c) 2016, Ville Timonen
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 */


#include "ket_gpu_test_matrix.h"
#include "keg_gpu_device_info.h"
#include "keg_gpu_device.h"


#include <vector>
#include <memory>
#include <chrono>


namespace kellerberrin::gpu::test {   //  organization::project level namespace


void GPUTestThread(bool* test_running, DriverDevice device, bool use_double_precision, bool use_tensor_cores) {

  std::unique_ptr<GPUTest> test_gpu_ptr(std::make_unique<GPUMatrixTest>(device, use_double_precision, use_tensor_cores));
  DeviceInformation device_information;
  // The device index is retrieved using the formatted device UUID.
  size_t device_info_index = device_information.getIndexUsingUUID(device.getDeviceFormattedUUID());
  size_t device_index = device.getDeviceIdent();

  size_t iteration_sum{0};
  size_t error_sum{0};
  const double flops_per_op = test_gpu_ptr->flopsPerOp();

  std::unique_ptr<GPUEvent> gpu_event_ptr(std::make_unique<GPUEvent>());

  std::chrono::steady_clock::time_point begin_test = std::chrono::steady_clock::now();
  do { // The work loop

    std::chrono::steady_clock::time_point begin_iteration = std::chrono::steady_clock::now();
    auto [iterations, errors] = test_gpu_ptr->testGPU();

    gpu_event_ptr->record();
    while (not gpu_event_ptr->pollOnRecord()) usleep(1000);
    std::chrono::steady_clock::time_point end_iteration = std::chrono::steady_clock::now();

    iteration_sum += iterations;
    error_sum += errors;
    size_t nano_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end_iteration - begin_iteration).count();

    double gflops_sec = (flops_per_op * static_cast<double>(iterations)) / static_cast<double>(nano_seconds);
    ExecEnv::log().info("GPU {}, {}, Iterations: {}, Errors: {}, GFLOPs: {}",
                        device_index, device.getDeviceName(), iterations, error_sum, gflops_sec);
    auto memory = device_information.getMemoryUsage(device_info_index);
    auto utilization = device_information.getGPUUtilization(device_info_index);
    ExecEnv::log().info("GPU {}, {}, Temp: {}C, Fan: {}%, Power: {}/{}W, Memory Usage: {}/{}MB, Kernel Utilization: {}%, Memory Utilization: {}%",
                        device_index, device.getDeviceName(),
                        device_information.getDeviceTemperature(device_info_index), device_information.getFanSpeed(device_info_index),
                        device_information.getPowerUsage(device_info_index), device_information.getMaxPower(device_info_index),
                        memory.first, memory.second, utilization.first, utilization.second);

  } while(*test_running); // work loop
  std::chrono::steady_clock::time_point end_test = std::chrono::steady_clock::now();

  gpu_event_ptr->synchronize();

  // Manually destruct the test objects to prevent any race conditions.
  // Destruct the event object before the test object (contains the device context).
  gpu_event_ptr = nullptr;
  test_gpu_ptr = nullptr;

  size_t total_nano_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end_test - begin_test).count();
  double average_gflops = (flops_per_op * static_cast<double>(iteration_sum)) / static_cast<double>(total_nano_seconds);
  ExecEnv::log().info("GPU {}, {}, Test completes, Total Iterations: {}, Total Errors: {}, Average GFLOPs: {} ({})",
                      device_index, device.getDeviceName(), iteration_sum, error_sum, average_gflops,
                      (use_double_precision ? "double" : "float"));

}



void startGPUTest(size_t run_length, bool use_double_precision, bool use_tensor_cores) {

  DeviceInformation device_info;
  auto version = device_info.getLibraryVersion();
  ExecEnv::log().info("GPU Library Version: {}.{}, Driver Version: {}", version.first, version.second, device_info.getDriverVersion());

  GPUDeviceList gpu_device_list;
  size_t device_count = gpu_device_list.getDeviceCount();
  if (device_count == 0) {

    ExecEnv::log().warn("*** No GPU devices detected ***");
    return;

  }

  bool test_running = true;
  std::vector<std::thread> thread_vector;
  for (size_t device = 0; device < device_count; ++device) {

    auto compute_level = gpu_device_list.getDevice(device).getComputeCapability();
    ExecEnv::log().info("GPU {}, {}, UUID: {}, Compute Capability: {}.{}, Total Memory: {}MB",
                        gpu_device_list.getDevice(device).getDeviceIdent(),
                        gpu_device_list.getDevice(device).getDeviceName(),
                        gpu_device_list.getDevice(device).getDeviceFormattedUUID(),
                        compute_level.first, compute_level.second,
                        gpu_device_list.getDevice(device).getDeviceMemoryMbtyes());

    // Create a test thread.
    thread_vector.emplace_back(&GPUTestThread, &test_running, gpu_device_list.getDevice(device), use_double_precision, use_tensor_cores);

  }

  // Wait until the test is complete.
  sleep(run_length);
  test_running = false;

  // Join on the test threads.
  for (auto& thread : thread_vector) {

    thread.join();

  }

  ExecEnv::log().info("GPU test completes");

}




} // namespace