//
// Created by kellerberrin on 17/10/20.
//

#ifndef KEG_CUDA_DEVICE_H
#define KEG_CUDA_DEVICE_H


#include "keg_gpu_device.h"

#include <cuda.h>

#include <map>
#include <iomanip>

namespace kellerberrin::gpu {   //  organization::project level namespace


class CheckCode {

public:

  CheckCode() = delete;
  ~CheckCode() = delete;

  static bool check(CUresult cuda_return_code, const std::string& module_text = "");


private:

  static std::pair<bool, std::string> checkCUResult(CUresult return_code);

};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GPUEventImpl {

public:

  GPUEventImpl();
  ~GPUEventImpl();

  // Record events in the stream
  bool record();
  // Returns false if outstanding events in the stream, true otherwise.
  bool pollOnRecord();
  // Wait on all events in the stream (conceptually a record() followed by a waitOnRecord()).
  bool synchronize();

private:

  CUevent event_;
  constexpr static const CUstream DEFAULT_STREAM_ = 0;

};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class DriverDeviceImpl {

public:

  explicit DriverDeviceImpl(CUdevice cu_device) : gpu_device_(cu_device) {}
  DriverDeviceImpl(const DriverDeviceImpl&) = default;
  ~DriverDeviceImpl() = default;

  [[nodiscard]] std::string getDeviceUUID() const;
  [[nodiscard]] std::string getDeviceFormattedUUID() const;
  [[nodiscard]] std::vector<std::byte> getDeviceUUIDBytes() const;

  // .first is the major capability level, .second is the minor capability level.
  [[nodiscard]] std::pair<size_t, size_t> getComputeCapability() const;
  [[nodiscard]] std::string getDeviceName() const;
  [[nodiscard]] size_t getDeviceMemoryMbtyes() const;
  [[nodiscard]] CUdevice getCuDevice() const { return gpu_device_; }
  [[nodiscard]] size_t getDeviceIdent() const { return static_cast<size_t>(gpu_device_); }
  // In bytes, .first total memory, .second available (free) memory.
  [[nodiscard]] std::pair<size_t, size_t>  memoryInfo() const;

private:

  CUdevice gpu_device_;

  constexpr static const size_t MBYTE_ = 1024 * 1024;

};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class ThreadContextImpl {

public:

  ThreadContextImpl(const DriverDeviceImpl& driver_device_impl);
  ~ThreadContextImpl();

  // Binds the calling thread to the context.
  bool bindToThread();

private:

  CUcontext cu_context_;
  bool valid_context_{false};
  bool bound_to_thread_{false};

};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class GPUDeviceListImpl {

public:

  GPUDeviceListImpl();
  ~GPUDeviceListImpl() = default;

  size_t getDeviceCount() const { return cuda_device_vec_.size(); }
  DriverDeviceImpl getDevice(size_t index) const { return cuda_device_vec_[index]; }

private:

  std::vector<DriverDeviceImpl> cuda_device_vec_;

};



} //namespace


#endif //GPU_LIBRARY_KEG_CUDA_DEVICE_H
