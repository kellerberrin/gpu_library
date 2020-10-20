//
// Created by kellerberrin on 9/10/20.
//

#ifndef KEG_GPU_DEVICE_H
#define KEG_GPU_DEVICE_H


#include <vector>
#include <stack>
#include <memory>
#include <optional>

namespace kellerberrin::gpu {   //  organization::project level namespace


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Opaque facade classes using the PIMPL idiom to hide the Nvidia Cuda device library.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward declaration of the implementation.
class GPUEventImpl;

class GPUEvent {

public:

  GPUEvent();
  ~GPUEvent();

  // Record events in the stream
  bool record();
  // Returns false if outstanding events in the stream, true otherwise.
  bool pollOnRecord();
  // Wait on all events in the stream (conceptually a record() followed by a waitOnRecord()).
  bool synchronize();

private:

  std::unique_ptr<GPUEventImpl> event_impl_;

};



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward declaration of the implementation.
class DriverDeviceImpl;

class DriverDevice {

public:

  // The implementation.

  DriverDevice(const DriverDevice& driver_device);
  explicit DriverDevice(const DriverDeviceImpl& driver_device_impl);
  ~DriverDevice();

  [[nodiscard]] std::string getDeviceFormattedUUID() const;
  [[nodiscard]] std::vector<std::byte> getDeviceUUIDBytes() const;
  // .first is the major capability level, .second is the minor capability level.
  [[nodiscard]] std::pair<size_t, size_t> getComputeCapability() const;
  [[nodiscard]] std::string getDeviceName() const;
  [[nodiscard]] size_t getDeviceMemoryMbtyes() const;
  [[nodiscard]] size_t getDeviceIdent() const;
  // Const impl access.
  [[nodiscard]] const DriverDeviceImpl* constImpl() const { return driver_device_impl_.get(); }

private:

  std::unique_ptr<DriverDeviceImpl> driver_device_impl_;

};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward declaration of the implementation.
class ThreadContextImpl;

class ThreadContext {

public:

  ThreadContext(const DriverDevice& driver_device);
  ~ThreadContext();

  bool bindToThread();

private:

  std::unique_ptr<ThreadContextImpl> thread_context_impl_;

};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward declaration of the implementation.
class GPUDeviceListImpl;

class GPUDeviceList {

public:

  GPUDeviceList();
  ~GPUDeviceList();

  // Number of active CUDA devices
  [[nodiscard]] size_t getDeviceCount() const;
  [[nodiscard]] DriverDevice getDevice(size_t index) const;

private:

  // The implementation.
  std::unique_ptr<GPUDeviceListImpl> cuda_driver_impl_;

  // Const impl access.
  [[nodiscard]] const GPUDeviceListImpl* constImpl() const { return cuda_driver_impl_.get(); }

};


} // namespace


#endif //KEG_GPU_DEVICE_H
