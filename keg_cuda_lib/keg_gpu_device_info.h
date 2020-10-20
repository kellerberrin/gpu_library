//
// Created by kellerberrin on 8/10/20.
//

#ifndef KEG_DEVICE_INFO_H
#define KEG_DEVICE_INFO_H


#include <memory>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// An opaque facade class using the PIMPL idiom to hide the Nvidia NVML device interrogation library.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


namespace kellerberrin::gpu {   //  organization::project level namespace

// Forward declaration of the implementation.
class DeviceInformationImpl;

class DeviceInformation {

public:

  explicit DeviceInformation();
  ~DeviceInformation();

  // Number of active CUDA devices
  size_t getDeviceCount();
  // Temperature of a specified device.
  size_t getDeviceTemperature(size_t device_index);
  // Get the Cuda software version .first is the major version, .second the minor.
  std::pair<size_t, size_t> getLibraryVersion();

  std::string getDriverVersion();

  std::string getDeviceName(size_t device_index);

  // Compute capability, .first is the major level, .second is the the minor
  std::pair<size_t, size_t> getComputeLevel(size_t device_index);

  size_t getFanSpeed(size_t device_index);

  size_t getMaxPower(size_t device_index);

  size_t getPowerUsage(size_t device_index);

  // Memory usage (Mbytes) is .first, total memory available is .second.
  std::pair<size_t, size_t> getMemoryUsage(size_t device_index);

  std::string getDeviceUUID(size_t device_index);

  // %Memory utilization is .first, %kernel utilization is .second.
  std::pair<size_t, size_t> getGPUUtilization(size_t device_index);

  size_t getIndexUsingUUID(const std::string& formatted_UUID);

private:

  std::unique_ptr<DeviceInformationImpl> device_information_impl_;

};



} // namespace

#endif // KEG_DEVICE_INFO_H
