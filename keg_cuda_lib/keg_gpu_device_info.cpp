//
// Created by kellerberrin on 8/10/20.
//


#include "keg_gpu_device_info.h"
#include "kel_exec_env.h"

#include <nvml.h>

#include <map>

namespace keg = kellerberrin::gpu;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// The PIMPL Nvidia NVML implementation class
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



class keg::DeviceInformationImpl {

public:

  explicit DeviceInformationImpl();
  ~DeviceInformationImpl();


  size_t getDeviceCount() { return nvml_device_vec_.size(); }
  size_t getDeviceTemperature(size_t device);
  std::pair<size_t, size_t> getLibraryVersion();
  std::string getDriverVersion();
  std::string getDeviceName(size_t device);
  std::pair<size_t, size_t> getComputeLevel(size_t device);
  size_t getFanSpeed(size_t device);
  size_t getMaxPower(size_t device);
  size_t getPowerUsage(size_t device);
  std::pair<size_t, size_t> getMemoryUsage(size_t device);
  std::string getDeviceUUID(size_t device);
  std::pair<size_t, size_t> getGPUUtilization(size_t device);
  size_t getIndexUsingUUID(const std::string& UUID);


private:

  std::vector<nvmlDevice_t> nvml_device_vec_;

  static std::pair<bool, std::string> checkInfoError(nvmlReturn_t return_code);
  static bool checkInfoCode(nvmlReturn_t nvml_return_code, const std::string& module_text);
  bool validDevice(size_t device);

};


bool keg::DeviceInformationImpl::validDevice(size_t device) {

  if (device >= nvml_device_vec_.size()) {

    ExecEnv::log().error("DeviceInformationPimpl::validDevice; invalid device ident: {}, exceeds devices available: {}", device, nvml_device_vec_.size());
    return false;

  }

  return true;

}


// Get all active devices.
keg::DeviceInformationImpl::DeviceInformationImpl() {

  if (checkInfoCode(nvmlInit(), "nvmlInit")) {

    unsigned int device_count;
    if (checkInfoCode(nvmlDeviceGetCount_v2(&device_count), "nvmlDeviceGetCount_v2")) {

      for (unsigned int device = 0; device < device_count; ++device) {

        nvmlDevice_t gpu_device;
        if (checkInfoCode(nvmlDeviceGetHandleByIndex_v2(device, &gpu_device), "nvmlDeviceGetHandleByIndex_v2")) {

          nvml_device_vec_.push_back(gpu_device);

        }

      }

    }

  }

}

keg::DeviceInformationImpl::~DeviceInformationImpl() {

  checkInfoCode(nvmlShutdown(), "nvmlShutdown");

}

std::pair<bool, std::string> keg::DeviceInformationImpl::checkInfoError(nvmlReturn_t return_code) {

  static std::map<nvmlReturn_t, std::string> nvml_error_strings = {

      { NVML_SUCCESS, "The operation was successful."},
      { NVML_ERROR_UNINITIALIZED, "NVML was not first initialized with nvmlInit()" },
      { NVML_ERROR_INVALID_ARGUMENT, "A supplied argument is invalid." },
      { NVML_ERROR_NOT_SUPPORTED, "The requested operation is not available on target device."},
      { NVML_ERROR_NO_PERMISSION, "The current user does not have permission for operation."},
      { NVML_ERROR_ALREADY_INITIALIZED, "Deprecated: Multiple initializations are now allowed through ref counting."},
      { NVML_ERROR_NOT_FOUND, "A query to find an object was unsuccessful."},
      { NVML_ERROR_INSUFFICIENT_SIZE, "An input argument is not large enough." },
      { NVML_ERROR_INSUFFICIENT_POWER, "A device's external power cables are not properly attached." },
      { NVML_ERROR_DRIVER_NOT_LOADED, "NVIDIA driver is not loaded." },
      { NVML_ERROR_TIMEOUT, "User provided timeout passed." },
      { NVML_ERROR_IRQ_ISSUE, "NVIDIA Kernel detected an interrupt issue with a GPU." },
      { NVML_ERROR_LIBRARY_NOT_FOUND, "NVML Shared Library couldn't be found or loaded." },
      { NVML_ERROR_FUNCTION_NOT_FOUND, "Local version of NVML doesn't implement this function."},
      { NVML_ERROR_CORRUPTED_INFOROM, "infoROM is corrupted" },
      { NVML_ERROR_GPU_IS_LOST, "The GPU has fallen off the bus or has otherwise become inaccessible."},
      { NVML_ERROR_RESET_REQUIRED, "The GPU requires a reset before it can be used again."},
      { NVML_ERROR_OPERATING_SYSTEM, "The GPU control device has been blocked by the operating system/cgroups."},
      { NVML_ERROR_LIB_RM_VERSION_MISMATCH, "RM detects a driver/library version mismatch." },
      { NVML_ERROR_IN_USE, "An operation cannot be performed because the GPU is currently in use." },
      { NVML_ERROR_MEMORY, "Insufficient memory." },
      { NVML_ERROR_NO_DATA, "No data." },
      { NVML_ERROR_VGPU_ECC_NOT_SUPPORTED, "The requested vgpu operation is not available on target device, because ECC is enabled."},
      { NVML_ERROR_INSUFFICIENT_RESOURCES, "Ran out of critical resources, other than memory."},
      { NVML_ERROR_UNKNOWN, "An internal driver error occurred." },

  };

  auto error_record = nvml_error_strings.find(return_code);

  if (error_record == nvml_error_strings.end()) {

    return {false, "DeviceInformation::checkInfoError; (nvmlReturn_t) return code not defined"};

  }

  return {(return_code == NVML_SUCCESS), error_record->second};

}


bool keg::DeviceInformationImpl::checkInfoCode(nvmlReturn_t nvml_return_code, const std::string& module_text) {

  auto text_return_code = checkInfoError(nvml_return_code);

  // If the module text is empty then the NVML call fails silently
  if (not text_return_code.first and not module_text.empty()) {

    ExecEnv::log().error("NVML module: {} failed; reason: '{}'", module_text, text_return_code.second);

  }

  return text_return_code.first;

}

size_t keg::DeviceInformationImpl::getDeviceTemperature(size_t device) {

  if (not validDevice(device)) {

    return 0;

  }

  unsigned int temp;
  if (not checkInfoCode(nvmlDeviceGetTemperature(nvml_device_vec_[device], NVML_TEMPERATURE_GPU, &temp), "nvmlDeviceGetTemperature")) {

    return 0;

  }

  return static_cast<size_t>(temp);

}


std::pair<size_t, size_t> keg::DeviceInformationImpl::getLibraryVersion() {

  int cuda_driver_version;
  if (not checkInfoCode(nvmlSystemGetCudaDriverVersion(&cuda_driver_version), "nvmlSystemGetCudaDriverVersion")) {

    return {0, 0};

  }

  auto major_version = static_cast<size_t>(cuda_driver_version / 1000);
  auto minor_version = static_cast<size_t>(cuda_driver_version % 1000);

  return {major_version, minor_version};

}



std::string keg::DeviceInformationImpl::getDriverVersion() {

  const size_t version_str_size = 512;
  char driver_version[version_str_size];
  if (not checkInfoCode(nvmlSystemGetDriverVersion(driver_version, version_str_size - 1), "nvmlSystemGetDriverVersion")) {

    return "";

  }


  return std::string(driver_version);

}


std::string keg::DeviceInformationImpl::getDeviceName(size_t device) {

  const size_t name_str_size = 128;
  char device_name[name_str_size];

  if (not validDevice(device)) {

    return "";

  }

  if (not checkInfoCode(nvmlDeviceGetName(nvml_device_vec_[device], device_name, name_str_size), "nvmlDeviceGetName")) {

    return "";

  }


  return std::string(device_name);

}



std::pair<size_t, size_t> keg::DeviceInformationImpl::getComputeLevel(size_t device) {

  if (not validDevice(device)) {

    return {0, 0};

  }

  int major_level;
  int minor_level;
  if (not checkInfoCode(nvmlDeviceGetCudaComputeCapability(nvml_device_vec_[device], &major_level, &minor_level), "nvmlDeviceGetCudaComputeCapability")) {

    return {0, 0};

  }

  return {major_level, minor_level};

}



size_t keg::DeviceInformationImpl::getFanSpeed(size_t device) {

  if (not validDevice(device)) {

    return 0;

  }

  unsigned int fan_speed;
  if (not checkInfoCode(nvmlDeviceGetFanSpeed(nvml_device_vec_[device], &fan_speed), "nvmlDeviceGetFanSpeed")) {

    return 0;

  }

  return fan_speed;

}


size_t keg::DeviceInformationImpl::getMaxPower(size_t device) {

  if (not validDevice(device)) {

    return 0;

  }

  unsigned int max_power;
  if (not checkInfoCode(nvmlDeviceGetPowerManagementLimit(nvml_device_vec_[device], &max_power), "")) {

    return 0;

  }

  return (max_power / 1000);

}


size_t keg::DeviceInformationImpl::getPowerUsage(size_t device) {

  if (not validDevice(device)) {

    return 0;

  }

  unsigned int power_usage;
  if (not checkInfoCode(nvmlDeviceGetPowerUsage(nvml_device_vec_[device], &power_usage), "")) {

    return 0;

  }

  return (power_usage / 1000);

}



std::pair<size_t, size_t> keg::DeviceInformationImpl::getMemoryUsage(size_t device) {

  if (not validDevice(device)) {

    return {0, 0};

  }

  nvmlMemory_t memory;
  if (not checkInfoCode(nvmlDeviceGetMemoryInfo(nvml_device_vec_[device], &memory), "nvmlDeviceGetMemoryInfo")) {

    return {0, 0};

  }

  return {(memory.used / (1024 * 1024)), (memory.total / (1024 * 1024))};

}


std::string keg::DeviceInformationImpl::getDeviceUUID(size_t device) {

  const size_t uuid_str_size = 128;
  char device_uuid[uuid_str_size];

  if (not validDevice(device)) {

    return "";

  }

  if (not checkInfoCode(nvmlDeviceGetUUID(nvml_device_vec_[device], device_uuid, uuid_str_size), "nvmlDeviceGetUUID")) {

    return "";

  }


  return std::string(device_uuid);

}



std::pair<size_t, size_t> keg::DeviceInformationImpl::getGPUUtilization(size_t device) {

  if (not validDevice(device)) {

    return {0, 0};

  }


  nvmlUtilization_t utilization;
  if (not checkInfoCode(nvmlDeviceGetUtilizationRates(nvml_device_vec_[device], &utilization), "nvmlDeviceGetUtilizationRates")) {

    return {0, 0};

  }

  return { utilization.gpu, utilization.memory };

}


size_t keg::DeviceInformationImpl::getIndexUsingUUID(const std::string& UUID) {

  nvmlDevice_t uuid_device;
 if (not checkInfoCode(nvmlDeviceGetHandleByUUID (UUID.c_str(), &uuid_device ), "nvmlDeviceGetHandleByUUID")) {

    return 0;

  }

  size_t index = 0;
  for (auto& device : nvml_device_vec_) {

    if (device == uuid_device) {

      return index;

    }

    index++;

  }

  // Index not found, complain and return.
  ExecEnv::log().error("DeviceInformation::getNVMLHandleUsingUUIDl; device index not found for UUID: {}", UUID);
  return 0;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Redirect function calls to the implementation object.
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

keg::DeviceInformation::DeviceInformation() {

  device_information_impl_ = std::make_unique<DeviceInformationImpl>();

}

keg::DeviceInformation::~DeviceInformation() {}


// Number of active CUDA devices
size_t keg::DeviceInformation::getDeviceCount() {

  return device_information_impl_->getDeviceCount();

}
// Temperature of a specified device.
size_t keg::DeviceInformation::getDeviceTemperature(size_t device) {

  return device_information_impl_->getDeviceTemperature(device);

}

std::pair<size_t, size_t> keg::DeviceInformation::getLibraryVersion() {

  return device_information_impl_->getLibraryVersion();

}

std::string keg::DeviceInformation::getDriverVersion() {

  return device_information_impl_->getDriverVersion();

}

std::string keg::DeviceInformation::getDeviceName(size_t device) {

  return device_information_impl_->getDeviceName(device);

}

std::pair<size_t, size_t> keg::DeviceInformation::getComputeLevel(size_t device) {

  return device_information_impl_->getComputeLevel(device);

}

size_t keg::DeviceInformation::getFanSpeed(size_t device) {

  return device_information_impl_->getFanSpeed(device);

}

size_t keg::DeviceInformation::getMaxPower(size_t device) {

  return device_information_impl_->getMaxPower(device);

}

size_t keg::DeviceInformation::getPowerUsage(size_t device) {

  return device_information_impl_->getPowerUsage(device);

}

std::pair<size_t, size_t> keg::DeviceInformation::getMemoryUsage(size_t device) {

  return device_information_impl_->getMemoryUsage(device);

}

std::string keg::DeviceInformation::getDeviceUUID(size_t device) {

  return device_information_impl_->getDeviceUUID(device);

}


std::pair<size_t, size_t> keg::DeviceInformation::getGPUUtilization(size_t device) {

  return device_information_impl_->getGPUUtilization(device);

}

size_t keg::DeviceInformation::getIndexUsingUUID(const std::string& UUID) {

  return device_information_impl_->getIndexUsingUUID(UUID);

}
