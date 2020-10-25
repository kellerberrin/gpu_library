//
// Created by kellerberrin on 17/10/20.
//

#include "keg_cuda_device.h"

#include "kel_exec_env.h"


namespace kel = kellerberrin;
namespace keg = kellerberrin::gpu;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// The Cuda implementation.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


std::pair<bool, std::string> keg::CheckCode::checkCudaError(CUresult return_code) {

  if (return_code == CUDA_SUCCESS) {

    return { true, "The CUDA operation was successful."};

  }

  const char* error_name_ptr;
  auto name_result = cuGetErrorName (return_code, &error_name_ptr);

  if (name_result != CUDA_SUCCESS) {

    kel::ExecEnv::log().error("CheckCode::checkCudaError; invalid (unrecognized) error code: {} returned", static_cast<size_t>(return_code));
    return { false, "CheckCode::checkCudaError; invalid (unrecognized) error code returned"};

  }

  const char* error_description_ptr;
  auto description_result = cuGetErrorString (return_code, &error_description_ptr );

  if (description_result != CUDA_SUCCESS) {

    kel::ExecEnv::log().error("CheckCode::checkCudaError; invalid (unrecognized) error code: {} returned", static_cast<size_t>(return_code));
    return { false, "CheckCode::checkCudaError; invalid (unrecognized) error code returned"};

  }

  std::string error_string = std::string(error_name_ptr) + std::string("; ") + std::string(error_description_ptr);

  return { false, error_string };

}


bool keg::CheckCode::check(CUresult cuda_return_code, const std::string& module_text) {

  auto text_return_code = checkCudaError(cuda_return_code);

  if (not text_return_code.first) {

    if (not module_text.empty()) {

      kel::ExecEnv::log().error("Cuda module: {} failed; reason: '{}'", module_text, text_return_code.second);

    } else {

      kel::ExecEnv::log().error("Cuda driver API call failed; reason: '{}'",text_return_code.second);

    }
    return false;

  }

  return true;

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

keg::GPUEventImpl::GPUEventImpl() {

  CheckCode::check(cuEventCreate(&event_, CU_EVENT_DEFAULT));

}

keg::GPUEventImpl::~GPUEventImpl() {

  CheckCode::check(cuEventDestroy(event_), "cuEventDestroy");

}

bool keg::GPUEventImpl::record() {

  return CheckCode::check(cuEventRecord(event_, DEFAULT_STREAM_));

}

bool keg::GPUEventImpl::pollOnRecord() {

  return cuEventQuery(event_) == CUDA_SUCCESS;

}

bool keg::GPUEventImpl::synchronize() {

  return CheckCode::check(cuEventSynchronize(event_));

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<size_t, size_t> keg::DriverDeviceImpl::getComputeCapability() const {

  int major_level{0};
  int minor_level{0};

  CheckCode::check(cuDeviceGetAttribute (&major_level, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, getCuDevice()));
  CheckCode::check(cuDeviceGetAttribute (&minor_level, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, getCuDevice()));

  return {major_level, minor_level};

}

std::vector<std::byte> keg::DriverDeviceImpl::getDeviceUUIDBytes() const {

  CUuuid uuid;
  std::vector<std::byte> uuid_vector;

  if (not CheckCode::check(cuDeviceGetUuid(&uuid, getCuDevice()))) {

    return uuid_vector;

  }

  for (size_t index = 0; index <  sizeof(uuid); ++index) {

    uuid_vector.push_back(static_cast<std::byte>(uuid.bytes[index]));

  }

  return uuid_vector;

}

std::string keg::DriverDeviceImpl::getDeviceUUID() const {

  auto byte_vector = getDeviceUUIDBytes();

  std::stringstream ss;
  ss << std::hex;

  for(auto uuid_byte : byte_vector) {

    ss << std::setw(2) << std::setfill('0') << static_cast<size_t>(uuid_byte);

  }

  return ss.str();

}


std::string keg::DriverDeviceImpl::getDeviceFormattedUUID() const {

  const size_t uuid_digit_count = 32;
  std::string unformatted_uuid = getDeviceUUID();

  if (unformatted_uuid.size() != uuid_digit_count) {

    ExecEnv::log().error("DriverDevicePimpl::getDeviceFormattedUUID; Expected {} UUID digits, actual digits: {} in UUID: {}",
                         uuid_digit_count, unformatted_uuid.size(), unformatted_uuid);
    return "";

  }

  std::string formatted_uuid = "GPU-";
  formatted_uuid += std::string_view(&unformatted_uuid[0], 8);
  formatted_uuid += std::string("-");
  formatted_uuid += std::string_view(&unformatted_uuid[8], 4);
  formatted_uuid += std::string("-");
  formatted_uuid += std::string_view(&unformatted_uuid[12], 4);
  formatted_uuid += std::string("-");
  formatted_uuid += std::string_view(&unformatted_uuid[16], 4);
  formatted_uuid += std::string("-");
  formatted_uuid += std::string_view(&unformatted_uuid[20]);

  return formatted_uuid;

}

std::string keg::DriverDeviceImpl::getDeviceName() const {

  const size_t device_name_size = 128;
  char device_name[device_name_size];

  if (not CheckCode::check(cuDeviceGetName (device_name, device_name_size, getCuDevice()))) {

    return "";

  }

  return std::string(device_name);

}

[[nodiscard]] size_t keg::DriverDeviceImpl::getDeviceMemoryMbtyes() const {

  size_t bytes{0};
  CheckCode::check(cuDeviceTotalMem (&bytes, getCuDevice()));

  return (bytes / MBYTE_);

}


std::pair<size_t, size_t>  keg::DriverDeviceImpl::memoryInfo() const {

  size_t total_memory{0}, free_memory{0};
  CheckCode::check(cuMemGetInfo(&free_memory, &total_memory));

  return { total_memory , free_memory };

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


keg::ThreadContextImpl::ThreadContextImpl(const DriverDeviceImpl& driver_device_impl) {

  valid_context_ = CheckCode::check(cuCtxCreate(&cu_context_, 0, driver_device_impl.getCuDevice()), "cuCtxCreate");

}

keg::ThreadContextImpl::~ThreadContextImpl() {

  if (valid_context_) {

    CheckCode::check(cuCtxDestroy(cu_context_), "cuCtxDestroy");

  }

}

bool keg::ThreadContextImpl::bindToThread() {

  if (valid_context_) {

    // Don't allow context stacks (yet).
    if (not bound_to_thread_) {

      bound_to_thread_ = CheckCode::check(cuCtxSetCurrent(cu_context_), "cuCtxSetCurrent");

    } else {

      ExecEnv::log().error("ThreadContext::bindToThread; context already bound (context stacks not permitted yet)");
      return false;

    }

  } else {

    ExecEnv::log().error("ThreadContext::bindToThread; attempt to bind invalid context to thread");
    return false;

  }

  return true;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Get all active devices.
keg::GPUDeviceListImpl::GPUDeviceListImpl() {

  if (CheckCode::check(cuInit(0), "cuInit")) {

    int device_count;
    if (CheckCode::check(cuDeviceGetCount(&device_count), "cuDeviceGetCount")) {

      for (int device = 0; device < device_count; ++device) {

        CUdevice gpu_device;
        if (CheckCode::check(cuDeviceGet(&gpu_device, device), "cuDeviceGet")) {

          cuda_device_vec_.emplace_back(DriverDeviceImpl(gpu_device));

        }

      }

    }

  }

}

