//
// Created by kellerberrin on 9/10/20.
//

#include "keg_gpu_device.h"
#include "kel_exec_env.h"

#include <map>

#include "keg_cuda_device.h"

namespace kel = kellerberrin;
namespace keg = kellerberrin::gpu;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Redirect function calls to the implementation object.
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


keg::GPUEvent::GPUEvent() {

  event_impl_ = std::make_unique<GPUEventImpl>();

}

keg::GPUEvent::~GPUEvent() = default;

bool keg::GPUEvent::record() {

  return event_impl_->record();

}

bool keg::GPUEvent::pollOnRecord() {

  return event_impl_->pollOnRecord();

}

bool keg::GPUEvent::synchronize() {

  return event_impl_->synchronize();

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


keg::ThreadContext::ThreadContext(const DriverDevice& driver_device) {

  thread_context_impl_ = std::make_unique<ThreadContextImpl>(*driver_device.constImpl());

}

keg::ThreadContext::~ThreadContext() = default;

bool keg::ThreadContext::bindToThread() {

  return thread_context_impl_->bindToThread();

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


keg::DriverDevice::DriverDevice(const DriverDevice& driver_device) :
  driver_device_impl_(std::make_unique<DriverDeviceImpl>(*driver_device.driver_device_impl_)) {}

keg::DriverDevice::DriverDevice(const DriverDeviceImpl& driver_device_impl) :
  driver_device_impl_(std::make_unique<DriverDeviceImpl>(driver_device_impl)) {}

keg::DriverDevice::~DriverDevice() = default;

size_t keg::DriverDevice::getDeviceIdent() const {

  return constImpl()->getDeviceIdent();

}

std::string keg::DriverDevice::getDeviceName() const {

  return constImpl()->getDeviceName();

}


std::string keg::DriverDevice::getDeviceFormattedUUID() const {

  return constImpl()->getDeviceFormattedUUID();

}

// .first is the major capability level .second is the minor capability level.
std::pair<size_t, size_t> keg::DriverDevice::getComputeCapability() const {

  return constImpl()->getComputeCapability();

}

size_t keg::DriverDevice::getDeviceMemoryMbtyes() const {

  return constImpl()->getDeviceMemoryMbtyes();

}

// In bytes, .first total memory, .second available (free) memory.
[[nodiscard]] std::pair<size_t, size_t>  keg::DriverDevice::memoryInfo() const {

  return constImpl()->memoryInfo();

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


keg::GPUDeviceList::GPUDeviceList() : cuda_driver_impl_(std::make_unique<GPUDeviceListImpl>()) {}

keg::GPUDeviceList::~GPUDeviceList() = default;

size_t keg::GPUDeviceList::getDeviceCount() const {

  return constImpl()->getDeviceCount();

}

keg::DriverDevice keg::GPUDeviceList::getDevice(size_t index) const {

  if (index >= getDeviceCount()) {

    ExecEnv::log().error("GPUDeviceList::getDevice, out of range device index: {}, devices available: {}", index, getDeviceCount());
    return DriverDevice(DriverDeviceImpl(static_cast<CUdevice>(0)));

  }

  return DriverDevice(constImpl()->getDevice(index));

}