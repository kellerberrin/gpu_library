//
// Created by kellerberrin on 20/10/20.
//

#include "keg_gpu_mem.h"
#include "kel_exec_env.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#include <new>

namespace keg = kellerberrin::gpu;
namespace kel = kellerberrin;



keg::HostMemory::HostMemory() noexcept {

  unique_host_ptr_ = nullptr;
  byte_size_ = 0;

}

keg::HostMemory::HostMemory(size_t byte_size) noexcept {

  // Guarantees that the memory is aligned, as we may be putting doubles etc in here.
  unique_host_ptr_ = new (byte_alignment_, std::nothrow) std::byte[byte_size];
  if (unique_host_ptr_ != nullptr) {

    byte_size_ = byte_size;

  } else {

    ExecEnv::log().error("HostMemory::HostMemory; requested memory allocation size: {} failed", byte_size);
    byte_size_ = 0;

  }

}

keg::HostMemory::HostMemory(HostMemory&& rvalue_copy) noexcept {

  unique_host_ptr_ = rvalue_copy.unique_host_ptr_;
  byte_size_ = rvalue_copy.byte_size_;
  rvalue_copy.byte_size_ = 0;
  rvalue_copy.unique_host_ptr_ = nullptr;

}

keg::HostMemory::~HostMemory() noexcept {

  if (unique_host_ptr_ != nullptr) {

    delete[](unique_host_ptr_);

  }

}

keg::HostMemory& keg::HostMemory::operator=(HostMemory&& copy) noexcept {

  unique_host_ptr_ = copy.unique_host_ptr_;
  byte_size_ = copy.byte_size_;
  copy.unique_host_ptr_ = nullptr;
  copy.byte_size_ = 0;
  return *this;

}


bool keg::HostMemory::reallocateMemory(size_t byte_size) noexcept {

  if (unique_host_ptr_ != nullptr) {

    delete[](unique_host_ptr_);

  }

  unique_host_ptr_ = new (byte_alignment_, std::nothrow) std::byte[byte_size];

  if (unique_host_ptr_) {

    byte_size_ = byte_size;
    return true;

  } else {

    ExecEnv::log().error("HostMemory::reallocateMemory; requested memory allocation size: {} failed", byte_size);
    byte_size_ = 0;
    return false;

  }

}

keg::MemBlock keg::HostMemory::subBlock(size_t byte_offset, size_t byte_size) const {

  if (byte_offset + byte_size > byte_size_) {

    ExecEnv::log().error("HostMemory::subBlock; memory sub-block with offset: {} + size: {} is outside allocated memory size: {}",
                         byte_offset, byte_size, byte_size_);
    return MemBlock(0, nullptr);

  }

  return MemBlock(byte_size, (unique_host_ptr_ + byte_offset));

}


bool keg::HostMemory::transferToGPU(GPUMemory &gpu_memory) const {


  if (gpu_memory.byteSize() != byteSize()) {

    if (not gpu_memory.reallocateMemory(byteSize())) {

      ExecEnv::log().error("HostMemory::transferToHost, failed to resize GPU memory to: {} bytes for transfer", byteSize());
      return false;

    }

  }
  checkCudaErrors(cudaMemcpy(static_cast<void *>(gpu_memory.memInfo().address()),
                             static_cast<void *>(memInfo().address()),
                             byteSize(), cudaMemcpyHostToDevice));

  return true;

}

bool keg::HostMemory::transferSubBlockToGPU(size_t byte_offset, size_t byte_size, GPUMemory &gpu_memory) const {

  // Check for a valid sub-block.
  if (subBlock(byte_offset, byte_size).byteSize() == 0) {

    ExecEnv::log().error("HostMemory::transferSubBlockToGPU; memory sub-block with offset: {} + size: {} is outside allocated memory size: {}",
                         byte_offset, byte_size, byte_size_);
    return false;

  }

  // Resize the device memory if required.
  if (gpu_memory.byteSize() != byte_size) {

    if (not gpu_memory.reallocateMemory(byte_size)) {

      ExecEnv::log().error("HostMemory::transferToHost, failed to resize GPU memory to: {} bytes for transfer", byte_size);
      return false;

    }

  }

  // Transfer the sub-block.
  checkCudaErrors(cudaMemcpy(static_cast<void *>(subBlock(byte_offset, byte_size).address()),
                             static_cast<void *>(gpu_memory.memInfo().address()),
                             byte_size, cudaMemcpyHostToDevice));

  return true;

}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU device memory. This is a private implementation object.

namespace kellerberrin::gpu {   //  organization::project level namespace

class DeviceMemory {

public:

  DeviceMemory();
  explicit DeviceMemory(size_t byte_size);
  ~DeviceMemory();

  [[nodiscard]] size_t byteSize() const { return byte_size_; }
  [[nodiscard]] void* getAddress() const { return device_mem_ptr_; }
  [[nodiscard]] void** getPtrAddress() { return &device_mem_ptr_; }

  bool reallocateMemory(size_t byte_size);

  // The host memory block is resized (if required) and the device memory contents transferred to the host memory.
  bool transferToHost(HostMemory &host_memory) const;
  // The Host memory block is resized (if required) and the sub block (if valid) contents transferred to the host memory.
  bool transferSubBlockToHost(size_t byte_offset, size_t byte_size, HostMemory &host_memory) const;

private:

  void *device_mem_ptr_;
  size_t byte_size_;


};

DeviceMemory::DeviceMemory() {

  device_mem_ptr_ = nullptr;
  byte_size_ = 0;

}


DeviceMemory::DeviceMemory(size_t byte_size) {

  checkCudaErrors(cudaMalloc(&device_mem_ptr_, byte_size));
  if (device_mem_ptr_ == nullptr) {

    ExecEnv::log().error("DeviceMemory::DeviceMemory; unable to allocate device memory, bytes: {}", byte_size);
    byte_size_ = 0;

  } else {

    byte_size_ = byte_size;

  }

}


DeviceMemory::~DeviceMemory() {

  if (device_mem_ptr_ != nullptr) {

    checkCudaErrors(cudaFree(device_mem_ptr_));

  }

}

// Allocating zero bytes frees the memory.
bool DeviceMemory::reallocateMemory(size_t byte_size) {

  // Delete any existing allocation.
  if (device_mem_ptr_ != nullptr) {

    checkCudaErrors(cudaFree(device_mem_ptr_));

  }

  // If byte size zero then just return.
  if (byte_size == 0) {

    byte_size_ = 0;
    device_mem_ptr_ = nullptr;
    return true;

  }

  // Else allocate the new block of device memory.
  checkCudaErrors(cudaMalloc(&device_mem_ptr_, byte_size));
  if (device_mem_ptr_ == nullptr) {

    byte_size_ = 0;
    ExecEnv::log().error("DeviceMemory::reallocateMemory; unable to allocate device memory, bytes: {}", byte_size);
    return false;

  } else {

    byte_size_ = byte_size;

  }

  return true;

}


bool DeviceMemory::transferToHost(HostMemory &host_memory) const {

  // Check dimensions.
  if (host_memory.byteSize() != byteSize()) {

    if (not host_memory.reallocateMemory(byteSize())) {

      ExecEnv::log().error("DeviceMemory::transferToHost, failed to resize host memory to: {} bytes for transfer", byteSize());
      return false;

    }

  }

  checkCudaErrors(cudaMemcpy(static_cast<void *>(host_memory.memInfo().address()),
                             static_cast<void *>(getAddress()),
                             byteSize(),
                             cudaMemcpyDeviceToHost));

  return true;

}


bool DeviceMemory::transferSubBlockToHost(size_t byte_offset, size_t byte_size, HostMemory &host_memory) const {

  // Check for a valid sub-block.
  if (byte_offset + byte_size > byte_size_) {

    ExecEnv::log().error("DeviceMemory::transferSubBlockToHost; memory sub-block with offset: {} + size: {} is outside allocated memory size: {}",
                         byte_offset, byte_size, byte_size_);
    return false;

  }

  // Resize the device memory if required.
  if (host_memory.byteSize() != byte_size) {

    if (not host_memory.reallocateMemory(byte_size)) {

      ExecEnv::log().error("DeviceMemory::transferSubBlockToHost, failed to resize host memory to: {} bytes for transfer", byte_size);
      return false;

    }

  }

  // Transfer the sub-block to the host memory.
  checkCudaErrors(cudaMemcpy(static_cast<void *>(host_memory.memInfo().address()),
                             static_cast<void *>(static_cast<std::byte*>(getAddress()) + byte_offset),
                             byte_size, cudaMemcpyDeviceToHost));

  return true;

}


} // namespace.

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU device memory held in a unique pointer to control lifetime and avoid multiple deletion

keg::GPUMemory::GPUMemory() noexcept {

  unique_device_ptr_ = std::make_unique<DeviceMemory>();

}

keg::GPUMemory::~GPUMemory() noexcept = default;


keg::GPUMemory::GPUMemory(size_t byte_size) noexcept {

  unique_device_ptr_ = std::make_unique<DeviceMemory>(byte_size);

}

keg::GPUMemory::GPUMemory(GPUMemory&& rvalue_copy) noexcept {

  // Transfer any device memory from the rvalue.
  unique_device_ptr_= std::move(rvalue_copy.unique_device_ptr_);
  // Reset the rvalue.
  rvalue_copy.unique_device_ptr_ = std::make_unique<DeviceMemory>();

}


keg::GPUMemory& keg::GPUMemory::operator=(GPUMemory&& copy) noexcept {

  unique_device_ptr_ = std::move(copy.unique_device_ptr_);
  return *this;

}

bool keg::GPUMemory::reallocateMemory(size_t byte_size) noexcept {

  return unique_device_ptr_->reallocateMemory(byte_size);

}

size_t keg::GPUMemory::byteSize() const {

  return unique_device_ptr_->byteSize();

}

std::byte* keg::GPUMemory::getAddress() const {

  return static_cast<std::byte*>(unique_device_ptr_->getAddress());

}

keg::MemBlock keg::GPUMemory::subBlock(size_t byte_offset, size_t byte_size) const {

  if (byte_offset + byte_size > unique_device_ptr_->byteSize()) {

    ExecEnv::log().error("GPUMemory::subBlock; memory sub-block with offset: {} + size: {} is outside allocated memory size: {}",
                         byte_offset, byte_size, unique_device_ptr_->byteSize());
    return MemBlock(0, nullptr);

  }

  return MemBlock(byte_size, (getAddress() + byte_offset));

}

bool keg::GPUMemory::transferToHost(HostMemory &host_memory) const {

  return unique_device_ptr_->transferToHost(host_memory);

}

bool keg::GPUMemory::transferSubBlockToHost(size_t byte_offset, size_t byte_size, HostMemory &host_memory) const {

  return unique_device_ptr_->transferSubBlockToHost(byte_offset, byte_size, host_memory);

}


