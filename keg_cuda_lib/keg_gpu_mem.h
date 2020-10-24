//
// Created by kellerberrin on 20/10/20.
//

#ifndef KEG_CUDA_MEM_H
#define KEG_CUDA_MEM_H

#include <string>
#include <memory>


namespace kellerberrin::gpu {   //  organization::project level namespace

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// A simple class that just returns information about device or host memory.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MemBlock {

private:

  // Can only be created by the Host and Device memory classes
  friend class HostMemory;
  friend class GPUMemory;
  MemBlock(size_t byte_size, std::byte* block_location) : byte_size_(byte_size), block_location_(block_location) {}

public:

  // Can only be std::moved, not copied or copy constructed.
  MemBlock(const MemBlock&) = delete;
  MemBlock(MemBlock&& rvalue) {

    byte_size_ = rvalue.byte_size_;
    block_location_ = rvalue.block_location_;
    rvalue.byte_size_ = 0;
    rvalue.block_location_ = nullptr;

  }
  ~MemBlock() = default;

  MemBlock& operator=(const MemBlock&) = delete;

  [[nodiscard]] std::byte* address() const { return block_location_; }
  [[nodiscard]] size_t byteSize() const { return byte_size_; }

private:

  size_t byte_size_;
  std::byte* block_location_;

};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Host (CPU) Memory is aligned to a 32 byte boundary.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class GPUMemory; // Forward.

class HostMemory {

public:

  // Can only be std::moved, not copied or copy constructed.
  HostMemory() noexcept;
  explicit HostMemory(size_t byte_size) noexcept;
  HostMemory(HostMemory&& rvalue_copy) noexcept;
  HostMemory(const HostMemory&) = delete;
  ~HostMemory() noexcept;

  HostMemory& operator=(const HostMemory&) = delete;
  HostMemory& operator=(HostMemory&& copy) noexcept;

  [[nodiscard]] size_t byteSize() const { return byte_size_; }
  [[nodiscard]] MemBlock memInfo() const { return MemBlock(byte_size_, unique_host_ptr_); }
  // The offset and size must be within the allocated memory boundaries.
  // If not, an error is issued and the zero sized MemBlock is returned.
  // There is no alignment requirement for sub blocks.
  [[nodiscard]] MemBlock subBlock(size_t byte_offset, size_t byte_size) const;

  bool reallocateMemory(size_t byte_size) noexcept;

  // The Device memory block is resized (if required) and the host memory contents transferred to the GPU memory.
  bool transferToGPU(GPUMemory &gpu_memory) const;
  // The Device memory block is resized (if required) and the sub block (if valid) contents transferred to the GPU memory.
  bool transferSubBlockToGPU(size_t byte_offset, size_t byte_size, GPUMemory &gpu_memory) const;


private:

  std::byte* unique_host_ptr_;
  size_t byte_size_;
  // Aligned on a 32 byte boundary.
  std::align_val_t byte_alignment_ { 32 };

};



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU device memory held in a unique pointer to control ownership and avoid multiple deletion
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class DeviceMemory; // Forward declaration of the implementation class.
// Alignment requirements are assumed to be handled by the the implementation class.
class GPUMemory {

public:

  GPUMemory() noexcept;
  explicit GPUMemory(size_t byte_size) noexcept;
  GPUMemory(GPUMemory&& rvalue_copy) noexcept;
  GPUMemory(GPUMemory& copy) = delete;
  ~GPUMemory() noexcept;

  GPUMemory& operator=(GPUMemory& copy) = delete;
  GPUMemory& operator=(GPUMemory&& copy) noexcept;

  // Delete any existing memory allocation and reallocate a block of byte_size device memory.
  [[nodiscard]] size_t byteSize() const;
  [[nodiscard]] MemBlock memInfo() const { return MemBlock(byteSize(), getAddress()); }
  // The offset and size must be within the allocated memory boundaries.
  // If not, an error is issued and the zero sized MemBlock is returned.
  [[nodiscard]] MemBlock subBlock(size_t byte_offset, size_t byte_size) const;

  bool reallocateMemory(size_t byte_size) noexcept;

  // The host memory block is resized (if required) and the device memory contents transferred to the host memory.
  bool transferToHost(HostMemory &host_memory) const;
  // The Host memory block is resized (if required) and the sub block (if valid) contents transferred to the host memory.
  bool transferSubBlockToHost(size_t byte_offset, size_t byte_size, HostMemory &host_memory) const;
  // The address of the underlying device memory handle.

private:

  std::unique_ptr<DeviceMemory> unique_device_ptr_;

  [[nodiscard]] std::byte* getAddress() const;


};



} // namespace

#endif //KEG_CUDA_MEM_H
