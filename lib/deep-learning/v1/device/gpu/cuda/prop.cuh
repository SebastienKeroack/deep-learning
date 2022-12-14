/* Copyright 2016, 2019 Sébastien Kéroack. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

class cuDeviceProp {
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
  // Variable from "cudaDeviceProp" in "driver_types.h"

 protected:
  char p_name[256];

  bool p_device_overlap = false;  // [Deprecated]
  bool p_kernel_execute_timeout_enabled = false;
  bool p_integrated = false;
  bool p_can_map_host_memory = false;
  bool p_concurrent_kernels = false;
  bool p_ECC_enabled = false;
  bool p_TCC_driver = false;
  bool p_unified_addressing = false;
  bool p_stream_priorities_supported = false;
  bool p_global_L1_cache_supported = false;
  bool p_local_L1_cache_supported = false;
  bool p_managed_memory = false;
  bool p_is_multi_gpu_board = false;
  bool p_host_native_atomic_supported = false;
  bool p_pageable_memory_access = false;
  bool p_concurrent_managed_access = false;
  bool p_compute_preemption_supported = false;
  bool p_can_use_host_pointer_for_registered_memory = false;
  bool p_cooperative_launch = false;
  bool p_cooperative_multi_device_launch = false;

  int p_major_compute_capability = -1;
  int p_minor_compute_capability = -1;
  size_t p_warp_size = 0;
  size_t p_number_multiprocessor = 0;
  size_t p_maximum_threads_per_block = 0;
  size_t p_maximum_threads_per_multiprocessor = 0;
  size_t p_registers_per_block = 0;           // 32-bit
  size_t p_registers_per_multiprocessor = 0;  // 32-bit
  size_t p_maximum_threads_dimension[3];
  size_t p_maximum_grid_size[3];
  size_t p_clock_rate = 0;  // Kilohertz.
  int p_compute_mode = 0;
  size_t p_maximum_texture_1D = 0;
  size_t p_maximum_texture_1D_mipmap = 0;
  size_t p_maximum_texture_1D_linear = 0;
  size_t p_maximum_texture_2D[2];
  size_t p_maximum_texture_2D_mipmap[2];
  size_t p_maximum_texture_2D_linear[3];
  size_t p_maximum_texture_2D_gather[2];
  size_t p_maximum_texture_3D[3];
  size_t p_maximum_texture_3D_alternate[3];
  size_t p_maximum_texture_cubemap = 0;
  size_t p_maximum_texture_1D_layered[2];
  size_t p_maximum_texture_2D_layered[3];
  size_t p_maximum_texture_cubemap_layered[2];
  size_t p_maximum_surface_1D = 0;
  size_t p_maximum_surface_2D[2];
  size_t p_maximum_surface_3D[3];
  size_t p_maximum_surface_1D_layered[2];
  size_t p_maximum_surface_2D_layered[3];
  size_t p_maximum_surface_cubemap = 0;
  size_t p_maximum_surface_cubemap_layered[2];
  int p_PCI_bus_ID = 0;
  int p_PCI_device_ID = 0;
  int p_PCI_domain_ID = 0;
  size_t p_async_engine_count = 0;
  size_t p_memory_clock_rate = 0;  // Kilohertz.
  size_t p_memory_bus_width = 0;   // Bits.
  size_t p_L2_cache_size = 0;      // Bytes.
  int p_multi_gpu_board_group_ID = 0;
  int p_single_to_double_precision_performance_ratio = 0;

  size_t p_minimum_threads_for_occupancy = 0;
  size_t p_minimum_threads_for_occupancy_custom = 0;
  size_t p_maximum_number_threads = 0;
  size_t p_number_concurrent_kernel = 0;
  size_t p_number_CUDA_cores = 0;
  size_t p_number_CUDA_cores_per_multiprocessor = 0;
  size_t p_maximum_blocks_per_multiprocessor = 0;
  size_t p_maximum_number_warps_per_multiprocessor = 0;
  size_t p_number_shared_memory_banks = 0;

  size_t p_total_global_memory = 0_UZ;               // Bytes.
  size_t p_total_constant_memory = 0_UZ;             // Bytes.
  size_t p_shared_memory_per_block = 0_UZ;           // Bytes.
  size_t p_shared_memory_per_multiprocessor = 0_UZ;  // Bytes.
  size_t p_shared_memory_per_block_opt_in = 0_UZ;
  size_t p_memory_pitch = 0_UZ;  // Bytes.
  size_t p_texture_alignment = 0_UZ;
  size_t p_texture_pitch_alignment = 0_UZ;
  size_t p_surface_alignment = 0_UZ;

 public:
  __host__ __device__ cuDeviceProp(void) {}
  __host__ __device__ ~cuDeviceProp(void) {}

  __host__ __device__ class cuDeviceProp &operator=(
      class cuDeviceProp const
          &ref_source_CUDA_Device_Information_received);

  __host__ __device__ void copy(
      class cuDeviceProp const
          &ref_source_CUDA_Device_Information_received);
  __host__ __device__ void Grid_Block_1Dimensions(
      size_t const elements_received, size_t const limit_blocks_received,
      struct dim3 &ref_dim3_grid_received, struct dim3 &ref_dim3_block_received,
      size_t const registers_per_thread_received = 32u,
      size_t const shared_memory_per_block_received = 0,
      size_t const shared_memory_variable_per_block_received = 0u) const;
  __host__ __device__ void Grid_Block_2Dimensions(
      size_t const rows, size_t const cols,
      size_t const limit_blocks_received, struct dim3 &ref_dim3_grid_received,
      struct dim3 &ref_dim3_block_received,
      size_t const registers_per_thread_received = 32u,
      size_t const shared_memory_per_block_received = 0,
      size_t const shared_memory_variable_per_block_received = 0u) const;
  __host__ __device__ void Grid_Block_Transpose_2Dimensions(
      size_t const rows, size_t const cols,
      size_t const limit_blocks_received, struct dim3 &ref_dim3_grid_received,
      struct dim3 &ref_dim3_block_received,
      size_t const registers_per_thread_received = 32u,
      size_t const shared_memory_per_block_received = 0,
      size_t const shared_memory_variable_per_block_received = 0u) const;
  __host__ __device__ void Grid_Block_cuRAND_1Dimensions(
      size_t const elements_received, size_t limit_blocks_received,
      struct dim3 &ref_dim3_grid_received,
      struct dim3 &ref_dim3_block_received) const;
  __host__ __device__ void Grid_Block_Dynamic_Parallelisme(
      size_t const elements_received, size_t limit_blocks_received,
      struct dim3 &ref_dim3_grid_received,
      struct dim3 &ref_dim3_block_received) const;
  __host__ __device__ void Grid_Block_Reduce_1Dimensions(
      size_t const elements_received, size_t const limit_blocks_received,
      struct dim3 &ref_dim3_grid_received, struct dim3 &ref_dim3_block_received,
      size_t const registers_per_thread_received = 32u,
      size_t const shared_memory_per_block_received = 0,
      size_t const shared_memory_variable_per_block_received = 0u) const;
  __host__ __device__ void Grid_Block_Reduce_Dynamic_Parallelisme(
      size_t const elements_received, size_t const limit_blocks_received,
      struct dim3 &ref_dim3_grid_received,
      struct dim3 &ref_dim3_block_received) const;
  __host__ __device__ void Set__Minimum_Threads_For_Occupancy(
      size_t const minimum_threads_per_received);

  __host__ bool Initialize(size_t const index_device_received);
  __host__ __device__ bool Initialize(
      size_t const index_device_received,
      struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
  __host__ __device__ bool Get__Device_Overlap(void) const;  // [Deprecated]
  __host__ __device__ bool Get__Kernel_Execute_Timeout_Enabled(void) const;
  __host__ __device__ bool Get__Integrated(void) const;
  __host__ __device__ bool Get__Can_Map_Host_Memory(void) const;
  __host__ __device__ bool Get__Concurrent_Kernels(void) const;
  __host__ __device__ bool Get__ECC_Enabled(void) const;
  __host__ __device__ bool Get__TCC_Driver(void) const;
  __host__ __device__ bool Get__Unified_Addressing(void) const;
  __host__ __device__ bool Get__Stream_Priorities_Supported(void) const;
  __host__ __device__ bool Get__Global_L1_Cache_Supported(void) const;
  __host__ __device__ bool Get__Local_L1_Cache_Supported(void) const;
  __host__ __device__ bool Get__Managed_Memory(void) const;
  __host__ __device__ bool Get__Is_Multi_GPU_Board(void) const;
  __host__ __device__ bool Get__Host_Native_Atomic_Supported(void) const;
  __host__ __device__ bool Get__Pageable_Memory_Access(void) const;
  __host__ __device__ bool Get__Concurrent_Managed_Access(void) const;
  __host__ __device__ bool Get__Compute_Preemption_Supported(void) const;
  __host__ __device__ bool Get__Can_Use__Host_Pointer_For_Registered_Memory(
      void) const;
  __host__ __device__ bool Get__Cooperative_Launch(void) const;
  __host__ __device__ bool Get__Cooperative_Multi_Device_Launch(void) const;

  __host__ __device__ int Get__Major_Compute_Capability(void) const;
  __host__ __device__ int Get__Minor_Compute_Capability(void) const;
  __host__ __device__ size_t Get__Warp_Size(void) const;
  __host__ __device__ size_t Get__Number_Multiprocessor(void) const;
  __host__ __device__ size_t Get__Maximum_Threads_Per_Block(void) const;
  __host__ __device__ size_t
  Get__Maximum_Threads_Per_Multiprocessor(void) const;
  __host__ __device__ size_t Get__Registers_Per_Block(void) const;  // 32-bit
  __host__ __device__ size_t
  Get__Registers_Per_Multiprocessor(void) const;  // 32-bit
  __host__ __device__ size_t
  Get__Maximum_Threads_Dimension(size_t const index_received) const;
  __host__ __device__ size_t
  Get__Maximum_Grid_Size(size_t const index_received) const;
  __host__ __device__ size_t Get__Clock_Rate(void) const;  // Kilohertz.
  __host__ __device__ int Get__Compute_Mode(void) const;
  __host__ __device__ size_t Get__Maximum_Texture_1D(void) const;
  __host__ __device__ size_t Get__Maximum_Texture_1D_Mipmap(void) const;
  __host__ __device__ size_t Get__Maximum_Texture_1D_Linear(void) const;
  __host__ __device__ size_t
  Get__Maximum_Texture_2D(size_t const index_received) const;
  __host__ __device__ size_t
  Get__Maximum_Texture_2D_Mipmap(size_t const index_received) const;
  __host__ __device__ size_t
  Get__Maximum_Texture_2D_Linear(size_t const index_received) const;
  __host__ __device__ size_t
  Get__Maximum_Texture_2D_Gather(size_t const index_received) const;
  __host__ __device__ size_t
  Get__Maximum_Texture_3D(size_t const index_received) const;
  __host__ __device__ size_t
  Get__Maximum_Texture_3D_Alternate(size_t const index_received) const;
  __host__ __device__ size_t Get__Maximum_Texture_Cubemap(void) const;
  __host__ __device__ size_t
  Get__Maximum_Texture_1D_Layered(size_t const index_received) const;
  __host__ __device__ size_t
  Get__Maximum_Texture_2D_Layered(size_t const index_received) const;
  __host__ __device__ size_t
  Get__Maximum_Texture_Cubemap_Layered(size_t const index_received) const;
  __host__ __device__ size_t Get__Maximum_Surface_1D(void) const;
  __host__ __device__ size_t
  Get__Maximum_Surface_2D(size_t const index_received) const;
  __host__ __device__ size_t
  Get__Maximum_Surface_3D(size_t const index_received) const;
  __host__ __device__ size_t
  Get__Maximum_Surface_1D_Layered(size_t const index_received) const;
  __host__ __device__ size_t
  Get__Maximum_Surface_2D_Layered(size_t const index_received) const;
  __host__ __device__ size_t Get__Maximum_Surface_Cubemap(void) const;
  __host__ __device__ size_t
  Get__Maximum_Surface_Cubemap_Layered(size_t const index_received) const;
  __host__ __device__ int Get__PCI_Bus_ID(void) const;
  __host__ __device__ int Get__PCI_Device_ID(void) const;
  __host__ __device__ int Get__PCI_Domain_ID(void) const;
  __host__ __device__ size_t Get__Async_Engine_Count(void) const;
  __host__ __device__ size_t Get__Memory_Clock_Rate(void) const;  // Kilohertz.
  __host__ __device__ size_t Get__Memory_Bus_Width(void) const;   // Bits.
  __host__ __device__ size_t Get__L2_Cache_Size(void) const;      // Bytes.
  __host__ __device__ int Get__Multi_GPU_Board_Group_ID(void) const;
  __host__ __device__ int Get__Single_To_Double_Precision_Performance_Ratio(
      void) const;
  __host__ __device__ int Get__ID(void) const;

  __host__ __device__ size_t
  Get__Minimum_Threads_For_Occupancy(bool const use_default_received) const;
  __host__ __device__ size_t Get__Maximum_Threads(void) const;
  __host__ __device__ size_t
  Get__Number_Concurrent_Kernel_By_Compute_Capability(void) const;
  __host__ __device__ size_t Get__Number_Concurrent_Kernel(void) const;
  __host__ __device__ size_t
  Get__Number_CUDA_Cores_By_Compute_Capability(void) const;
  __host__ __device__ size_t CUDA__Number_CUDA_Cores(void) const;
  __host__ __device__ size_t
  Get__Number_CUDA_Cores_Per_Multiprocessor(void) const;
  __host__ __device__ size_t
  Get__Maximum_Blocks_Per_Multiprocessor_By_Compute_Capability(void) const;
  __host__ __device__ size_t Get__Maximum_Blocks_Per_Multiprocessor(void) const;
  __host__ __device__ size_t
  Get__Maximum_Warps_Per_Multiprocessor_By_Compute_Capability(void) const;
  __host__ __device__ size_t Get__Maximum_Warps_Per_Multiprocessor(void) const;
  __host__ __device__ size_t
  Get__Number_Shared_Memory_Banks_By_Compute_Capability(void) const;
  __host__ __device__ size_t Get__Number_Shared_Memory_Banks(void) const;
  __host__ __device__ size_t Get__Limit_Block_Due_To_Warp_Per_Multiprocessor(
      size_t const number_warps_received) const;

  __host__ __device__ size_t Get__Total_Global_Memory(void) const;
  __host__ __device__ size_t Get__Total_Constant_Memory(void) const;
  __host__ __device__ size_t Get__Shared_Memory_Per_Block(void) const;
  __host__ __device__ size_t Get__Shared_Memory_Per_Multiprocessor(void) const;
  __host__ __device__ size_t Get__Shared_Memory_Per_Block_Opt_In(void) const;
  __host__ __device__ size_t Get__Memory_Pitch(void) const;
  __host__ __device__ size_t Get__Texture_Alignment(void) const;
  __host__ __device__ size_t Get__Texture_Pitch_Alignment(void) const;
  __host__ __device__ size_t Get__Surface_Alignment(void) const;

  __host__ __device__ double OccupencyOfEachMultiprocessor(
      size_t const thread_count_received,
      size_t const registers_per_thread_received = 32u,
      size_t const shared_memory_per_block_received = 0u) const;

 private:
  int _ID = -1;
};

class cuDevicesProp {
 public:
  __host__ __device__ cuDevicesProp(void);
  __host__ __device__ ~cuDevicesProp(void);

  __host__ bool push_back(int const index_device_received);
  __host__ __device__ bool push_back(
      int const index_device_received,
      struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
  __host__ __device__ bool Update(
      struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
  __host__ __device__ bool Select_CUDA_Device(int const index_received);
  __host__ __device__ bool Deallocate(void);

  __host__ __device__ size_t Get__Number_CUDA_Devices(void) const;
  __host__ __device__ int Get__Selected_CUDA_Device(void) const;

  __host__ __device__ class cuDeviceProp *Get__CUDA_Device(
      void) const;
  __host__ __device__ class cuDeviceProp *Get__CUDA_Device(
      size_t const index_received) const;

 private:
  size_t _number_cuda_devices = 0;

  int _selected_cuda_device = -1;

  class cuDeviceProp *_ptr_Class_Device_Information_sum = nullptr;
  class cuDeviceProp *_ptr_Class_Device_Information_higher = nullptr;
  class cuDeviceProp *_ptr_Class_Device_Information_lower = nullptr;
  class cuDeviceProp *_ptr_array_Class_Device_Information = nullptr;
};
