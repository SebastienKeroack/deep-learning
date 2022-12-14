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

__host__ __device__ class cuDeviceProp &
cuDeviceProp::operator=(
    class cuDeviceProp const
        &ref_source_CUDA_Device_Information_received) {
  if (&ref_source_CUDA_Device_Information_received != this) {
    this->copy(ref_source_CUDA_Device_Information_received);
  }

  return *this;
}

__host__ __device__ void cuDeviceProp::copy(
    class cuDeviceProp const
        &ref_source_CUDA_Device_Information_received) {
  Memory::Copy_Loop(ref_source_CUDA_Device_Information_received.p_name,
                    ref_source_CUDA_Device_Information_received.p_name + 256,
                    this->p_name);

  this->p_device_overlap =
      ref_source_CUDA_Device_Information_received.p_device_overlap;
  this->p_kernel_execute_timeout_enabled =
      ref_source_CUDA_Device_Information_received
          .p_kernel_execute_timeout_enabled;
  this->p_integrated = ref_source_CUDA_Device_Information_received.p_integrated;
  this->p_can_map_host_memory =
      ref_source_CUDA_Device_Information_received.p_can_map_host_memory;
  this->p_concurrent_kernels =
      ref_source_CUDA_Device_Information_received.p_concurrent_kernels;
  this->p_ECC_enabled =
      ref_source_CUDA_Device_Information_received.p_ECC_enabled;
  this->p_TCC_driver = ref_source_CUDA_Device_Information_received.p_TCC_driver;
  this->p_unified_addressing =
      ref_source_CUDA_Device_Information_received.p_unified_addressing;
  this->p_stream_priorities_supported =
      ref_source_CUDA_Device_Information_received.p_stream_priorities_supported;
  this->p_global_L1_cache_supported =
      ref_source_CUDA_Device_Information_received.p_global_L1_cache_supported;
  this->p_local_L1_cache_supported =
      ref_source_CUDA_Device_Information_received.p_local_L1_cache_supported;
  this->p_managed_memory =
      ref_source_CUDA_Device_Information_received.p_managed_memory;
  this->p_is_multi_gpu_board =
      ref_source_CUDA_Device_Information_received.p_is_multi_gpu_board;
  this->p_host_native_atomic_supported =
      ref_source_CUDA_Device_Information_received
          .p_host_native_atomic_supported;
  this->p_pageable_memory_access =
      ref_source_CUDA_Device_Information_received.p_pageable_memory_access;
  this->p_concurrent_managed_access =
      ref_source_CUDA_Device_Information_received.p_concurrent_managed_access;
  this->p_compute_preemption_supported =
      ref_source_CUDA_Device_Information_received
          .p_compute_preemption_supported;
  this->p_can_use_host_pointer_for_registered_memory =
      ref_source_CUDA_Device_Information_received
          .p_can_use_host_pointer_for_registered_memory;
  this->p_cooperative_launch =
      ref_source_CUDA_Device_Information_received.p_cooperative_launch;
  this->p_cooperative_multi_device_launch =
      ref_source_CUDA_Device_Information_received
          .p_cooperative_multi_device_launch;

  this->p_major_compute_capability =
      ref_source_CUDA_Device_Information_received.p_major_compute_capability;
  this->p_minor_compute_capability =
      ref_source_CUDA_Device_Information_received.p_minor_compute_capability;
  this->p_warp_size = ref_source_CUDA_Device_Information_received.p_warp_size;
  this->p_number_multiprocessor =
      ref_source_CUDA_Device_Information_received.p_number_multiprocessor;
  this->p_maximum_threads_per_block =
      ref_source_CUDA_Device_Information_received.p_maximum_threads_per_block;
  this->p_maximum_threads_per_multiprocessor =
      ref_source_CUDA_Device_Information_received
          .p_maximum_threads_per_multiprocessor;
  this->p_registers_per_block = ref_source_CUDA_Device_Information_received
                                    .p_registers_per_block;  // 32-bit
  this->p_registers_per_multiprocessor =
      ref_source_CUDA_Device_Information_received
          .p_registers_per_multiprocessor;  // 32-bit

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_threads_dimension,
      ref_source_CUDA_Device_Information_received.p_maximum_threads_dimension +
          3,
      this->p_maximum_threads_dimension);

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_grid_size,
      ref_source_CUDA_Device_Information_received.p_maximum_grid_size + 3,
      this->p_maximum_grid_size);

  this->p_clock_rate =
      ref_source_CUDA_Device_Information_received.p_clock_rate;  // Kilohertz.
  this->p_compute_mode =
      ref_source_CUDA_Device_Information_received.p_compute_mode;
  this->p_maximum_texture_1D =
      ref_source_CUDA_Device_Information_received.p_maximum_texture_1D;
  this->p_maximum_texture_1D_mipmap =
      ref_source_CUDA_Device_Information_received.p_maximum_texture_1D_mipmap;
  this->p_maximum_texture_1D_linear =
      ref_source_CUDA_Device_Information_received.p_maximum_texture_1D_linear;

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_texture_2D,
      ref_source_CUDA_Device_Information_received.p_maximum_texture_2D + 2,
      this->p_maximum_texture_2D);

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_texture_2D_mipmap,
      ref_source_CUDA_Device_Information_received.p_maximum_texture_2D_mipmap +
          2,
      this->p_maximum_texture_2D_mipmap);

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_texture_2D_linear,
      ref_source_CUDA_Device_Information_received.p_maximum_texture_2D_linear +
          3,
      this->p_maximum_texture_2D_linear);

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_texture_2D_gather,
      ref_source_CUDA_Device_Information_received.p_maximum_texture_2D_gather +
          2,
      this->p_maximum_texture_2D_gather);

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_texture_3D,
      ref_source_CUDA_Device_Information_received.p_maximum_texture_3D + 3,
      this->p_maximum_texture_3D);

  Memory::Copy_Loop(ref_source_CUDA_Device_Information_received
                        .p_maximum_texture_3D_alternate,
                    ref_source_CUDA_Device_Information_received
                            .p_maximum_texture_3D_alternate +
                        3,
                    this->p_maximum_texture_3D_alternate);

  this->p_maximum_texture_cubemap =
      ref_source_CUDA_Device_Information_received.p_maximum_texture_cubemap;

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_texture_1D_layered,
      ref_source_CUDA_Device_Information_received.p_maximum_texture_1D_layered +
          2,
      this->p_maximum_texture_1D_layered);

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_texture_2D_layered,
      ref_source_CUDA_Device_Information_received.p_maximum_texture_2D_layered +
          3,
      this->p_maximum_texture_2D_layered);

  Memory::Copy_Loop(ref_source_CUDA_Device_Information_received
                        .p_maximum_texture_cubemap_layered,
                    ref_source_CUDA_Device_Information_received
                            .p_maximum_texture_cubemap_layered +
                        2,
                    this->p_maximum_texture_cubemap_layered);

  this->p_maximum_surface_1D =
      ref_source_CUDA_Device_Information_received.p_maximum_surface_1D;

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_surface_2D,
      ref_source_CUDA_Device_Information_received.p_maximum_surface_2D + 2,
      this->p_maximum_surface_2D);

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_surface_3D,
      ref_source_CUDA_Device_Information_received.p_maximum_surface_3D + 3,
      this->p_maximum_surface_3D);

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_surface_1D_layered,
      ref_source_CUDA_Device_Information_received.p_maximum_surface_1D_layered +
          2,
      this->p_maximum_surface_1D_layered);

  Memory::Copy_Loop(
      ref_source_CUDA_Device_Information_received.p_maximum_surface_2D_layered,
      ref_source_CUDA_Device_Information_received.p_maximum_surface_2D_layered +
          3,
      this->p_maximum_surface_2D_layered);

  this->p_maximum_surface_cubemap =
      ref_source_CUDA_Device_Information_received.p_maximum_surface_cubemap;

  Memory::Copy_Loop(ref_source_CUDA_Device_Information_received
                        .p_maximum_surface_cubemap_layered,
                    ref_source_CUDA_Device_Information_received
                            .p_maximum_surface_cubemap_layered +
                        2,
                    this->p_maximum_surface_cubemap_layered);

  this->p_PCI_bus_ID = ref_source_CUDA_Device_Information_received.p_PCI_bus_ID;
  this->p_PCI_device_ID =
      ref_source_CUDA_Device_Information_received.p_PCI_device_ID;
  this->p_PCI_domain_ID =
      ref_source_CUDA_Device_Information_received.p_PCI_domain_ID;
  this->p_async_engine_count =
      ref_source_CUDA_Device_Information_received.p_async_engine_count;
  this->p_memory_clock_rate = ref_source_CUDA_Device_Information_received
                                  .p_memory_clock_rate;  // Kilohertz.
  this->p_memory_bus_width =
      ref_source_CUDA_Device_Information_received.p_memory_bus_width;  // Bits.
  this->p_L2_cache_size =
      ref_source_CUDA_Device_Information_received.p_L2_cache_size;  // Bytes.
  this->p_multi_gpu_board_group_ID =
      ref_source_CUDA_Device_Information_received.p_multi_gpu_board_group_ID;
  this->p_single_to_double_precision_performance_ratio =
      ref_source_CUDA_Device_Information_received
          .p_single_to_double_precision_performance_ratio;

  this->p_minimum_threads_for_occupancy =
      ref_source_CUDA_Device_Information_received
          .p_minimum_threads_for_occupancy;
  this->p_minimum_threads_for_occupancy_custom =
      ref_source_CUDA_Device_Information_received
          .p_minimum_threads_for_occupancy_custom;
  this->p_maximum_number_threads =
      ref_source_CUDA_Device_Information_received.p_maximum_number_threads;
  this->p_number_concurrent_kernel =
      ref_source_CUDA_Device_Information_received.p_number_concurrent_kernel;
  this->p_number_CUDA_cores =
      ref_source_CUDA_Device_Information_received.p_number_CUDA_cores;
  this->p_number_CUDA_cores_per_multiprocessor =
      ref_source_CUDA_Device_Information_received
          .p_number_CUDA_cores_per_multiprocessor;
  this->p_maximum_blocks_per_multiprocessor =
      ref_source_CUDA_Device_Information_received
          .p_maximum_blocks_per_multiprocessor;
  this->p_maximum_number_warps_per_multiprocessor =
      ref_source_CUDA_Device_Information_received
          .p_maximum_number_warps_per_multiprocessor;
  this->p_number_shared_memory_banks =
      ref_source_CUDA_Device_Information_received.p_number_shared_memory_banks;

  this->p_total_global_memory = ref_source_CUDA_Device_Information_received
                                    .p_total_global_memory;  // Bytes.
  this->p_total_constant_memory = ref_source_CUDA_Device_Information_received
                                      .p_total_constant_memory;  // Bytes.
  this->p_shared_memory_per_block = ref_source_CUDA_Device_Information_received
                                        .p_shared_memory_per_block;  // Bytes.
  this->p_shared_memory_per_multiprocessor =
      ref_source_CUDA_Device_Information_received
          .p_shared_memory_per_multiprocessor;  // Bytes.
  this->p_shared_memory_per_block_opt_in =
      ref_source_CUDA_Device_Information_received
          .p_shared_memory_per_block_opt_in;
  this->p_memory_pitch =
      ref_source_CUDA_Device_Information_received.p_memory_pitch;  // Bytes.
  this->p_texture_alignment =
      ref_source_CUDA_Device_Information_received.p_texture_alignment;
  this->p_texture_pitch_alignment =
      ref_source_CUDA_Device_Information_received.p_texture_pitch_alignment;
  this->p_surface_alignment =
      ref_source_CUDA_Device_Information_received.p_surface_alignment;
}

__global__ void kernel__Class_Device_Information__Initialize(
    size_t const index_device_received,
    struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received,
    class cuDeviceProp
        *const ptr_Class_Device_Information_received) {
  ptr_Class_Device_Information_received->Initialize(
      index_device_received, ptr_struct_cudaDeviceProp_received);
}

__host__ bool cuDeviceProp::Initialize(
    size_t const index_device_received) {
  struct cudaDeviceProp tmp_struct_cudaDeviceProp,
      *tmp_ptr_device_struct_cudaDeviceProp(NULL);

  CUDA__Safe_Call(cudaGetDeviceProperties(
      &tmp_struct_cudaDeviceProp, static_cast<int>(index_device_received)));

  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_struct_cudaDeviceProp,
                             sizeof(struct cudaDeviceProp)));

  CUDA__Safe_Call(cudaMemcpy(
      tmp_ptr_device_struct_cudaDeviceProp, &tmp_struct_cudaDeviceProp,
      sizeof(struct cudaDeviceProp), cudaMemcpyKind::cudaMemcpyHostToDevice));

  kernel__Class_Device_Information__Initialize<<<1, 1u>>>(
      index_device_received, tmp_ptr_device_struct_cudaDeviceProp, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_struct_cudaDeviceProp));

  return true;
}

__host__ __device__ bool cuDeviceProp::Initialize(
    size_t const index_device_received,
    struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received) {
  if (ptr_struct_cudaDeviceProp_received == nullptr) {
    return false;
  }

#ifndef __CUDA_ARCH__
  kernel__Class_Device_Information__Initialize<<<1, 1u>>>(
      index_device_received, ptr_struct_cudaDeviceProp_received, this);

  CUDA__Check_Error();
#else
  size_t tmp_index;

  this->_ID = index_device_received;

  Memory::Copy_Loop(ptr_struct_cudaDeviceProp_received->name,
                    ptr_struct_cudaDeviceProp_received->name + 256,
                    this->p_name);

  this->p_device_overlap =
      ptr_struct_cudaDeviceProp_received->deviceOverlap;  // [Deprecated]

  this->p_kernel_execute_timeout_enabled =
      ptr_struct_cudaDeviceProp_received->kernelExecTimeoutEnabled;

  this->p_integrated = ptr_struct_cudaDeviceProp_received->integrated;

  this->p_can_map_host_memory =
      ptr_struct_cudaDeviceProp_received->canMapHostMemory;

  this->p_concurrent_kernels =
      ptr_struct_cudaDeviceProp_received->concurrentKernels;

  this->p_ECC_enabled = ptr_struct_cudaDeviceProp_received->ECCEnabled;

  this->p_TCC_driver = ptr_struct_cudaDeviceProp_received->tccDriver;

  this->p_unified_addressing =
      ptr_struct_cudaDeviceProp_received->unifiedAddressing;

  this->p_stream_priorities_supported =
      ptr_struct_cudaDeviceProp_received->streamPrioritiesSupported;

  this->p_global_L1_cache_supported =
      ptr_struct_cudaDeviceProp_received->globalL1CacheSupported;

  this->p_local_L1_cache_supported =
      ptr_struct_cudaDeviceProp_received->localL1CacheSupported;

  this->p_managed_memory = ptr_struct_cudaDeviceProp_received->managedMemory;

  this->p_is_multi_gpu_board =
      ptr_struct_cudaDeviceProp_received->isMultiGpuBoard;

  this->p_host_native_atomic_supported =
      ptr_struct_cudaDeviceProp_received->hostNativeAtomicSupported;

  this->p_pageable_memory_access =
      ptr_struct_cudaDeviceProp_received->pageableMemoryAccess;

  this->p_concurrent_managed_access =
      ptr_struct_cudaDeviceProp_received->concurrentManagedAccess;

  this->p_compute_preemption_supported =
      ptr_struct_cudaDeviceProp_received->computeMode;

  this->p_can_use_host_pointer_for_registered_memory =
      ptr_struct_cudaDeviceProp_received->canUseHostPointerForRegisteredMem;

  this->p_cooperative_launch =
      ptr_struct_cudaDeviceProp_received->cooperativeLaunch;

  this->p_cooperative_multi_device_launch =
      ptr_struct_cudaDeviceProp_received->cooperativeMultiDeviceLaunch;

  this->p_major_compute_capability = ptr_struct_cudaDeviceProp_received->major;

  this->p_minor_compute_capability = ptr_struct_cudaDeviceProp_received->minor;

  this->p_warp_size =
      static_cast<size_t>(ptr_struct_cudaDeviceProp_received->warpSize);

  this->p_number_multiprocessor = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->multiProcessorCount);

  this->p_maximum_number_threads = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->multiProcessorCount *
      ptr_struct_cudaDeviceProp_received->maxThreadsPerMultiProcessor);

  this->p_maximum_threads_per_block = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->maxThreadsPerBlock);

  this->p_maximum_threads_per_multiprocessor = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->maxThreadsPerMultiProcessor);

  this->p_registers_per_block = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->regsPerBlock);  // 32-bit

  this->p_registers_per_multiprocessor = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->regsPerMultiprocessor);  // 32-bit

  for (tmp_index = 0_UZ; tmp_index != 3_UZ; ++tmp_index) {
    this->p_maximum_threads_dimension[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxThreadsDim[tmp_index]);
  }

  for (tmp_index = 0_UZ; tmp_index != 3_UZ; ++tmp_index) {
    this->p_maximum_grid_size[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxGridSize[tmp_index]);
  }

  this->p_clock_rate = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->clockRate);  // Kilohertz.

  this->p_compute_mode = ptr_struct_cudaDeviceProp_received->computeMode;

  this->p_maximum_texture_1D =
      static_cast<size_t>(ptr_struct_cudaDeviceProp_received->maxTexture1D);

  this->p_maximum_texture_1D_mipmap = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->maxTexture1DMipmap);

  this->p_maximum_texture_1D_linear = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->maxTexture1DLinear);

  for (tmp_index = 0_UZ; tmp_index != 2_UZ; ++tmp_index) {
    this->p_maximum_texture_2D[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxTexture2D[tmp_index]);
  }

  for (tmp_index = 0_UZ; tmp_index != 2_UZ; ++tmp_index) {
    this->p_maximum_texture_2D_mipmap[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxTexture2DMipmap[tmp_index]);
  }

  for (tmp_index = 0_UZ; tmp_index != 3_UZ; ++tmp_index) {
    this->p_maximum_texture_2D_linear[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxTexture2DLinear[tmp_index]);
  }

  for (tmp_index = 0_UZ; tmp_index != 2_UZ; ++tmp_index) {
    this->p_maximum_texture_2D_gather[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxTexture2DGather[tmp_index]);
  }

  for (tmp_index = 0_UZ; tmp_index != 3_UZ; ++tmp_index) {
    this->p_maximum_texture_3D[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxTexture3D[tmp_index]);
  }

  for (tmp_index = 0_UZ; tmp_index != 3_UZ; ++tmp_index) {
    this->p_maximum_texture_3D_alternate[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxTexture3DAlt[tmp_index]);
  }

  this->p_maximum_texture_cubemap = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->maxTextureCubemap);

  for (tmp_index = 0_UZ; tmp_index != 2_UZ; ++tmp_index) {
    this->p_maximum_texture_1D_layered[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxTexture1DLayered[tmp_index]);
  }

  for (tmp_index = 0_UZ; tmp_index != 3_UZ; ++tmp_index) {
    this->p_maximum_texture_2D_layered[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxTexture2DLayered[tmp_index]);
  }

  for (tmp_index = 0_UZ; tmp_index != 2_UZ; ++tmp_index) {
    this->p_maximum_texture_cubemap_layered[tmp_index] =
        static_cast<size_t>(ptr_struct_cudaDeviceProp_received
                                ->maxTextureCubemapLayered[tmp_index]);
  }

  this->p_maximum_surface_1D =
      static_cast<size_t>(ptr_struct_cudaDeviceProp_received->maxSurface1D);

  for (tmp_index = 0_UZ; tmp_index != 2_UZ; ++tmp_index) {
    this->p_maximum_surface_2D[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxSurface2D[tmp_index]);
  }

  for (tmp_index = 0_UZ; tmp_index != 3_UZ; ++tmp_index) {
    this->p_maximum_surface_3D[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxSurface3D[tmp_index]);
  }

  for (tmp_index = 0_UZ; tmp_index != 2_UZ; ++tmp_index) {
    this->p_maximum_surface_1D_layered[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxSurface1DLayered[tmp_index]);
  }

  for (tmp_index = 0_UZ; tmp_index != 3_UZ; ++tmp_index) {
    this->p_maximum_surface_2D_layered[tmp_index] = static_cast<size_t>(
        ptr_struct_cudaDeviceProp_received->maxSurface2DLayered[tmp_index]);
  }

  this->p_maximum_surface_cubemap = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->maxSurfaceCubemap);

  for (tmp_index = 0_UZ; tmp_index != 2_UZ; ++tmp_index) {
    this->p_maximum_surface_cubemap_layered[tmp_index] =
        static_cast<size_t>(ptr_struct_cudaDeviceProp_received
                                ->maxSurfaceCubemapLayered[tmp_index]);
  }

  this->p_PCI_bus_ID = ptr_struct_cudaDeviceProp_received->pciBusID;

  this->p_PCI_device_ID = ptr_struct_cudaDeviceProp_received->pciDeviceID;

  this->p_PCI_domain_ID = ptr_struct_cudaDeviceProp_received->pciDomainID;

  this->p_async_engine_count =
      static_cast<size_t>(ptr_struct_cudaDeviceProp_received->asyncEngineCount);

  this->p_memory_clock_rate = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->memoryClockRate);  // Kilohertz.

  this->p_memory_bus_width = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->memoryBusWidth);  // Bits.

  this->p_L2_cache_size = static_cast<size_t>(
      ptr_struct_cudaDeviceProp_received->l2CacheSize);  // Bytes.

  this->p_multi_gpu_board_group_ID =
      ptr_struct_cudaDeviceProp_received->multiGpuBoardGroupID;

  this->p_single_to_double_precision_performance_ratio =
      ptr_struct_cudaDeviceProp_received->singleToDoublePrecisionPerfRatio;

  this->p_total_global_memory =
      ptr_struct_cudaDeviceProp_received->totalGlobalMem;  // Bytes.

  this->p_total_constant_memory =
      ptr_struct_cudaDeviceProp_received->totalConstMem;  // Bytes.

  this->p_shared_memory_per_block =
      ptr_struct_cudaDeviceProp_received->sharedMemPerBlock;  // Bytes.

  this->p_shared_memory_per_multiprocessor =
      ptr_struct_cudaDeviceProp_received->sharedMemPerMultiprocessor;  // Bytes.

  this->p_shared_memory_per_block_opt_in =
      ptr_struct_cudaDeviceProp_received->sharedMemPerBlockOptin;

  this->p_memory_pitch =
      ptr_struct_cudaDeviceProp_received->memPitch;  // Bytes.

  this->p_texture_alignment =
      ptr_struct_cudaDeviceProp_received->textureAlignment;

  this->p_texture_pitch_alignment =
      ptr_struct_cudaDeviceProp_received->texturePitchAlignment;

  this->p_surface_alignment =
      ptr_struct_cudaDeviceProp_received->surfaceAlignment;

  this->p_number_concurrent_kernel =
      this->Get__Number_Concurrent_Kernel_By_Compute_Capability();

  this->p_number_CUDA_cores =
      this->Get__Number_CUDA_Cores_By_Compute_Capability();

  this->p_number_CUDA_cores_per_multiprocessor =
      this->p_number_CUDA_cores / this->p_number_multiprocessor;

  this->p_maximum_blocks_per_multiprocessor =
      this->Get__Maximum_Blocks_Per_Multiprocessor_By_Compute_Capability();

  this->p_minimum_threads_for_occupancy_custom =
      this->p_minimum_threads_for_occupancy =
          ptr_struct_cudaDeviceProp_received->maxThreadsPerMultiProcessor /
          this->p_maximum_blocks_per_multiprocessor;

  this->p_maximum_number_warps_per_multiprocessor =
      this->Get__Maximum_Warps_Per_Multiprocessor_By_Compute_Capability();

  this->p_number_shared_memory_banks =
      this->Get__Number_Shared_Memory_Banks_By_Compute_Capability();
#endif

  return true;
}

__host__ __device__ bool cuDeviceProp::Get__Device_Overlap(
    void) const {
  return (this->p_device_overlap);
}

__host__ __device__ bool
cuDeviceProp::Get__Kernel_Execute_Timeout_Enabled(void) const {
  return (this->p_kernel_execute_timeout_enabled);
}

__host__ __device__ bool cuDeviceProp::Get__Integrated(void) const {
  return (this->p_integrated);
}

__host__ __device__ bool cuDeviceProp::Get__Can_Map_Host_Memory(
    void) const {
  return (this->p_can_map_host_memory);
}

__host__ __device__ bool cuDeviceProp::Get__Concurrent_Kernels(
    void) const {
  return (this->p_concurrent_kernels);
}

__host__ __device__ bool cuDeviceProp::Get__ECC_Enabled(void) const {
  return (this->p_ECC_enabled);
}

__host__ __device__ bool cuDeviceProp::Get__TCC_Driver(void) const {
  return (this->p_TCC_driver);
}

__host__ __device__ bool cuDeviceProp::Get__Unified_Addressing(
    void) const {
  return (this->p_unified_addressing);
}

__host__ __device__ bool
cuDeviceProp::Get__Stream_Priorities_Supported(void) const {
  return (this->p_stream_priorities_supported);
}

__host__ __device__ bool
cuDeviceProp::Get__Global_L1_Cache_Supported(void) const {
  return (this->p_global_L1_cache_supported);
}

__host__ __device__ bool cuDeviceProp::Get__Local_L1_Cache_Supported(
    void) const {
  return (this->p_local_L1_cache_supported);
}

__host__ __device__ bool cuDeviceProp::Get__Managed_Memory(
    void) const {
  return (this->p_managed_memory);
}

__host__ __device__ bool cuDeviceProp::Get__Is_Multi_GPU_Board(
    void) const {
  return (this->p_is_multi_gpu_board);
}

__host__ __device__ bool
cuDeviceProp::Get__Host_Native_Atomic_Supported(void) const {
  return (this->p_host_native_atomic_supported);
}

__host__ __device__ bool cuDeviceProp::Get__Pageable_Memory_Access(
    void) const {
  return (this->p_pageable_memory_access);
}

__host__ __device__ bool
cuDeviceProp::Get__Concurrent_Managed_Access(void) const {
  return (this->p_concurrent_managed_access);
}

__host__ __device__ bool
cuDeviceProp::Get__Compute_Preemption_Supported(void) const {
  return (this->p_compute_preemption_supported);
}

__host__ __device__ bool
cuDeviceProp::Get__Can_Use__Host_Pointer_For_Registered_Memory(
    void) const {
  return (this->p_can_use_host_pointer_for_registered_memory);
}

__host__ __device__ bool cuDeviceProp::Get__Cooperative_Launch(
    void) const {
  return (this->p_cooperative_launch);
}

__host__ __device__ bool
cuDeviceProp::Get__Cooperative_Multi_Device_Launch(void) const {
  return (this->p_cooperative_multi_device_launch);
}

__host__ __device__ int cuDeviceProp::Get__Major_Compute_Capability(
    void) const {
  return (this->p_major_compute_capability);
}

__host__ __device__ int cuDeviceProp::Get__Minor_Compute_Capability(
    void) const {
  return (this->p_minor_compute_capability);
}

__host__ __device__ size_t cuDeviceProp::Get__Warp_Size(void) const {
  return (this->p_warp_size);
}

__host__ __device__ size_t
cuDeviceProp::Get__Number_Multiprocessor(void) const {
  return (this->p_number_multiprocessor);
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Threads_Per_Block(void) const {
  return (this->p_maximum_threads_per_block);
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Threads_Per_Multiprocessor(void) const {
  return (this->p_maximum_threads_per_multiprocessor);
}

__host__ __device__ size_t
cuDeviceProp::Get__Registers_Per_Block(void) const {
  return (this->p_registers_per_block);
}  // 32-bit

__host__ __device__ size_t
cuDeviceProp::Get__Registers_Per_Multiprocessor(void) const {
  return (this->p_registers_per_multiprocessor);
}  // 32-bit

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Threads_Dimension(
    size_t const index_received) const {
  if (index_received < 3_UZ) {
    return (this->p_maximum_threads_dimension[index_received]);
  } else {
    return (this->p_maximum_threads_dimension[0]);
  }
}

__host__ __device__ size_t cuDeviceProp::Get__Maximum_Grid_Size(
    size_t const index_received) const {
  if (index_received < 3_UZ) {
    return (this->p_maximum_grid_size[index_received]);
  } else {
    return (this->p_maximum_grid_size[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Clock_Rate(void) const {
  return (this->p_clock_rate);
}  // Kilohertz.

__host__ __device__ int cuDeviceProp::Get__Compute_Mode(void) const {
  return (this->p_compute_mode);
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Texture_1D(void) const {
  return (this->p_maximum_texture_1D);
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Texture_1D_Mipmap(void) const {
  return (this->p_maximum_texture_1D_mipmap);
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Texture_1D_Linear(void) const {
  return (this->p_maximum_texture_1D_linear);
}

__host__ __device__ size_t cuDeviceProp::Get__Maximum_Texture_2D(
    size_t const index_received) const {
  if (index_received < 2_UZ) {
    return (this->p_maximum_texture_2D[index_received]);
  } else {
    return (this->p_maximum_texture_2D[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Texture_2D_Mipmap(
    size_t const index_received) const {
  if (index_received < 2_UZ) {
    return (this->p_maximum_texture_2D_mipmap[index_received]);
  } else {
    return (this->p_maximum_texture_2D_mipmap[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Texture_2D_Linear(
    size_t const index_received) const {
  if (index_received < 2_UZ) {
    return (this->p_maximum_texture_2D_linear[index_received]);
  } else {
    return (this->p_maximum_texture_2D_linear[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Texture_2D_Gather(
    size_t const index_received) const {
  if (index_received < 2_UZ) {
    return (this->p_maximum_texture_2D_gather[index_received]);
  } else {
    return (this->p_maximum_texture_2D_gather[0]);
  }
}

__host__ __device__ size_t cuDeviceProp::Get__Maximum_Texture_3D(
    size_t const index_received) const {
  if (index_received < 3_UZ) {
    return (this->p_maximum_texture_3D[index_received]);
  } else {
    return (this->p_maximum_texture_3D[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Texture_3D_Alternate(
    size_t const index_received) const {
  if (index_received < 3_UZ) {
    return (this->p_maximum_texture_3D_alternate[index_received]);
  } else {
    return (this->p_maximum_texture_3D_alternate[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Texture_Cubemap(void) const {
  return (this->p_maximum_texture_cubemap);
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Texture_1D_Layered(
    size_t const index_received) const {
  if (index_received < 2_UZ) {
    return (this->p_maximum_texture_1D_layered[index_received]);
  } else {
    return (this->p_maximum_texture_1D_layered[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Texture_2D_Layered(
    size_t const index_received) const {
  if (index_received < 3_UZ) {
    return (this->p_maximum_texture_2D_layered[index_received]);
  } else {
    return (this->p_maximum_texture_2D_layered[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Texture_Cubemap_Layered(
    size_t const index_received) const {
  if (index_received < 2_UZ) {
    return (this->p_maximum_texture_cubemap_layered[index_received]);
  } else {
    return (this->p_maximum_texture_cubemap_layered[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Surface_1D(void) const {
  return (this->p_maximum_surface_1D);
}

__host__ __device__ size_t cuDeviceProp::Get__Maximum_Surface_2D(
    size_t const index_received) const {
  if (index_received < 2_UZ) {
    return (this->p_maximum_surface_2D[index_received]);
  } else {
    return (this->p_maximum_surface_2D[0]);
  }
}

__host__ __device__ size_t cuDeviceProp::Get__Maximum_Surface_3D(
    size_t const index_received) const {
  if (index_received < 3_UZ) {
    return (this->p_maximum_surface_3D[index_received]);
  } else {
    return (this->p_maximum_surface_3D[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Surface_1D_Layered(
    size_t const index_received) const {
  if (index_received < 2_UZ) {
    return (this->p_maximum_surface_1D_layered[index_received]);
  } else {
    return (this->p_maximum_surface_1D_layered[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Surface_2D_Layered(
    size_t const index_received) const {
  if (index_received < 3_UZ) {
    return (this->p_maximum_surface_2D_layered[index_received]);
  } else {
    return (this->p_maximum_surface_2D_layered[0]);
  }
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Surface_Cubemap(void) const {
  return (this->p_maximum_surface_cubemap);
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Surface_Cubemap_Layered(
    size_t const index_received) const {
  if (index_received < 2_UZ) {
    return (this->p_maximum_surface_cubemap_layered[index_received]);
  } else {
    return (this->p_maximum_surface_cubemap_layered[0]);
  }
}

__host__ __device__ int cuDeviceProp::Get__PCI_Bus_ID(void) const {
  return (this->p_PCI_bus_ID);
}

__host__ __device__ int cuDeviceProp::Get__PCI_Device_ID(
    void) const {
  return (this->p_PCI_device_ID);
}

__host__ __device__ int cuDeviceProp::Get__PCI_Domain_ID(
    void) const {
  return (this->p_PCI_domain_ID);
}

__host__ __device__ size_t
cuDeviceProp::Get__Async_Engine_Count(void) const {
  return (this->p_async_engine_count);
}

__host__ __device__ size_t
cuDeviceProp::Get__Memory_Clock_Rate(void) const {
  return (this->p_memory_clock_rate);
}  // Kilohertz.

__host__ __device__ size_t
cuDeviceProp::Get__Memory_Bus_Width(void) const {
  return (this->p_memory_bus_width);
}  // Bits.

__host__ __device__ size_t
cuDeviceProp::Get__L2_Cache_Size(void) const {
  return (this->p_L2_cache_size);
}  // Bytes.

__host__ __device__ int cuDeviceProp::Get__Multi_GPU_Board_Group_ID(
    void) const {
  return (this->p_multi_gpu_board_group_ID);
}

__host__ __device__ int
cuDeviceProp::Get__Single_To_Double_Precision_Performance_Ratio(
    void) const {
  return (this->p_single_to_double_precision_performance_ratio);
}

__host__ __device__ int cuDeviceProp::Get__ID(void) const {
  return (this->p_device_overlap);
}

__host__ __device__ size_t
cuDeviceProp::Get__Minimum_Threads_For_Occupancy(
    bool const use_default_received) const {
  if (use_default_received) {
    return (this->p_minimum_threads_for_occupancy);
  } else {
    return (this->p_minimum_threads_for_occupancy_custom);
  }
}

__host__ __device__ void
cuDeviceProp::Set__Minimum_Threads_For_Occupancy(
    size_t const minimum_threads_per_received) {
  if (minimum_threads_per_received > this->p_maximum_threads_per_block) {
    ERR(
        L"Minimum threads (%zu) can't be greater than %zu "
        "threads.", minimum_threads_per_received,
        this->p_maximum_threads_per_block);

    return;
  }

  this->p_minimum_threads_for_occupancy_custom = minimum_threads_per_received;
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Threads(void) const {
  return (this->p_maximum_number_threads);
}

__host__ __device__ size_t
cuDeviceProp::Get__Number_Concurrent_Kernel_By_Compute_Capability(
    void) const {
  size_t tmp_max_concurrent_kernel(0_UZ);

  switch (this->p_major_compute_capability) {
    case 2:  // Fermi.
      tmp_max_concurrent_kernel = 16_UZ;
      break;
    case 3:  // Kepler.
      if (this->p_minor_compute_capability == 2) {
        tmp_max_concurrent_kernel = 4_UZ;
      }
      if (this->p_minor_compute_capability == 5 ||
          this->p_minor_compute_capability == 7) {
        tmp_max_concurrent_kernel = 32_UZ;
      } else {
        ERR(L"Unknown minor device version.");
      }
      break;
    case 5:  // Maxwell.
      if (this->p_minor_compute_capability == 0 ||
          this->p_minor_compute_capability == 2) {
        tmp_max_concurrent_kernel = 32_UZ;
      } else if (this->p_minor_compute_capability == 3) {
        tmp_max_concurrent_kernel = 16_UZ;
      } else {
        ERR(L"Unknown minor device version.");
      }
      break;
    case 6:  // Pascal.
      if (this->p_minor_compute_capability == 0) {
        tmp_max_concurrent_kernel = 128_UZ;
      } else if (this->p_minor_compute_capability == 1) {
        tmp_max_concurrent_kernel = 32_UZ;
      } else if (this->p_minor_compute_capability == 2) {
        tmp_max_concurrent_kernel = 16_UZ;
      } else {
        ERR(L"Unknown minor device version.");
      }
      break;
    case 7:  // Volta.
      if (this->p_minor_compute_capability == 0) {
        tmp_max_concurrent_kernel = 128_UZ;
      } else {
        ERR(L"Unknown minor device version.");
      }
      break;
    default:
      ERR(L"Unknown major device version.");
      break;
  }

  return (tmp_max_concurrent_kernel);
}

__host__ __device__ size_t
cuDeviceProp::Get__Number_Concurrent_Kernel(void) const {
  return (this->p_number_concurrent_kernel);
}

__host__ __device__ size_t
cuDeviceProp::Get__Number_CUDA_Cores_By_Compute_Capability(
    void) const {
  size_t tmp_cores(0u);

  switch (this->p_major_compute_capability) {
    case 2:  // Fermi.
      if (this->p_minor_compute_capability == 1) {
        tmp_cores = this->p_number_multiprocessor * 48_UZ;
      } else {
        tmp_cores = this->p_number_multiprocessor * 32_UZ;
      }
      break;
    case 3:  // Kepler.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-3-0
      tmp_cores = this->p_number_multiprocessor * 192_UZ;
      break;
    case 5:  // Maxwell.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x
      tmp_cores = this->p_number_multiprocessor * 128_UZ;
      break;
    case 6:  // Pascal.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-6-x
      if (this->p_minor_compute_capability == 0) {
        tmp_cores = this->p_number_multiprocessor * 64_UZ;
      } else if (this->p_minor_compute_capability == 1 ||
                 this->p_minor_compute_capability == 2) {
        tmp_cores = this->p_number_multiprocessor * 128_UZ;
      } else {
        ERR(L"Unknown minor device version.");
      }
      break;
    case 7:  // Volta.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x
      if (this->p_minor_compute_capability == 0) {
        tmp_cores = this->p_number_multiprocessor * 64_UZ;
      } else {
        ERR(L"Unknown minor device version.");
      }
      break;
    default:
      ERR(L"Unknown major device version.");
      break;
  }

  return (tmp_cores);
}

__host__ __device__ size_t
cuDeviceProp::CUDA__Number_CUDA_Cores(void) const {
  return (this->p_number_CUDA_cores);
}

__host__ __device__ size_t
cuDeviceProp::Get__Number_CUDA_Cores_Per_Multiprocessor(void) const {
  return (this->p_number_CUDA_cores_per_multiprocessor);
}

__host__ __device__ size_t cuDeviceProp::
    Get__Maximum_Blocks_Per_Multiprocessor_By_Compute_Capability(void) const {
  size_t tmp_maximum_number_blocks_per_multiprocessor(0u);

  switch (this->p_major_compute_capability) {
    case 2:  // Fermi.
      tmp_maximum_number_blocks_per_multiprocessor = 8_UZ;
    case 3:  // Kepler.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-3-0
      tmp_maximum_number_blocks_per_multiprocessor = 16_UZ;
      break;
    case 5:  // Maxwell.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x
    case 6:  // Pascal.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-6-x
    case 7:  // Volta.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x
      tmp_maximum_number_blocks_per_multiprocessor = 32_UZ;
      break;
    default:
      ERR(L"Unknown major device version.");
      break;
  }

  return (tmp_maximum_number_blocks_per_multiprocessor);
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Blocks_Per_Multiprocessor(void) const {
  return (this->p_maximum_blocks_per_multiprocessor);
}

__host__ __device__ size_t cuDeviceProp::
    Get__Maximum_Warps_Per_Multiprocessor_By_Compute_Capability(void) const {
  return (64);
}

__host__ __device__ size_t
cuDeviceProp::Get__Maximum_Warps_Per_Multiprocessor(void) const {
  return (this->p_maximum_number_warps_per_multiprocessor);
}

__host__ __device__ size_t
cuDeviceProp::Get__Number_Shared_Memory_Banks_By_Compute_Capability(
    void) const {
  return (32);
}

__host__ __device__ size_t
cuDeviceProp::Get__Number_Shared_Memory_Banks(void) const {
  return (this->p_number_shared_memory_banks);
}

__host__ __device__ size_t
cuDeviceProp::Get__Limit_Block_Due_To_Warp_Per_Multiprocessor(
    size_t const number_warps_received) const {
  size_t const tmp_LimitBlocksPerMultiprocessor(
      this->p_maximum_blocks_per_multiprocessor),  // Limit blocks per
                                                   // multiprocessor
      tmp_LimitWarpsPerMultiprocessor(
          this->p_maximum_number_warps_per_multiprocessor),  // Limit warps per
                                                             // multiprocessor
      tmp_LimitBlocksDueToWarps(std::min<size_t>(
          tmp_LimitBlocksPerMultiprocessor,
          static_cast<size_t>(tmp_LimitWarpsPerMultiprocessor /
                              number_warps_received)));

  return (tmp_LimitBlocksDueToWarps);
}

__host__ __device__ size_t
cuDeviceProp::Get__Total_Global_Memory(void) const {
  return (this->p_total_global_memory);
}

__host__ __device__ size_t
cuDeviceProp::Get__Total_Constant_Memory(void) const {
  return (this->p_total_constant_memory);
}

__host__ __device__ size_t
cuDeviceProp::Get__Shared_Memory_Per_Block(void) const {
  return (this->p_shared_memory_per_block);
}

__host__ __device__ size_t
cuDeviceProp::Get__Shared_Memory_Per_Multiprocessor(void) const {
  return (this->p_shared_memory_per_multiprocessor);
}

__host__ __device__ size_t
cuDeviceProp::Get__Shared_Memory_Per_Block_Opt_In(void) const {
  return (this->p_shared_memory_per_block_opt_in);
}

__host__ __device__ size_t
cuDeviceProp::Get__Memory_Pitch(void) const {
  return (this->p_memory_pitch);
}

__host__ __device__ size_t
cuDeviceProp::Get__Texture_Alignment(void) const {
  return (this->p_texture_alignment);
}

__host__ __device__ size_t
cuDeviceProp::Get__Texture_Pitch_Alignment(void) const {
  return (this->p_texture_pitch_alignment);
}

__host__ __device__ size_t
cuDeviceProp::Get__Surface_Alignment(void) const {
  return (this->p_surface_alignment);
}

__host__ __device__ double
cuDeviceProp::OccupencyOfEachMultiprocessor(
    size_t const thread_count_received,
    size_t const registers_per_thread_received,
    size_t const shared_memory_per_block_received) const {
  size_t const tmp_ThreadsPerWarp(this->p_warp_size),  // Threads per warp.
      tmp_LimitBlocksPerMultiprocessor(
          this->p_maximum_blocks_per_multiprocessor),  // Limit blocks per
                                                       // multiprocessor
      tmp_LimitWarpsPerMultiprocessor(
          this->p_maximum_number_warps_per_multiprocessor),  // Limit warps per
                                                             // multiprocessor
      tmp_MyWarpsPerBlock(
          static_cast<size_t>(ceil(static_cast<double>(thread_count_received) /
                                   static_cast<double>(tmp_ThreadsPerWarp)))),
      tmp_LimitBlocksDueToWarps(std::min<size_t>(
          tmp_LimitBlocksPerMultiprocessor,
          static_cast<size_t>(tmp_LimitWarpsPerMultiprocessor /
                              tmp_MyWarpsPerBlock))),
      tmp_LimitSharedMemoryPerMultiprocessor(
          this->p_shared_memory_per_multiprocessor),
      tmp_LimitSharedMemoryPerBlock(this->p_shared_memory_per_block),
      tmp_SharedMemoryAllocationUnitSize(256),
      tmp_MySharedMemoryPerBlock(
          std::max<size_t>(shared_memory_per_block_received,
                                      tmp_SharedMemoryAllocationUnitSize)),
      tmp_LimitBlocksDueToSharedMemory(
          tmp_MySharedMemoryPerBlock > tmp_LimitSharedMemoryPerBlock
              ? 0_UZ
              : (tmp_MySharedMemoryPerBlock > 0_UZ
                     ? static_cast<size_t>(
                           tmp_LimitSharedMemoryPerMultiprocessor /
                           tmp_MySharedMemoryPerBlock)
                     : tmp_LimitBlocksPerMultiprocessor)),
      tmp_ActivateThreadBlocksPerMultiprocessor(std::min<size_t>(
          tmp_LimitBlocksDueToWarps, tmp_LimitBlocksDueToSharedMemory)),
      tmp_ActivateWarpsPerMultiprocessor(
          tmp_ActivateThreadBlocksPerMultiprocessor * tmp_MyWarpsPerBlock);

  return (static_cast<double>(tmp_ActivateWarpsPerMultiprocessor) /
          static_cast<double>(tmp_LimitWarpsPerMultiprocessor) * 100.0);
}

__host__ __device__ void cuDeviceProp::Grid_Block_1Dimensions(
    size_t const elements_received, size_t limit_blocks_received,
    struct dim3 &ref_dim3_grid_received, struct dim3 &ref_dim3_block_received,
    size_t const registers_per_thread_received,
    size_t const shared_memory_per_block_received,
    size_t const shared_memory_variable_per_block_received) const {
  if (elements_received == 0u) {
    ERR(L"Number elements received equal zero.",);

    return;
  }

  size_t const tmp_NumberOfMultiprocessors(
      this->p_number_multiprocessor),  // Number of multiprocessors.
      tmp_LimitBlocksPerMultiprocessor(
          this->p_maximum_blocks_per_multiprocessor),  // Limit blocks per
                                                       // multiprocessor
      tmp_LimitBlocksDueToGPU(
          tmp_NumberOfMultiprocessors *
          tmp_LimitBlocksPerMultiprocessor),  // Limit blocks per multiprocessor
      tmp_MaxThreadsPerBlock(
          this->p_maximum_threads_per_block),  // Max threads per block.
      tmp_MaxThreadsPerMultiprocessor(
          this->p_maximum_threads_per_multiprocessor),  // Max threads per
                                                        // block.
      tmp_MaxActiveThreadBlocksPerMultiprocessor(
          tmp_MaxThreadsPerMultiprocessor / tmp_MaxThreadsPerBlock),
      tmp_MaxActiveThreadBlocksDueToGPU(
          tmp_MaxActiveThreadBlocksPerMultiprocessor *
          tmp_NumberOfMultiprocessors);
  size_t tmp_BestNeededBlock(1), &tmp_BlocksAsk(tmp_BestNeededBlock),
      tmp_BestNumberElementsPerBlock(0), tmp_NumberElementsPerBlock;

  double tmp_OccupencyValue, tmp_BestOccupencyValue(-DBL_MAX);

  // Assign default value grid dimensions.
  ref_dim3_grid_received.y = 1u;
  ref_dim3_grid_received.z = 1u;

  // Assign default value block dimensions.
  ref_dim3_block_received.y = 1u;
  ref_dim3_block_received.z = 1u;

  if (limit_blocks_received == 0u) {
    limit_blocks_received = tmp_LimitBlocksDueToGPU;
  } else {
    limit_blocks_received = std::min<size_t>(
        limit_blocks_received, tmp_LimitBlocksDueToGPU);
  }

  size_t i(
      static_cast<size_t>(ceil(static_cast<double>(elements_received) /
                               static_cast<double>(tmp_MaxThreadsPerBlock))));

  i = std::min<size_t>(i, limit_blocks_received);

  do {
    // We divide 'number of elements' by 'number of blocks so far'.
    tmp_NumberElementsPerBlock = static_cast<size_t>(elements_received / i);
    // With a maximum of 'max threads per block'.
    tmp_NumberElementsPerBlock = ref_dim3_block_received.x =
        static_cast<unsigned int>(std::min<size_t>(
            tmp_NumberElementsPerBlock, tmp_MaxThreadsPerBlock));

    // save the best configuration.
    tmp_OccupencyValue = this->OccupencyOfEachMultiprocessor(
        ref_dim3_block_received.x, registers_per_thread_received,
        ref_dim3_block_received.x * shared_memory_variable_per_block_received +
            shared_memory_per_block_received);

    if (tmp_OccupencyValue - tmp_BestOccupencyValue >=
        1.0)  // ((100-95=)5 >= 3)=true
    {
      tmp_BestOccupencyValue = tmp_OccupencyValue;

      tmp_BestNumberElementsPerBlock = tmp_NumberElementsPerBlock;

      tmp_BestNeededBlock = i;
    } else {
      break;
    }
  } while (++i < tmp_MaxActiveThreadBlocksDueToGPU &&
           i < limit_blocks_received);

  ref_dim3_block_received.x =
      static_cast<unsigned int>(tmp_BestNumberElementsPerBlock);

  size_t const tmp_ThreadsPerWarp(this->p_warp_size),  // Threads per warp.
      tmp_MyWarpsPerBlock(static_cast<size_t>(
          ceil(static_cast<double>(tmp_BestNumberElementsPerBlock) /
               static_cast<double>(tmp_ThreadsPerWarp)))),
      tmp_LimitWarpsPerMultiprocessor(
          this->p_maximum_number_warps_per_multiprocessor),  // Limit warps per
                                                             // multiprocessor
      tmp_Delta(static_cast<size_t>(tmp_LimitWarpsPerMultiprocessor /
                                    tmp_MyWarpsPerBlock)),
      tmp_LimitBlocksDueToWarpsPerSM(std::min<size_t>(
          tmp_LimitBlocksPerMultiprocessor, tmp_Delta)),
      tmp_LimitBlocksDueToWarpsTotal(tmp_LimitBlocksDueToWarpsPerSM *
                                     tmp_NumberOfMultiprocessors);

  ref_dim3_grid_received.x =
      static_cast<unsigned int>(std::min<size_t>(
          tmp_BlocksAsk, tmp_LimitBlocksDueToWarpsTotal));
}

__host__ __device__ void cuDeviceProp::Grid_Block_2Dimensions(
    size_t const rows, size_t const cols,
    size_t limit_blocks_received, struct dim3 &ref_dim3_grid_received,
    struct dim3 &ref_dim3_block_received,
    size_t const registers_per_thread_received,
    size_t const shared_memory_per_block_received,
    size_t const shared_memory_variable_per_block_received) const {
  if (rows * cols == 0u) {
    ERR(L"Number elements received equal zero.",);

    return;
  }

  size_t const tmp_NumberOfMultiprocessors(
      this->p_number_multiprocessor),  // Number of multiprocessors.
      tmp_LimitBlocksPerMultiprocessor(
          this->p_maximum_blocks_per_multiprocessor),  // Limit blocks per
                                                       // multiprocessor
      tmp_LimitBlocksDueToGPU(
          tmp_NumberOfMultiprocessors *
          tmp_LimitBlocksPerMultiprocessor),  // Limit blocks per multiprocessor
      tmp_MaxThreadsPerBlock(
          this->p_maximum_threads_per_block),  // Max threads per block.
      tmp_MaxThreadsPerMultiprocessor(
          this->p_maximum_threads_per_multiprocessor),  // Max threads per
                                                        // block.
      tmp_MaxActiveThreadBlocksPerMultiprocessor(
          tmp_MaxThreadsPerMultiprocessor / tmp_MaxThreadsPerBlock),
      tmp_MaxActiveThreadBlocksDueToGPU(
          tmp_MaxActiveThreadBlocksPerMultiprocessor *
          tmp_NumberOfMultiprocessors);
  size_t tmp_BestNumberRowsPerBlock(0), tmp_BestNumberColumnsPerBlock(0),
      tmp_NumberRowsPerBlock, tmp_NumberColumnsPerBlock;

  double tmp_OccupencyValue, tmp_BestOccupencyValue(-DBL_MAX);

  // Assign default value grid dimensions.
  ref_dim3_grid_received.z = 1u;

  // Assign default value block dimensions.
  ref_dim3_block_received.z = 1u;

  if (limit_blocks_received == 0u) {
    limit_blocks_received = tmp_LimitBlocksDueToGPU;
  } else {
    limit_blocks_received = std::min<size_t>(
        limit_blocks_received, tmp_LimitBlocksDueToGPU);
  }

  size_t tmp_initial_block_by_row(
      static_cast<size_t>(ceil(static_cast<double>(rows) /
                               static_cast<double>(tmp_MaxThreadsPerBlock)))),
      tmp_initial_block_by_column(static_cast<size_t>(
          ceil(static_cast<double>(cols) /
               static_cast<double>(tmp_MaxThreadsPerBlock)))),
      i(std::max<size_t>(tmp_initial_block_by_row,
                                    tmp_initial_block_by_column));

  i = std::min<size_t>(i, limit_blocks_received);

  do {
    // We divide 'number of rows' by 'number of blocks so far'.
    tmp_NumberRowsPerBlock = static_cast<size_t>(rows / i);
    // With a maximum of 'max threads per block'.
    tmp_NumberRowsPerBlock = std::min<size_t>(
        tmp_NumberRowsPerBlock, tmp_MaxThreadsPerBlock);

    // We divide 'number of columns' by 'number of blocks so far'.
    tmp_NumberColumnsPerBlock = static_cast<size_t>(cols / i);
    // With a maximum of 'max threads per block'.
    tmp_NumberColumnsPerBlock = std::min<size_t>(
        tmp_NumberColumnsPerBlock, tmp_MaxThreadsPerBlock);

    // If overflow.
    if (tmp_NumberRowsPerBlock * tmp_NumberColumnsPerBlock >
        tmp_MaxThreadsPerBlock) {
      // If rows is greater than columns.
      if (tmp_NumberRowsPerBlock > tmp_NumberColumnsPerBlock) {
        tmp_NumberRowsPerBlock = static_cast<size_t>(tmp_MaxThreadsPerBlock /
                                                     tmp_NumberColumnsPerBlock);
      }
      // else if columns is greater or equal to rows.
      else {
        tmp_NumberColumnsPerBlock = static_cast<size_t>(tmp_MaxThreadsPerBlock /
                                                        tmp_NumberRowsPerBlock);
      }
    }

    ref_dim3_block_received.x =
        static_cast<unsigned int>(tmp_NumberRowsPerBlock);

    ref_dim3_block_received.y =
        static_cast<unsigned int>(tmp_NumberColumnsPerBlock);

    // save the best configuration.
    tmp_OccupencyValue = this->OccupencyOfEachMultiprocessor(
        ref_dim3_block_received.x * ref_dim3_block_received.y,
        registers_per_thread_received,
        ref_dim3_block_received.x * ref_dim3_block_received.y *
                shared_memory_variable_per_block_received +
            shared_memory_per_block_received);

    if (tmp_OccupencyValue - tmp_BestOccupencyValue >=
        1.0)  // ((100-95=)5 >= 3)=true
    {
      tmp_BestOccupencyValue = tmp_OccupencyValue;

      tmp_BestNumberRowsPerBlock = tmp_NumberRowsPerBlock;
      tmp_BestNumberColumnsPerBlock = tmp_NumberColumnsPerBlock;
    } else {
      break;
    }
  } while (++i < tmp_MaxActiveThreadBlocksDueToGPU &&
           i < limit_blocks_received);

  ref_dim3_block_received.x =
      static_cast<unsigned int>(tmp_BestNumberRowsPerBlock);
  ref_dim3_block_received.y =
      static_cast<unsigned int>(tmp_BestNumberColumnsPerBlock);

  size_t const tmp_ThreadsPerWarp(this->p_warp_size),  // Threads per warp.
      tmp_MyWarpsPerBlock(static_cast<size_t>(
          ceil(static_cast<double>(tmp_BestNumberRowsPerBlock *
                                   tmp_BestNumberColumnsPerBlock) /
               static_cast<double>(tmp_ThreadsPerWarp)))),
      tmp_LimitWarpsPerMultiprocessor(
          this->p_maximum_number_warps_per_multiprocessor),  // Limit warps per
                                                             // multiprocessor
      tmp_Delta(static_cast<size_t>(tmp_LimitWarpsPerMultiprocessor /
                                    tmp_MyWarpsPerBlock)),
      tmp_LimitBlocksDueToWarpsPerSM(std::min<size_t>(
          tmp_LimitBlocksPerMultiprocessor, tmp_Delta));
  size_t tmp_LimitBlocksDueToWarpsTotal(tmp_LimitBlocksDueToWarpsPerSM *
                                        tmp_NumberOfMultiprocessors),
      tmp_scale_grid_x, tmp_scale_grid_y;

  tmp_scale_grid_x =
      static_cast<size_t>(ceil(static_cast<double>(rows) /
                               static_cast<double>(ref_dim3_grid_received.x)));
  ref_dim3_grid_received.x = static_cast<unsigned int>(tmp_scale_grid_x);

  tmp_scale_grid_y =
      static_cast<size_t>(ceil(static_cast<double>(cols) /
                               static_cast<double>(ref_dim3_grid_received.y)));
  ref_dim3_grid_received.y = static_cast<unsigned int>(tmp_scale_grid_y);

  tmp_LimitBlocksDueToWarpsTotal = std::min<size_t>(
      tmp_LimitBlocksDueToWarpsTotal,
      ref_dim3_grid_received.x * ref_dim3_grid_received.y);

  ref_dim3_grid_received.x =
      static_cast<unsigned int>(std::min<size_t>(
          ref_dim3_grid_received.x, tmp_LimitBlocksDueToWarpsTotal));

  ref_dim3_grid_received.y =
      static_cast<unsigned int>(std::min<size_t>(
          ref_dim3_grid_received.y, tmp_LimitBlocksDueToWarpsTotal));

  // If overflow.
  if (ref_dim3_grid_received.x * ref_dim3_grid_received.y >
      tmp_LimitBlocksDueToWarpsTotal) {
    // If rows is greater than columns.
    if (ref_dim3_grid_received.x > ref_dim3_grid_received.y) {
      ref_dim3_grid_received.x = static_cast<unsigned int>(
          tmp_LimitBlocksDueToWarpsTotal / ref_dim3_grid_received.y);
    }
    // else if columns is greater or equal to rows.
    else {
      ref_dim3_grid_received.y = static_cast<unsigned int>(
          tmp_LimitBlocksDueToWarpsTotal / ref_dim3_grid_received.x);
    }
  }
}

__host__ __device__ void
cuDeviceProp::Grid_Block_Transpose_2Dimensions(
    size_t const rows, size_t const cols,
    size_t limit_blocks_received, struct dim3 &ref_dim3_grid_received,
    struct dim3 &ref_dim3_block_received,
    size_t const registers_per_thread_received,
    size_t const shared_memory_per_block_received,
    size_t const shared_memory_variable_per_block_received) const {
  if (rows * cols == 0u) {
    ERR(L"Number elements received equal zero.",);

    return;
  }

  size_t const tmp_NumberOfMultiprocessors(
      this->p_number_multiprocessor),  // Number of multiprocessors.
      tmp_LimitBlocksPerMultiprocessor(
          this->p_maximum_blocks_per_multiprocessor),  // Limit blocks per
                                                       // multiprocessor
      tmp_LimitBlocksDueToGPU(
          tmp_NumberOfMultiprocessors *
          tmp_LimitBlocksPerMultiprocessor),  // Limit blocks per multiprocessor
      tmp_MaxThreadsPerBlock(
          this->p_maximum_threads_per_block),  // Max threads per block.
      tmp_SqrtMaxThreadsPerBlock(static_cast<size_t>(sqrt(static_cast<double>(
          this->p_maximum_threads_per_block)))),  // Square root of max threads
                                                  // per block.
      tmp_MaxThreadsPerMultiprocessor(
          this->p_maximum_threads_per_multiprocessor),  // Max threads per
                                                        // block.
      tmp_MaxActiveThreadBlocksPerMultiprocessor(
          tmp_MaxThreadsPerMultiprocessor / tmp_MaxThreadsPerBlock),
      tmp_MaxActiveThreadBlocksDueToGPU(
          tmp_MaxActiveThreadBlocksPerMultiprocessor *
          tmp_NumberOfMultiprocessors);
  size_t tmp_BestNumberSquareElementsPerBlock(0),
      tmp_NumberSquareElementsPerBlock;

  double tmp_OccupencyValue, tmp_BestOccupencyValue(-DBL_MAX);

  // Assign default value grid dimensions.
  ref_dim3_grid_received.z = 1u;

  // Assign default value block dimensions.
  ref_dim3_block_received.z = 1u;

  if (limit_blocks_received == 0u) {
    limit_blocks_received = tmp_LimitBlocksDueToGPU;
  } else {
    limit_blocks_received = std::min<size_t>(
        limit_blocks_received, tmp_LimitBlocksDueToGPU);
  }

  size_t i(static_cast<size_t>(
      ceil(static_cast<double>(rows * cols) /
           static_cast<double>(tmp_MaxThreadsPerBlock))));

  i = std::min<size_t>(i, limit_blocks_received);

  do {
    // We divide 'number of rows' times 'number of columns' by 'number of blocks
    // so far'. Then root square the result and divide by two (multiple
    // transpose per thread).
    tmp_NumberSquareElementsPerBlock = static_cast<size_t>(
        ceil(sqrt(static_cast<double>(rows * cols) /
                  static_cast<double>(i)) /
             2.0));
    // With a minimum of '8 threads per axis'.
    tmp_NumberSquareElementsPerBlock =
        std::max<size_t>(tmp_NumberSquareElementsPerBlock, 8u);
    // With a maximum of 'square root maximum threads per block'.
    tmp_NumberSquareElementsPerBlock = std::min<size_t>(
        tmp_NumberSquareElementsPerBlock, tmp_SqrtMaxThreadsPerBlock);

    ref_dim3_block_received.x =
        static_cast<unsigned int>(tmp_NumberSquareElementsPerBlock);
    ref_dim3_block_received.y =
        static_cast<unsigned int>(tmp_NumberSquareElementsPerBlock);

    // save the best configuration.
    tmp_OccupencyValue = this->OccupencyOfEachMultiprocessor(
        ref_dim3_block_received.x * ref_dim3_block_received.y,
        registers_per_thread_received,
        ref_dim3_block_received.x * ref_dim3_block_received.y *
                shared_memory_variable_per_block_received +
            shared_memory_per_block_received);

    if (tmp_OccupencyValue - tmp_BestOccupencyValue >=
        1.0)  // ((100-95=)5 >= 3)=true
    {
      tmp_BestOccupencyValue = tmp_OccupencyValue;

      tmp_BestNumberSquareElementsPerBlock = tmp_NumberSquareElementsPerBlock;
    } else {
      break;
    }
  } while (++i < tmp_MaxActiveThreadBlocksDueToGPU &&
           i < limit_blocks_received);

  ref_dim3_block_received.x =
      static_cast<unsigned int>(tmp_BestNumberSquareElementsPerBlock);
  ref_dim3_block_received.y =
      static_cast<unsigned int>(tmp_BestNumberSquareElementsPerBlock);

  size_t const tmp_ThreadsPerWarp(this->p_warp_size),  // Threads per warp.
      tmp_MyWarpsPerBlock(static_cast<size_t>(
          ceil(static_cast<double>(tmp_BestNumberSquareElementsPerBlock *
                                   tmp_BestNumberSquareElementsPerBlock) /
               static_cast<double>(tmp_ThreadsPerWarp)))),
      tmp_LimitWarpsPerMultiprocessor(
          this->p_maximum_number_warps_per_multiprocessor),  // Limit warps per
                                                             // multiprocessor
      tmp_Delta(static_cast<size_t>(tmp_LimitWarpsPerMultiprocessor /
                                    tmp_MyWarpsPerBlock)),
      tmp_LimitBlocksDueToWarpsPerSM(std::min<size_t>(
          tmp_LimitBlocksPerMultiprocessor, tmp_Delta));
  size_t tmp_LimitBlocksDueToWarpsTotal(tmp_LimitBlocksDueToWarpsPerSM *
                                        tmp_NumberOfMultiprocessors),
      tmp_scale_grid_x, tmp_scale_grid_y;

  // Total rows to transpose divide by rows per thread block times two (multiple
  // transpose per thread).
  tmp_scale_grid_x = static_cast<size_t>(
      ceil(static_cast<double>(rows) /
           (static_cast<double>(ref_dim3_block_received.x) * 2.0)));
  ref_dim3_grid_received.x = static_cast<unsigned int>(tmp_scale_grid_x);

  // Total columns to transpose divide by columns per thread block times two
  // (multiple transpose per thread).
  tmp_scale_grid_y = static_cast<size_t>(
      ceil(static_cast<double>(cols) /
           (static_cast<double>(ref_dim3_block_received.y) * 2.0)));
  ref_dim3_grid_received.y = static_cast<unsigned int>(tmp_scale_grid_y);

  tmp_LimitBlocksDueToWarpsTotal = std::min<size_t>(
      tmp_LimitBlocksDueToWarpsTotal,
      ref_dim3_grid_received.x * ref_dim3_grid_received.y);

  ref_dim3_grid_received.x =
      static_cast<unsigned int>(std::min<size_t>(
          ref_dim3_grid_received.x, tmp_LimitBlocksDueToWarpsTotal));

  ref_dim3_grid_received.y =
      static_cast<unsigned int>(std::min<size_t>(
          ref_dim3_grid_received.y, tmp_LimitBlocksDueToWarpsTotal));

  // If overflow.
  if (static_cast<size_t>(ref_dim3_grid_received.x * ref_dim3_grid_received.y) >
      tmp_LimitBlocksDueToWarpsTotal) {
    ref_dim3_grid_received.y = static_cast<unsigned int>(
        tmp_LimitBlocksDueToWarpsTotal / ref_dim3_grid_received.x);
  }
}

__host__ __device__ void cuDeviceProp::Grid_Block_cuRAND_1Dimensions(
    size_t const elements_received, size_t limit_blocks_received,
    struct dim3 &ref_dim3_grid_received,
    struct dim3 &ref_dim3_block_received) const {
  if (elements_received == 0u) {
    ERR(L"Number elements received equal zero.",);

    return;
  }

  size_t const tmp_LimitFullStatesDueToGPU(
      static_cast<size_t>(this->Get__Maximum_Threads() / 256_UZ)),
      tmp_LimitStatesDueToGPU(
          std::max<size_t>(tmp_LimitFullStatesDueToGPU, 1u));

  if (limit_blocks_received == 0u) {
    limit_blocks_received = tmp_LimitStatesDueToGPU;
  } else {
    limit_blocks_received = std::min<size_t>(
        limit_blocks_received, tmp_LimitStatesDueToGPU);
  }

  // Number of states require
  ref_dim3_grid_received.x =
      static_cast<unsigned int>(elements_received / 256_UZ);
  // Minimum of one state.
  ref_dim3_grid_received.x = static_cast<unsigned int>(
      std::max<size_t>(ref_dim3_grid_received.x, 1_UZ));
  // Don't overflow the maximum usable states.
  ref_dim3_grid_received.x =
      static_cast<unsigned int>(std::min<size_t>(
          ref_dim3_grid_received.x, limit_blocks_received));
  ref_dim3_grid_received.y = 1u;
  ref_dim3_grid_received.z = 1u;

  ref_dim3_block_received.x = static_cast<unsigned int>(
      std::min<size_t>(elements_received, 256_UZ));
  ref_dim3_block_received.y = 1u;
  ref_dim3_block_received.z = 1u;
}

__host__ __device__ void
cuDeviceProp::Grid_Block_Dynamic_Parallelisme(
    size_t const elements_received, size_t limit_blocks_received,
    struct dim3 &ref_dim3_grid_received,
    struct dim3 &ref_dim3_block_received) const {
  if (elements_received == 0u) {
    ERR(L"Number elements received equal zero.",);

    return;
  }

  if (limit_blocks_received == 0u) {
    limit_blocks_received = this->Get__Number_Concurrent_Kernel();
  } else {
    limit_blocks_received = std::min<size_t>(
        limit_blocks_received, this->Get__Number_Concurrent_Kernel());
  }

  ref_dim3_grid_received.x = static_cast<unsigned int>(
      std::min<size_t>(elements_received, limit_blocks_received));
  ref_dim3_grid_received.y = 1u;
  ref_dim3_grid_received.z = 1u;

  ref_dim3_block_received.x = static_cast<unsigned int>(
      elements_received / static_cast<size_t>(ref_dim3_grid_received.x));
  ref_dim3_block_received.x = static_cast<unsigned int>(
      std::max<size_t>(ref_dim3_block_received.x, 1_UZ));

  ref_dim3_block_received.y = 1u;
  ref_dim3_block_received.z = 1u;
}

__host__ __device__ void cuDeviceProp::Grid_Block_Reduce_1Dimensions(
    size_t const elements_received, size_t limit_blocks_received,
    struct dim3 &ref_dim3_grid_received, struct dim3 &ref_dim3_block_received,
    size_t const registers_per_thread_received,
    size_t const shared_memory_per_block_received,
    size_t const shared_memory_variable_per_block_received) const {
  if (elements_received == 0u) {
    ERR(L"Number elements received equal zero.",);

    return;
  }

  size_t const tmp_NumberOfMultiprocessors(
      this->p_number_multiprocessor),  // Number of multiprocessors.
      tmp_LimitBlocksPerMultiprocessor(
          this->p_maximum_blocks_per_multiprocessor),  // Limit blocks per
                                                       // multiprocessor
      tmp_LimitBlocksDueToGPU(
          tmp_NumberOfMultiprocessors *
          tmp_LimitBlocksPerMultiprocessor),  // Limit blocks per multiprocessor
      tmp_MaxThreadsPerBlock(
          this->p_maximum_threads_per_block),  // Max threads per block.
      tmp_MaxThreadsPerMultiprocessor(
          this->p_maximum_threads_per_multiprocessor),  // Max threads per
                                                        // block.
      tmp_MaxActiveThreadBlocksPerMultiprocessor(
          tmp_MaxThreadsPerMultiprocessor / tmp_MaxThreadsPerBlock),
      tmp_MaxActiveThreadBlocksDueToGPU(
          tmp_MaxActiveThreadBlocksPerMultiprocessor *
          tmp_NumberOfMultiprocessors);
  size_t tmp_BestNeededBlock(1), &tmp_BlocksAsk(tmp_BestNeededBlock),
      tmp_BestNumberElementsPerBlock(0), tmp_NumberElementsPerBlock;

  double tmp_OccupencyValue, tmp_BestOccupencyValue(-DBL_MAX);

  // Assign default value grid dimensions.
  ref_dim3_grid_received.y = 1u;
  ref_dim3_grid_received.z = 1u;

  // Assign default value block dimensions.
  ref_dim3_block_received.y = 1u;
  ref_dim3_block_received.z = 1u;

  if (limit_blocks_received == 0u) {
    limit_blocks_received = tmp_LimitBlocksDueToGPU;
  } else {
    limit_blocks_received = std::min<size_t>(
        limit_blocks_received, tmp_LimitBlocksDueToGPU);
  }

  // Divide by "two" is for multiple add.
  size_t i(static_cast<size_t>(
      ceil(static_cast<double>(elements_received) /
           static_cast<double>(tmp_MaxThreadsPerBlock) / 2.0)));

  do {
    // We divide 'number of elements' by 'number of blocks so far'.
    // Divide by "two" is for multiple add.
    tmp_NumberElementsPerBlock = static_cast<size_t>(ceil(
        static_cast<double>(elements_received) / static_cast<double>(i) / 2.0));

    // Round down at power of two.
    tmp_NumberElementsPerBlock = DL::Math::Round_Down_At_Power_Of_Two<size_t>(
        tmp_NumberElementsPerBlock);

    // With a maximum of 'max threads per block'.
    tmp_NumberElementsPerBlock = ref_dim3_block_received.x =
        static_cast<unsigned int>(std::min<size_t>(
            tmp_NumberElementsPerBlock, tmp_MaxThreadsPerBlock));

    // save the best configuration.
    tmp_OccupencyValue = this->OccupencyOfEachMultiprocessor(
        ref_dim3_block_received.x, registers_per_thread_received,
        ref_dim3_block_received.x * shared_memory_variable_per_block_received +
            shared_memory_per_block_received);

    if (tmp_OccupencyValue - tmp_BestOccupencyValue >=
        1.0)  // ((100-95=)5 >= 3)=true
    {
      tmp_BestOccupencyValue = tmp_OccupencyValue;

      tmp_BestNumberElementsPerBlock = tmp_NumberElementsPerBlock;

      tmp_BestNeededBlock = i;
    }
    // Break the loop.
    else {
      break;
    }
  } while (++i < tmp_MaxActiveThreadBlocksDueToGPU);

  ref_dim3_block_received.x =
      static_cast<unsigned int>(tmp_BestNumberElementsPerBlock);

  size_t const tmp_ThreadsPerWarp(this->p_warp_size),  // Threads per warp.
      tmp_MyWarpsPerBlock(static_cast<size_t>(
          ceil(static_cast<double>(tmp_BestNumberElementsPerBlock) /
               static_cast<double>(tmp_ThreadsPerWarp)))),
      tmp_LimitWarpsPerMultiprocessor(
          this->p_maximum_number_warps_per_multiprocessor),  // Limit warps per
                                                             // multiprocessor
      tmp_Delta(static_cast<size_t>(tmp_LimitWarpsPerMultiprocessor /
                                    tmp_MyWarpsPerBlock)),
      tmp_LimitBlocksDueToWarpsPerSM(std::min<size_t>(
          tmp_LimitBlocksPerMultiprocessor, tmp_Delta)),
      tmp_LimitBlocksDueToWarpsTotal(tmp_LimitBlocksDueToWarpsPerSM *
                                     tmp_NumberOfMultiprocessors);

  ref_dim3_grid_received.x =
      static_cast<unsigned int>(std::min<size_t>(
          tmp_BlocksAsk, tmp_LimitBlocksDueToWarpsTotal));
}

__host__ __device__ void
cuDeviceProp::Grid_Block_Reduce_Dynamic_Parallelisme(
    size_t const elements_received, size_t limit_blocks_received,
    struct dim3 &ref_dim3_grid_received,
    struct dim3 &ref_dim3_block_received) const {
  if (elements_received == 0u) {
    ERR(L"Number elements received equal zero.",);

    return;
  }

  size_t const tmp_elements_divided(
      static_cast<size_t>(ceil(static_cast<double>(elements_received) / 2.0)));

  if (limit_blocks_received == 0u) {
    limit_blocks_received = this->Get__Number_Concurrent_Kernel();
  } else {
    limit_blocks_received = std::min<size_t>(
        limit_blocks_received, this->Get__Number_Concurrent_Kernel());
  }

  ref_dim3_grid_received.x = static_cast<unsigned int>(
      std::min<size_t>(tmp_elements_divided, limit_blocks_received));
  ref_dim3_grid_received.y = 1u;
  ref_dim3_grid_received.z = 1u;

  ref_dim3_block_received.x = static_cast<unsigned int>(
      ceil(static_cast<double>(tmp_elements_divided) /
           static_cast<double>(ref_dim3_grid_received.x)));
  ref_dim3_block_received.x =
      static_cast<unsigned int>(DL::Math::Round_Down_At_Power_Of_Two<size_t>(
          ref_dim3_block_received.x));
  ref_dim3_block_received.y = 1u;
  ref_dim3_block_received.z = 1u;
}

__host__ __device__
cuDevicesProp::cuDevicesProp(void) {}

__global__ void kernel__Class_Device_Information_Array__Push_Back(
    int const index_device_received,
    struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received,
    class cuDevicesProp
        *const ptr_Class_Device_Information_Array_received) {
  ptr_Class_Device_Information_Array_received->push_back(
      index_device_received, ptr_struct_cudaDeviceProp_received);
}

__host__ bool cuDevicesProp::push_back(
    int const index_device_received) {
  struct cudaDeviceProp tmp_struct_cudaDeviceProp,
      *tmp_ptr_device_struct_cudaDeviceProp(NULL);

  CUDA__Safe_Call(cudaGetDeviceProperties(&tmp_struct_cudaDeviceProp,
                                          index_device_received));

  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_struct_cudaDeviceProp,
                             sizeof(struct cudaDeviceProp)));

  CUDA__Safe_Call(cudaMemcpy(
      tmp_ptr_device_struct_cudaDeviceProp, &tmp_struct_cudaDeviceProp,
      sizeof(struct cudaDeviceProp), cudaMemcpyKind::cudaMemcpyHostToDevice));

  kernel__Class_Device_Information_Array__Push_Back<<<1, 1u>>>(
      index_device_received, tmp_ptr_device_struct_cudaDeviceProp, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_struct_cudaDeviceProp));

  return true;
}

__host__ __device__ bool cuDevicesProp::push_back(
    int const index_device_received,
    struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received) {
  if (ptr_struct_cudaDeviceProp_received == nullptr) {
    return false;
  }

#ifndef __CUDA_ARCH__
  kernel__Class_Device_Information_Array__Push_Back<<<1, 1u>>>(
      index_device_received, ptr_struct_cudaDeviceProp_received, this);

  CUDA__Check_Error();
#else
  if (CUDA__Required_Compatibility_Device(
          *ptr_struct_cudaDeviceProp_received)) {
    for (size_t i(0); i != this->_number_cuda_devices; ++i) {
      if (this->_ptr_array_Class_Device_Information[i].Get__ID() ==
          index_device_received) {
        return true;
      }
    }

    if (this->_ptr_array_Class_Device_Information == nullptr) {
      this->_ptr_Class_Device_Information_sum =
          new class cuDeviceProp;
      this->_ptr_Class_Device_Information_higher =
          new class cuDeviceProp;
      this->_ptr_Class_Device_Information_lower =
          new class cuDeviceProp;
      this->_ptr_array_Class_Device_Information =
          new class cuDeviceProp[1];
    } else {
      class cuDeviceProp *tmp_ptr_array_Class_Device_Information(
          Memory::reallocate_objects_cpp<class cuDeviceProp>(
              this->_ptr_array_Class_Device_Information,
              this->_number_cuda_devices + 1, this->_number_cuda_devices));

      if (tmp_ptr_array_Class_Device_Information == nullptr) {
        ERR(
            L"'tmp_ptr_array_Class_Device_Information' is a "
            "nullptr.",);

        return false;
      }

      this->_ptr_array_Class_Device_Information =
          tmp_ptr_array_Class_Device_Information;
    }

    if (this->_ptr_array_Class_Device_Information[this->_number_cuda_devices]
            .Initialize(index_device_received,
                        ptr_struct_cudaDeviceProp_received)) {
      this->Update(ptr_struct_cudaDeviceProp_received);
    }

    this->_selected_cuda_device = this->_number_cuda_devices;

    ++this->_number_cuda_devices;
  }
#endif

  return true;
}

__global__ void kernel__Class_Device_Information_Array__Refresh(
    struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received,
    class cuDevicesProp
        *const ptr_Class_Device_Information_Array_received) {
  ptr_Class_Device_Information_Array_received->Update(
      ptr_struct_cudaDeviceProp_received);
}

[[deprecated("Not properly implemented.")]] __host__ __device__ bool cuDevicesProp::Update(
    struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received) {
  if (ptr_struct_cudaDeviceProp_received == nullptr) {
    return false;
  }

#ifndef __CUDA_ARCH__
  kernel__Class_Device_Information_Array__Refresh<<<1, 1u>>>(
      ptr_struct_cudaDeviceProp_received, this);

  CUDA__Check_Error();
#else
  // NotImplementedError
  // ...
  // ...
  // Sum += ptr_struct_cudaDeviceProp_received
  // Higher > ptr_struct_cudaDeviceProp_received
  // Lower < ptr_struct_cudaDeviceProp_received
#endif

  return true;
}

__host__ __device__ bool cuDevicesProp::Deallocate(void) {
  SAFE_DELETE(this->_ptr_Class_Device_Information_sum);
  SAFE_DELETE(this->_ptr_Class_Device_Information_higher);
  SAFE_DELETE(this->_ptr_Class_Device_Information_lower);
  SAFE_DELETE_ARRAY(this->_ptr_array_Class_Device_Information);

  return true;
}

__host__ __device__ bool cuDevicesProp::Select_CUDA_Device(
    int const index_received) {
  if (Get__Number_CUDA_Devices() > index_received) {
    this->_selected_cuda_device = index_received;

    return true;
  } else {
    ERR(L"Index overflow.",);

    return false;
  }
}

__host__ __device__ size_t
cuDevicesProp::Get__Number_CUDA_Devices(void) const {
  return (this->_number_cuda_devices);
}

__host__ __device__ int
cuDevicesProp::Get__Selected_CUDA_Device(void) const {
  return (this->_selected_cuda_device);
}

__host__ __device__ class cuDeviceProp *
cuDevicesProp::Get__CUDA_Device(void) const {
  if (static_cast<int>(this->Get__Number_CUDA_Devices()) >
          this->_selected_cuda_device &&
      this->_selected_cuda_device >= 0) {
    return (&this->_ptr_array_Class_Device_Information
                 [this->_selected_cuda_device]);
  } else {
    return nullptr;
  }
}

__host__ __device__ class cuDeviceProp *
cuDevicesProp::Get__CUDA_Device(
    size_t const index_received) const {
  if (this->Get__Number_CUDA_Devices() > index_received) {
    return (&this->_ptr_array_Class_Device_Information[index_received]);
  } else {
    return nullptr;
  }
}

__host__ __device__
    cuDevicesProp::~cuDevicesProp(void) {
  this->Deallocate();
}
