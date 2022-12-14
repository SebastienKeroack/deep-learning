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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Configuration/Configuration.hpp>
//#include <cooperative_groups.hpp>
//#include <cooperative_groups_helpers.hpp>
#include <cstdio>

#ifndef __CUDA_ARCH__
#define NUMBER_SHARED_MEMORY_BANKS 0u
#define MAXIMUM_THREADS_PER_BLOCK 0u
#elif __CUDA_ARCH__ < 300
#define NUMBER_SHARED_MEMORY_BANKS 16u
#define MAXIMUM_THREADS_PER_BLOCK 1024u
#elif __CUDA_ARCH__ >= 300
#define NUMBER_SHARED_MEMORY_BANKS 32u
#define MAXIMUM_THREADS_PER_BLOCK 1024u
#endif

#ifdef __CUDA_ARCH__
#define __Lch_Bds__(max_threads_per_block, min_blocks_per_multiprocessor) \
  __launch_bounds__(max_threads_per_block, min_blocks_per_multiprocessor)
#else
#define __Lch_Bds__(max_threads_per_block, min_blocks_per_multiprocessor)
#endif

#define FULL_MASK 0xFFFFFFFF

#define PREPROCESSED_CONCAT_(x, y) x##y
#define PREPROCESSED_CONCAT(x, y) PREPROCESSED_CONCAT_(x, y)

#define LAUNCH_KERNEL_1D(kernel_name, grid_received, block_received,           \
                         size_t_shared_memory, size, ...)                      \
  if (grid_received.x * block_received.x < size) {                             \
    PREPROCESSED_CONCAT(                                                       \
        kernel_while__,                                                        \
        kernel_name)<<<grid_received, block_received, size_t_shared_memory>>>( \
        size, __VA_ARGS__);                                                    \
  } else if (grid_received.x * block_received.x > size) {                      \
    PREPROCESSED_CONCAT(                                                       \
        kernel__,                                                              \
        kernel_name)<<<grid_received, block_received, size_t_shared_memory>>>( \
        size, __VA_ARGS__);                                                    \
  } else {                                                                     \
    PREPROCESSED_CONCAT(                                                       \
        kernel__,                                                              \
        kernel_name)<<<grid_received, block_received, size_t_shared_memory>>>( \
        __VA_ARGS__);                                                          \
  }

#define LAUNCH_KERNEL_POINTER_1D(kernel_name, ptr_grid, ptr_block,     \
                                 size_t_shared_memory, size, ...)      \
  if (ptr_grid->x * ptr_block->x < size) {                             \
    PREPROCESSED_CONCAT(                                               \
        kernel_while__,                                                \
        kernel_name)<<<*ptr_grid, *ptr_block, size_t_shared_memory>>>( \
        size, __VA_ARGS__);                                            \
  } else if (ptr_grid->x * ptr_block->x > size) {                      \
    PREPROCESSED_CONCAT(                                               \
        kernel__,                                                      \
        kernel_name)<<<*ptr_grid, *ptr_block, size_t_shared_memory>>>( \
        size, __VA_ARGS__);                                            \
  } else {                                                             \
    PREPROCESSED_CONCAT(                                               \
        kernel__,                                                      \
        kernel_name)<<<*ptr_grid, *ptr_block, size_t_shared_memory>>>( \
        __VA_ARGS__);                                                  \
  }

#ifdef _DEBUG
#define CUDA__Safe_Call(err) cuda_run(err, __FILE__, __LINE__)
#define CUDA__Check_Error() cuda_sync(__FILE__, __LINE__)
#define CUDA__Last_Error() cuda_last_err(__FILE__, __LINE__)
#else
#define CUDA__Safe_Call(err) expr
#define CUDA__Check_Error() cudaDeviceSynchronize()
#define CUDA__Last_Error()
#endif

__host__ __device__ static inline void cuda_run(cudaError const err,
                                                char const *const file_name,
                                                int const line_pos) {
  if (cudaError::cudaSuccess != err) {
    ERR(L"Failed at %ls:%i: %ls _%d",
                 file_name, line_pos, cudaGetErrorString(err), err);
  }
}
__host__ __device__ static inline void cuda_sync(char const *const file_name,
                                                 int const line_pos) {
  cudaError tmp_cudaError(cudaDeviceSynchronize());
  if (cudaError::cudaSuccess != tmp_cudaError) {
    ERR(L"Synchronization failed at %ls:%i: %ls _%d", file_name, line_pos,
                 cudaGetErrorString(tmp_cudaError), tmp_cudaError);
  } else if (cudaError::cudaSuccess != (tmp_cudaError = cudaGetLastError())) {
    ERR(L"Failed at %ls:%i: %ls _%d",
                 file_name, line_pos, cudaGetErrorString(tmp_cudaError),
                 tmp_cudaError);
  }
}
__host__ __device__ static inline void cuda_last_err(
    char const *const file_name, int const line_pos) {
  cudaError const tmp_error(cudaGetLastError());
  if (cudaError::cudaSuccess != tmp_error) {
    ERR(L"Failed at %ls:%i: %ls _%d",
                 file_name, line_pos, cudaGetErrorString(tmp_error), tmp_error);
  }
}

__device__ __forceinline__ void CUDA__ThreadBlockSynchronize(void) {
#ifdef __CUDA_ARCH__
  __syncthreads();

  if (threadIdx.x == 0u) {
    CUDA__Check_Error();
  }

  __syncthreads();
#endif
}

enum ENUM_TYPE_MEMORY_ALLOCATE : unsigned int {
  TYPE_MEMORY_ALLOCATE_UNKNOW = 0,
  TYPE_MEMORY_ALLOCATE_CPU = 1,
  TYPE_MEMORY_ALLOCATE_GPU = 2,
  TYPE_MEMORY_ALLOCATE_MANAGED = 3u
};

enum ENUM_TYPE_DEVICE_SYNCHRONIZED : unsigned int {
  TYPE_DEVICE_SYNCHRONIZED_NONE = 0,
  TYPE_DEVICE_SYNCHRONIZED_THREAD = 1,
  TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK = 2u
};

__device__ __forceinline__ void CUDA__Device_Synchronise(
    enum ENUM_TYPE_DEVICE_SYNCHRONIZED const type_device_synchronise_received) {
  switch (type_device_synchronise_received) {
    case ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD:
      CUDA__Check_Error();
      break;
    case ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK:
      CUDA__ThreadBlockSynchronize();
      break;
    default:
      break;
  }
}

__device__ __forceinline__ void CUDA__Device_Synchronise(
    bool &ref_synchronized_received,
    enum ENUM_TYPE_DEVICE_SYNCHRONIZED const type_device_synchronise_received) {
  switch (type_device_synchronise_received) {
    case ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD:
      if (ref_synchronized_received == false) {
        ref_synchronized_received = true;

        CUDA__Check_Error();
      }
      break;
    case ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREADBLOCK:
      if (ref_synchronized_received == false) {
        ref_synchronized_received = true;

        CUDA__ThreadBlockSynchronize();
      }
      break;
    default:
      break;
  }
}

void CUDA__Initialize__Device(struct cudaDeviceProp const &device,
                              size_t const memory_allocate);

void CUDA__Set__Device(int const device_index);

void CUDA__Set__Synchronization_Depth(size_t const depth);

void CUDA__Reset(void);
