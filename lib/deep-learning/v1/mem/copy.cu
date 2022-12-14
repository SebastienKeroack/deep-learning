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

#define LAUNCH_KERNEL_POINTER_1D(                                              \
    kernel_name_received, ptr_grid_received, ptr_block_received,               \
    size_t_shared_memory_received, size_received, ...)                         \
  if (ptr_grid_received->x * ptr_block_received->x < size_received) {          \
    PREPROCESSED_CONCAT(                                                       \
        kernel_while__,                                                        \
        kernel_name_received)<<<*ptr_grid_received, *ptr_block_received,       \
                                size_t_shared_memory_received>>>(              \
        size_received, __VA_ARGS__);                                           \
  } else if (ptr_grid_received->x * ptr_block_received->x > size_received) {   \
    PREPROCESSED_CONCAT(                                                       \
        kernel__,                                                              \
        kernel_name_received)<<<*ptr_grid_received, *ptr_block_received,       \
                                size_t_shared_memory_received>>>(              \
        size_received, __VA_ARGS__);                                           \
  } else {                                                                     \
    PREPROCESSED_CONCAT(                                                       \
        kernel__,                                                              \
        kernel_name_received)<<<*ptr_grid_received, *ptr_block_received,       \
                                size_t_shared_memory_received>>>(__VA_ARGS__); \
  }

namespace DL::v1::Mem {
template <typename T>
__global__ void kernel__Memory_Copy_1D(
    T *const ptr_array_destination_received,
    T const *const ptr_array_source_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_destination_received[tmp_thread_global_index] =
      ptr_array_source_received[tmp_thread_global_index];
}

template <typename T>
__global__ void kernel__Memory_Copy_1D(
    size_t const size_received, T *const ptr_array_destination_received,
    T const *const ptr_array_source_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    ptr_array_destination_received[tmp_thread_global_index] =
        ptr_array_source_received[tmp_thread_global_index];
  }
}

template <typename T>
__global__ void kernel_while__Memory_Copy_1D(
    size_t const size_received, T *const ptr_array_destination_received,
    T const *const ptr_array_source_received) {
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_destination_received[tmp_thread_global_index] =
        ptr_array_source_received[tmp_thread_global_index];

    tmp_thread_global_index += gridDim.x * blockDim.x;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__device__ void Memory_Copy_1D(
    size_t const size_received, T *const ptr_array_destination_received,
    T const *const ptr_array_source_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize * warpSize) {
    LAUNCH_KERNEL_POINTER_1D(Memory_Copy_1D<T>, ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             ptr_array_destination_received,
                             ptr_array_source_received)
  } else {
    for (size_t i(0_UZ); i != size_received; ++i) {
      ptr_array_destination_received[i] = ptr_array_source_received[i];
    }
  }
}
}  // namespace DL::Mem
