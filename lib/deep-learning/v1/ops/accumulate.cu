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

namespace DL::v1::Math {
template <typename T>
__global__ void kernel__Accumulate_X_X_1D(
    T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_outputs_received[tmp_thread_global_index] +=
      ptr_array_inputs_received[tmp_thread_global_index];
}

template <typename T>
__global__ void kernel__Accumulate_X_X_1D(
    size_t const size_received, T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    ptr_array_outputs_received[tmp_thread_global_index] +=
        ptr_array_inputs_received[tmp_thread_global_index];
  }
}

template <typename T>
__global__ void kernel_while__Accumulate_X_X_1D(
    size_t const size_received, T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_outputs_received[tmp_thread_global_index] +=
        ptr_array_inputs_received[tmp_thread_global_index];

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__device__ void Accumulate_X_X_1D(
    size_t const size_received, T *ptr_array_outputs_received,
    T const *ptr_array_inputs_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(Accumulate_X_X_1D<T>, ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             ptr_array_outputs_received,
                             ptr_array_inputs_received)
  } else {
    for (T const *const ptr_output_end(ptr_array_outputs_received +
                                       size_received);
         ptr_array_outputs_received != ptr_output_end;
         ++ptr_array_outputs_received, ++ptr_array_inputs_received) {
      *ptr_array_outputs_received += *ptr_array_inputs_received;
    }
  }
}

}  // namespace Accumulate