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

#include "deep-learning-lib/v1/data/shared_memory.cuh"
#include "deep-learning-lib/v1/ops/accumulate.cuh"

namespace DL::v1::Math {
template <typename T>
__device__ inline void Reduce_Loop(size_t const size_received,
                                   T *const ptr_array_outputs_received,
                                   T const *ptr_array_inputs_received) {
  T const *tmp_ptr_last_element(ptr_array_inputs_received + size_received);
  T tmp_summation(0);

  for (; ptr_array_inputs_received != tmp_ptr_last_element;
       ++ptr_array_inputs_received) {
    tmp_summation += *ptr_array_inputs_received;
  }

  *ptr_array_outputs_received = tmp_summation;
}

DECLARE_EXTERN_SHARED_MEMORY_TEMPLATE(struct_kernel__Reduce)

template <typename T, size_t BLOCK_SIZE>
__global__ void kernel__Reduce(size_t size_received,
                               T *const ptr_array_outputs_received,
                               T const *const ptr_array_inputs_received) {
  // [0...1...1023]
  size_t const &tmp_thread_block_index(threadIdx.x),
      tmp_grid_stride(gridDim.x * BLOCK_SIZE * 2u);
  // 0 * 1024 * 2 + [0...1...1023] = 0 + [0...1...1023]
  // 1 * 1024 * 2 + [0...1...1023] = 2048 + [0...1...1023]
  // 2 * 1024 * 2 + [0...1...1023] = 4096 + [0...1...1023]
  // 3 * 1024 * 2 + [0...1...1023] = 6144 + [0...1...1023]
  size_t tmp_thread_global_index(blockIdx.x * BLOCK_SIZE * 2u +
                                 tmp_thread_block_index),
      tmp_thread_global_index_offset(tmp_thread_global_index + BLOCK_SIZE);

  T tmp_thread_reduced_value, *tmp_ptr_array_reduced;

  EXTERN_SHARED_MEMORY_TEMPLATE(T, tmp_ptr_array_reduced, struct_kernel__Reduce)

  tmp_ptr_array_reduced[tmp_thread_block_index] = T(0);

  // Add by two load from GMEM.
  do {
    // BlockIdx.x[0]: [0...1...1023] += [0...1...1023] + [0...1...1023] + 1024
    // BlockIdx.x[1]: [0...1...1023] += [2048...2049...3071] +
    // [2048...2049...3071] + 1024 BlockIdx.x[2]: [0...1...1023] +=
    // [4096...4097...5019] + [4096...4097...5019] + 1024 BlockIdx.x[3]:
    // [0...1...1023] += [6144...6145...7167] + [6144...6145...7167] + 1024
    tmp_ptr_array_reduced[tmp_thread_block_index] +=
        ptr_array_inputs_received[tmp_thread_global_index] +
        ptr_array_inputs_received[tmp_thread_global_index_offset];

    // BlockIdx.x[0]: [0...1...1023] += 4 * 1024 * 2
    // BlockIdx.x[1]: [2048...2049...3071] += 4 * 1024 * 2
    // BlockIdx.x[2]: [4096...4097...5019] += 4 * 1024 * 2
    // BlockIdx.x[3]: [6144...6145...7167] += 4 * 1024 * 2
    tmp_thread_global_index += tmp_grid_stride;
    tmp_thread_global_index_offset += tmp_grid_stride;
  } while (tmp_thread_global_index_offset < size_received);

  // Add by one load from GMEM (Remaining elements).
  if (tmp_thread_global_index < size_received) {
    // BlockIdx.x[0]: [0...1...1023] += [0...1...1023] + [0...1...1023] + 1024
    // BlockIdx.x[1]: [0...1...1023] += [2048...2049...3071] +
    // [2048...2049...3071] + 1024 BlockIdx.x[2]: [0...1...1023] +=
    // [4096...4097...5019] + [4096...4097...5019] + 1024 BlockIdx.x[3]:
    // [0...1...1023] += [6144...6145...7167] + [6144...6145...7167] + 1024
    tmp_ptr_array_reduced[tmp_thread_block_index] +=
        ptr_array_inputs_received[tmp_thread_global_index];
  }

  __syncthreads();

  if (BLOCK_SIZE >= 1024u) {
    if (tmp_thread_block_index < 512u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 512u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 512u) {
    if (tmp_thread_block_index < 256u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 256u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 256u) {
    if (tmp_thread_block_index < 128u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 128u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 128u) {
    if (tmp_thread_block_index < 64u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 64u];
    }

    __syncthreads();
  }

  // Last warp.
  if (tmp_thread_block_index < warpSize) {
    switch (BLOCK_SIZE) {
      case 32:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 16);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 8);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 16:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 8u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 8:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 4u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 4:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 2u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 2:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 1u];
        __syncwarp();
        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 1:
        break;
      default:  // BLOCK_SIZE >= 64
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 32u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 16);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 8);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
    }
  }

  if (tmp_thread_block_index == 0u) {
    ptr_array_outputs_received[blockIdx.x] = tmp_ptr_array_reduced[0];
  }
}

template <typename T>
__device__ void Launch_Reduce(
    size_t const size_received, T *const ptr_array_outputs_recieved,
    T const *const ptr_array_inputs_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved) {
  switch (ptr_dimension_block_recieved->x) {
    case 1024:
      kernel__Reduce<T, 1024u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             1024u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                  ptr_array_inputs_received);
      break;
    case 512:
      kernel__Reduce<T, 512u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             512u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_received);
      break;
    case 256:
      kernel__Reduce<T, 256u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             256u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_received);
      break;
    case 128:
      kernel__Reduce<T, 128u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             128u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_received);
      break;
    case 64:
      kernel__Reduce<T, 64u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             64u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                ptr_array_inputs_received);
      break;
    case 32:
      kernel__Reduce<T, 32u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (32u + 16u) * sizeof(T)>>>(size_received,
                                        ptr_array_outputs_recieved,
                                        ptr_array_inputs_received);
      break;
    case 16:
      kernel__Reduce<T, 16u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (16u + 8u) * sizeof(T)>>>(size_received,
                                       ptr_array_outputs_recieved,
                                       ptr_array_inputs_received);
      break;
    case 8:
      kernel__Reduce<T, 8u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (8u + 4u) * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                      ptr_array_inputs_received);
      break;
    case 4:
      kernel__Reduce<T, 4u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (4u + 2u) * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                      ptr_array_inputs_received);
      break;
    case 2:
      kernel__Reduce<T, 2u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (2u + 1u) * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                      ptr_array_inputs_received);
      break;
    case 1:
      kernel__Reduce<T, 1u><<<*ptr_dimension_grid_recieved,
                              *ptr_dimension_block_recieved, sizeof(T)>>>(
          size_received, ptr_array_outputs_recieved, ptr_array_inputs_received);
      break;
  }
}

template <typename T>
__device__ void Reduce(size_t const size_received,
                       size_t const stride_dim3_received,
                       T *const ptr_array_outputs_received,
                       T const *const ptr_array_inputs_received,
                       struct dim3 const *const ptr_dimension_grid_recieved,
                       struct dim3 const *const ptr_dimension_block_recieved) {
  if (size_received > 1u) {
    if (USE_PARALLEL && size_received >= warpSize * 2u) {
      size_t tmp_iteration(0u), tmp_number_elements_to_reduce;

      Launch_Reduce<T>(size_received, ptr_array_outputs_received,
                       ptr_array_inputs_received, ptr_dimension_grid_recieved,
                       ptr_dimension_block_recieved);

      tmp_number_elements_to_reduce = ptr_dimension_grid_recieved->x;

      while (tmp_number_elements_to_reduce != 1u) {
        if (tmp_number_elements_to_reduce >= warpSize * 2u) {
          ++tmp_iteration;

          Launch_Reduce<T>(tmp_number_elements_to_reduce,
                           ptr_array_outputs_received,
                           ptr_array_outputs_received,
                           ptr_dimension_grid_recieved +
                               tmp_iteration * stride_dim3_received,
                           ptr_dimension_block_recieved +
                               tmp_iteration * stride_dim3_received);

          tmp_number_elements_to_reduce =
              ptr_dimension_grid_recieved[tmp_iteration * stride_dim3_received]
                  .x;
        } else {
          Reduce_Loop(tmp_number_elements_to_reduce, ptr_array_outputs_received,
                      ptr_array_outputs_received);

          tmp_number_elements_to_reduce = 1u;
        }
      }
    } else {
      T tmp_summation(0);

      for (size_t i(0_UZ); i != size_received; ++i) {
        tmp_summation += ptr_array_inputs_received[i];
      }

      *ptr_array_outputs_received = tmp_summation;
    }
  } else if (size_received == 1u) {
    *ptr_array_outputs_received = *ptr_array_inputs_received;
  } else {
    ERR(L"No element to reduce!",);
  }
}

DECLARE_EXTERN_SHARED_MEMORY_TEMPLATE(struct_kernel__Reduce_Square)

template <typename T, size_t BLOCK_SIZE>
__global__ void kernel__Reduce_Square(
    size_t size_received, T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_received) {
  // [0...1...1023]
  size_t const &tmp_thread_block_index(threadIdx.x),
      tmp_grid_stride(gridDim.x * BLOCK_SIZE * 2u);
  // 0 * 1024 * 2 + [0...1...1023] = 0 + [0...1...1023]
  // 1 * 1024 * 2 + [0...1...1023] = 2048 + [0...1...1023]
  // 2 * 1024 * 2 + [0...1...1023] = 4096 + [0...1...1023]
  // 3 * 1024 * 2 + [0...1...1023] = 6144 + [0...1...1023]
  size_t tmp_thread_global_index(blockIdx.x * BLOCK_SIZE * 2u +
                                 tmp_thread_block_index),
      tmp_thread_global_index_offset(tmp_thread_global_index + BLOCK_SIZE);

  T tmp_thread_reduced_value_0, tmp_thread_reduced_value_1,
      *tmp_ptr_array_reduced;

  EXTERN_SHARED_MEMORY_TEMPLATE(T, tmp_ptr_array_reduced,
                                struct_kernel__Reduce_Square)

  tmp_ptr_array_reduced[tmp_thread_block_index] = T(0);

  // Add by two load from GMEM.
  do {
    // BlockIdx.x[0]: [0...1...1023] += [0...1...1023] + [0...1...1023] + 1024
    // BlockIdx.x[1]: [0...1...1023] += [2048...2049...3071] +
    // [2048...2049...3071] + 1024 BlockIdx.x[2]: [0...1...1023] +=
    // [4096...4097...5019] + [4096...4097...5019] + 1024 BlockIdx.x[3]:
    // [0...1...1023] += [6144...6145...7167] + [6144...6145...7167] + 1024
    tmp_thread_reduced_value_0 =
        ptr_array_inputs_received[tmp_thread_global_index];
    tmp_thread_reduced_value_1 =
        ptr_array_inputs_received[tmp_thread_global_index_offset];
    tmp_ptr_array_reduced[tmp_thread_block_index] +=
        tmp_thread_reduced_value_0 * tmp_thread_reduced_value_0 +
        tmp_thread_reduced_value_1 * tmp_thread_reduced_value_1;

    // BlockIdx.x[0]: [0...1...1023] += 4 * 1024 * 2
    // BlockIdx.x[1]: [2048...2049...3071] += 4 * 1024 * 2
    // BlockIdx.x[2]: [4096...4097...5019] += 4 * 1024 * 2
    // BlockIdx.x[3]: [6144...6145...7167] += 4 * 1024 * 2
    tmp_thread_global_index += tmp_grid_stride;
    tmp_thread_global_index_offset += tmp_grid_stride;
  } while (tmp_thread_global_index_offset < size_received);

  // Add by one load from GMEM (Remaining elements).
  if (tmp_thread_global_index < size_received) {
    // BlockIdx.x[0]: [0...1...1023] += [0...1...1023] + [0...1...1023] + 1024
    // BlockIdx.x[1]: [0...1...1023] += [2048...2049...3071] +
    // [2048...2049...3071] + 1024 BlockIdx.x[2]: [0...1...1023] +=
    // [4096...4097...5019] + [4096...4097...5019] + 1024 BlockIdx.x[3]:
    // [0...1...1023] += [6144...6145...7167] + [6144...6145...7167] + 1024
    tmp_thread_reduced_value_0 =
        ptr_array_inputs_received[tmp_thread_global_index];
    tmp_ptr_array_reduced[tmp_thread_block_index] +=
        tmp_thread_reduced_value_0 * tmp_thread_reduced_value_0;
  }

  __syncthreads();

  if (BLOCK_SIZE >= 1024u) {
    if (tmp_thread_block_index < 512u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 512u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 512u) {
    if (tmp_thread_block_index < 256u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 256u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 256u) {
    if (tmp_thread_block_index < 128u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 128u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 128u) {
    if (tmp_thread_block_index < 64u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 64u];
    }

    __syncthreads();
  }

  // Last warp.
  if (tmp_thread_block_index < warpSize) {
    switch (BLOCK_SIZE) {
      case 32:
        tmp_thread_reduced_value_0 =
            tmp_ptr_array_reduced[tmp_thread_block_index];
        __syncwarp();

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 16);

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 8);

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 4);

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 2);

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value_0;
        break;
      case 16:
        tmp_thread_reduced_value_0 =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 8u];
        __syncwarp();

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 4);

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 2);

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value_0;
        break;
      case 8:
        tmp_thread_reduced_value_0 =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 4u];
        __syncwarp();

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 2);

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value_0;
        break;
      case 4:
        tmp_thread_reduced_value_0 =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 2u];
        __syncwarp();

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value_0;
        break;
      case 2:
        tmp_thread_reduced_value_0 =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 1u];
        __syncwarp();
        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value_0;
        break;
      case 1:
        break;
      default:  // BLOCK_SIZE >= 64
        tmp_thread_reduced_value_0 =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 32u];
        __syncwarp();

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 16);

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 8);

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 4);

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 2);

        tmp_thread_reduced_value_0 +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value_0, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value_0;
        break;
    }
  }

  if (tmp_thread_block_index == 0u) {
    ptr_array_outputs_received[blockIdx.x] = tmp_ptr_array_reduced[0];
  }
}

template <typename T>
__device__ void Launch_Reduce_Square(
    size_t const size_received, T *const ptr_array_outputs_recieved,
    T const *const ptr_array_inputs_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved) {
  switch (ptr_dimension_block_recieved->x) {
    case 1024:
      kernel__Reduce_Square<T, 1024u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             1024u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                  ptr_array_inputs_received);
      break;
    case 512:
      kernel__Reduce_Square<T, 512u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             512u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_received);
      break;
    case 256:
      kernel__Reduce_Square<T, 256u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             256u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_received);
      break;
    case 128:
      kernel__Reduce_Square<T, 128u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             128u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_received);
      break;
    case 64:
      kernel__Reduce_Square<T, 64u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             64u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                ptr_array_inputs_received);
      break;
    case 32:
      kernel__Reduce_Square<T, 32u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (32u + 16u) * sizeof(T)>>>(size_received,
                                        ptr_array_outputs_recieved,
                                        ptr_array_inputs_received);
      break;
    case 16:
      kernel__Reduce_Square<T, 16u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (16u + 8u) * sizeof(T)>>>(size_received,
                                       ptr_array_outputs_recieved,
                                       ptr_array_inputs_received);
      break;
    case 8:
      kernel__Reduce_Square<T, 8u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (8u + 4u) * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                      ptr_array_inputs_received);
      break;
    case 4:
      kernel__Reduce_Square<T, 4u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (4u + 2u) * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                      ptr_array_inputs_received);
      break;
    case 2:
      kernel__Reduce_Square<T, 2u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (2u + 1u) * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                      ptr_array_inputs_received);
      break;
    case 1:
      kernel__Reduce_Square<T, 1u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                          ptr_array_inputs_received);
      break;
  }
}

template <typename T>
__device__ void Reduce_Square(
    size_t const size_received, size_t const stride_dim3_received,
    T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved) {
  T tmp_value_square;

  if (size_received > 1u) {
    if (USE_PARALLEL && size_received >= warpSize * 2u) {
      size_t tmp_iteration(0u), tmp_number_elements_to_reduce;

      Launch_Reduce_Square<T>(
          size_received, ptr_array_outputs_received, ptr_array_inputs_received,
          ptr_dimension_grid_recieved, ptr_dimension_block_recieved);

      tmp_number_elements_to_reduce = ptr_dimension_grid_recieved->x;

      while (tmp_number_elements_to_reduce != 1u) {
        if (tmp_number_elements_to_reduce >= warpSize * 2u) {
          ++tmp_iteration;

          Launch_Reduce<T>(tmp_number_elements_to_reduce,
                           ptr_array_outputs_received,
                           ptr_array_outputs_received,
                           ptr_dimension_grid_recieved +
                               tmp_iteration * stride_dim3_received,
                           ptr_dimension_block_recieved +
                               tmp_iteration * stride_dim3_received);

          tmp_number_elements_to_reduce =
              ptr_dimension_grid_recieved[tmp_iteration * stride_dim3_received]
                  .x;
        } else {
          Reduce_Loop(tmp_number_elements_to_reduce, ptr_array_outputs_received,
                      ptr_array_outputs_received);

          tmp_number_elements_to_reduce = 1u;
        }
      }
    } else {
      T tmp_summation(0);

      for (size_t i(0_UZ); i != size_received; ++i) {
        tmp_value_square = ptr_array_inputs_received[i];

        tmp_summation += tmp_value_square * tmp_value_square;
      }

      *ptr_array_outputs_received = tmp_summation;
    }
  } else if (size_received == 1u) {
    tmp_value_square = *ptr_array_inputs_received;

    *ptr_array_outputs_received = tmp_value_square * tmp_value_square;
  } else {
    ERR(L"No element to reduce!",);
  }
}

DECLARE_EXTERN_SHARED_MEMORY_TEMPLATE(struct_kernel__Reduce_XX)

template <typename T, size_t BLOCK_SIZE>
__global__ void kernel__Reduce_XX(size_t size_received,
                                  T *const ptr_array_outputs_received,
                                  T const *const ptr_array_inputs_X0_received,
                                  T const *const ptr_array_inputs_X1_received) {
  // [0...1...1023]
  size_t const &tmp_thread_block_index(threadIdx.x),
      tmp_grid_stride(gridDim.x * BLOCK_SIZE * 2u);
  // 0 * 1024 * 2 + [0...1...1023] = 0 + [0...1...1023]
  // 1 * 1024 * 2 + [0...1...1023] = 2048 + [0...1...1023]
  // 2 * 1024 * 2 + [0...1...1023] = 4096 + [0...1...1023]
  // 3 * 1024 * 2 + [0...1...1023] = 6144 + [0...1...1023]
  size_t tmp_thread_global_index(blockIdx.x * BLOCK_SIZE * 2u +
                                 tmp_thread_block_index),
      tmp_thread_global_index_offset(tmp_thread_global_index + BLOCK_SIZE);

  T tmp_thread_reduced_value, *tmp_ptr_array_reduced;

  EXTERN_SHARED_MEMORY_TEMPLATE(T, tmp_ptr_array_reduced,
                                struct_kernel__Reduce_XX)

  tmp_ptr_array_reduced[tmp_thread_block_index] = T(0);

  // Add by two load from GMEM.
  do {
    // BlockIdx.x[0]: [0...1...1023] += [0...1...1023] + [0...1...1023] + 1024
    // BlockIdx.x[1]: [0...1...1023] += [2048...2049...3071] +
    // [2048...2049...3071] + 1024 BlockIdx.x[2]: [0...1...1023] +=
    // [4096...4097...5019] + [4096...4097...5019] + 1024 BlockIdx.x[3]:
    // [0...1...1023] += [6144...6145...7167] + [6144...6145...7167] + 1024
    tmp_ptr_array_reduced[tmp_thread_block_index] +=
        ptr_array_inputs_X0_received[tmp_thread_global_index] *
            ptr_array_inputs_X1_received[tmp_thread_global_index] +
        ptr_array_inputs_X0_received[tmp_thread_global_index_offset] *
            ptr_array_inputs_X1_received[tmp_thread_global_index_offset];

    // BlockIdx.x[0]: [0...1...1023] += 4 * 1024 * 2
    // BlockIdx.x[1]: [2048...2049...3071] += 4 * 1024 * 2
    // BlockIdx.x[2]: [4096...4097...5019] += 4 * 1024 * 2
    // BlockIdx.x[3]: [6144...6145...7167] += 4 * 1024 * 2
    tmp_thread_global_index += tmp_grid_stride;
    tmp_thread_global_index_offset += tmp_grid_stride;
  } while (tmp_thread_global_index_offset < size_received);

  // Add by one load from GMEM (Remaining elements).
  if (tmp_thread_global_index < size_received) {
    // BlockIdx.x[0]: [0...1...1023] += [0...1...1023] + [0...1...1023] + 1024
    // BlockIdx.x[1]: [0...1...1023] += [2048...2049...3071] +
    // [2048...2049...3071] + 1024 BlockIdx.x[2]: [0...1...1023] +=
    // [4096...4097...5019] + [4096...4097...5019] + 1024 BlockIdx.x[3]:
    // [0...1...1023] += [6144...6145...7167] + [6144...6145...7167] + 1024
    tmp_ptr_array_reduced[tmp_thread_block_index] +=
        ptr_array_inputs_X0_received[tmp_thread_global_index] *
        ptr_array_inputs_X1_received[tmp_thread_global_index];
  }

  __syncthreads();

  if (BLOCK_SIZE >= 1024u) {
    if (tmp_thread_block_index < 512u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 512u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 512u) {
    if (tmp_thread_block_index < 256u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 256u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 256u) {
    if (tmp_thread_block_index < 128u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 128u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 128u) {
    if (tmp_thread_block_index < 64u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 64u];
    }

    __syncthreads();
  }

  // Last warp.
  if (tmp_thread_block_index < warpSize) {
    switch (BLOCK_SIZE) {
      case 32:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 16);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 8);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 16:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 8u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 8:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 4u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 4:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 2u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 2:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 1u];
        __syncwarp();
        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 1:
        break;
      default:  // BLOCK_SIZE >= 64
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 32u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 16);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 8);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
    }
  }

  if (tmp_thread_block_index == 0u) {
    ptr_array_outputs_received[blockIdx.x] = tmp_ptr_array_reduced[0];
  }
}

template <typename T>
__device__ void Launch_Reduce_XX(
    size_t const size_received, T *const ptr_array_outputs_recieved,
    T const *const ptr_array_inputs_X0_received,
    T const *const ptr_array_inputs_X1_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved) {
  switch (ptr_dimension_block_recieved->x) {
    case 1024:
      kernel__Reduce_XX<T, 1024u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             1024u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                  ptr_array_inputs_X0_received,
                                  ptr_array_inputs_X1_received);
      break;
    case 512:
      kernel__Reduce_XX<T, 512u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             512u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_X0_received,
                                 ptr_array_inputs_X1_received);
      break;
    case 256:
      kernel__Reduce_XX<T, 256u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             256u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_X0_received,
                                 ptr_array_inputs_X1_received);
      break;
    case 128:
      kernel__Reduce_XX<T, 128u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             128u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_X0_received,
                                 ptr_array_inputs_X1_received);
      break;
    case 64:
      kernel__Reduce_XX<T, 64u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             64u * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                ptr_array_inputs_X0_received,
                                ptr_array_inputs_X1_received);
      break;
    case 32:
      kernel__Reduce_XX<T, 32u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (32u + 16u) * sizeof(T)>>>(
              size_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X0_received, ptr_array_inputs_X1_received);
      break;
    case 16:
      kernel__Reduce_XX<T, 16u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (16u + 8u) * sizeof(T)>>>(
              size_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X0_received, ptr_array_inputs_X1_received);
      break;
    case 8:
      kernel__Reduce_XX<T, 8u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (8u + 4u) * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                      ptr_array_inputs_X0_received,
                                      ptr_array_inputs_X1_received);
      break;
    case 4:
      kernel__Reduce_XX<T, 4u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (4u + 2u) * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                      ptr_array_inputs_X0_received,
                                      ptr_array_inputs_X1_received);
      break;
    case 2:
      kernel__Reduce_XX<T, 2u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (2u + 1u) * sizeof(T)>>>(size_received, ptr_array_outputs_recieved,
                                      ptr_array_inputs_X0_received,
                                      ptr_array_inputs_X1_received);
      break;
    case 1:
      kernel__Reduce_XX<T, 1u><<<*ptr_dimension_grid_recieved,
                                 *ptr_dimension_block_recieved, sizeof(T)>>>(
          size_received, ptr_array_outputs_recieved,
          ptr_array_inputs_X0_received, ptr_array_inputs_X1_received);
      break;
  }
}

template <typename T>
__device__ void Reduce_XX(
    size_t const size_received, size_t const stride_dim3_received,
    T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_X0_received,
    T const *const ptr_array_inputs_X1_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved) {
  if (size_received > 1u) {
    if (size_received >= warpSize * 2u) {
      size_t tmp_iteration(0u), tmp_number_elements_to_reduce;

      Launch_Reduce_XX<T>(
          size_received, ptr_array_outputs_received,
          ptr_array_inputs_X0_received, ptr_array_inputs_X1_received,
          ptr_dimension_grid_recieved, ptr_dimension_block_recieved);

      tmp_number_elements_to_reduce = ptr_dimension_grid_recieved->x;

      while (tmp_number_elements_to_reduce != 1u) {
        if (tmp_number_elements_to_reduce >= warpSize * 2u) {
          ++tmp_iteration;

          Launch_Reduce<T>(tmp_number_elements_to_reduce,
                           ptr_array_outputs_received,
                           ptr_array_outputs_received,
                           ptr_dimension_grid_recieved +
                               tmp_iteration * stride_dim3_received,
                           ptr_dimension_block_recieved +
                               tmp_iteration * stride_dim3_received);

          tmp_number_elements_to_reduce =
              ptr_dimension_grid_recieved[tmp_iteration * stride_dim3_received]
                  .x;
        } else {
          Reduce_Loop(tmp_number_elements_to_reduce, ptr_array_outputs_received,
                      ptr_array_outputs_received);

          tmp_number_elements_to_reduce = 1u;
        }
      }
    } else {
      T tmp_summation(0);

      for (size_t i(0_UZ); i != size_received; ++i) {
        tmp_summation +=
            ptr_array_inputs_X0_received[i] * ptr_array_inputs_X1_received[i];
      }

      *ptr_array_outputs_received = tmp_summation;
    }
  } else if (size_received == 1u) {
    *ptr_array_outputs_received =
        *ptr_array_inputs_X0_received * *ptr_array_inputs_X1_received;
  } else {
    ERR(L"No element to reduce!",);
  }
}

DECLARE_EXTERN_SHARED_MEMORY_TEMPLATE(struct_kernel__Reduce_XZ)

template <typename T, size_t BLOCK_SIZE>
__global__ void kernel__Reduce_XZ(size_t size_received,
                                  size_t stride_Z_received,
                                  T *const ptr_array_outputs_received,
                                  T const *const ptr_array_inputs_X_received,
                                  T const *const ptr_array_inputs_Z_received) {
  // [0...1...1023]
  size_t const &tmp_thread_block_index(threadIdx.x),
      tmp_grid_stride(gridDim.x * BLOCK_SIZE * 2u);
  // 0 * 1024 * 2 + [0...1...1023] = 0 + [0...1...1023]
  // 1 * 1024 * 2 + [0...1...1023] = 2048 + [0...1...1023]
  // 2 * 1024 * 2 + [0...1...1023] = 4096 + [0...1...1023]
  // 3 * 1024 * 2 + [0...1...1023] = 6144 + [0...1...1023]
  size_t tmp_thread_global_index(blockIdx.x * BLOCK_SIZE * 2u +
                                 tmp_thread_block_index),
      tmp_thread_global_index_offset(tmp_thread_global_index + BLOCK_SIZE);

  T tmp_thread_reduced_value, *tmp_ptr_array_reduced;

  EXTERN_SHARED_MEMORY_TEMPLATE(T, tmp_ptr_array_reduced,
                                struct_kernel__Reduce_XZ)

  tmp_ptr_array_reduced[tmp_thread_block_index] = T(0);

  // Add by two load from GMEM.
  do {
    // BlockIdx.x[0]: [0...1...1023] += [0...1...1023] + [0...1...1023] + 1024
    // BlockIdx.x[1]: [0...1...1023] += [2048...2049...3071] +
    // [2048...2049...3071] + 1024 BlockIdx.x[2]: [0...1...1023] +=
    // [4096...4097...5019] + [4096...4097...5019] + 1024 BlockIdx.x[3]:
    // [0...1...1023] += [6144...6145...7167] + [6144...6145...7167] + 1024
    tmp_ptr_array_reduced[tmp_thread_block_index] +=
        ptr_array_inputs_X_received[tmp_thread_global_index] *
            ptr_array_inputs_Z_received[stride_Z_received *
                                        tmp_thread_global_index] +
        ptr_array_inputs_X_received[tmp_thread_global_index_offset] *
            ptr_array_inputs_Z_received[stride_Z_received *
                                        tmp_thread_global_index_offset];

    // BlockIdx.x[0]: [0...1...1023] += 4 * 1024 * 2
    // BlockIdx.x[1]: [2048...2049...3071] += 4 * 1024 * 2
    // BlockIdx.x[2]: [4096...4097...5019] += 4 * 1024 * 2
    // BlockIdx.x[3]: [6144...6145...7167] += 4 * 1024 * 2
    tmp_thread_global_index += tmp_grid_stride;
    tmp_thread_global_index_offset += tmp_grid_stride;
  } while (tmp_thread_global_index_offset < size_received);

  // Add by one load from GMEM (Remaining elements).
  if (tmp_thread_global_index < size_received) {
    // BlockIdx.x[0]: [0...1...1023] += [0...1...1023] + [0...1...1023] + 1024
    // BlockIdx.x[1]: [0...1...1023] += [2048...2049...3071] +
    // [2048...2049...3071] + 1024 BlockIdx.x[2]: [0...1...1023] +=
    // [4096...4097...5019] + [4096...4097...5019] + 1024 BlockIdx.x[3]:
    // [0...1...1023] += [6144...6145...7167] + [6144...6145...7167] + 1024
    tmp_ptr_array_reduced[tmp_thread_block_index] +=
        ptr_array_inputs_X_received[tmp_thread_global_index] *
        ptr_array_inputs_Z_received[stride_Z_received *
                                    tmp_thread_global_index];
  }

  __syncthreads();

  if (BLOCK_SIZE >= 1024u) {
    if (tmp_thread_block_index < 512u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 512u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 512u) {
    if (tmp_thread_block_index < 256u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 256u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 256u) {
    if (tmp_thread_block_index < 128u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 128u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 128u) {
    if (tmp_thread_block_index < 64u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 64u];
    }

    __syncthreads();
  }

  // Last warp.
  if (tmp_thread_block_index < warpSize) {
    switch (BLOCK_SIZE) {
      case 32:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 16);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 8);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 16:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 8u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 8:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 4u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 4:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 2u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 2:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 1u];
        __syncwarp();
        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 1:
        break;
      default:  // BLOCK_SIZE >= 64
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 32u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 16);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 8);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
    }
  }

  if (tmp_thread_block_index == 0u) {
    ptr_array_outputs_received[blockIdx.x] = tmp_ptr_array_reduced[0];
  }
}

template <typename T>
__device__ void Launch_Reduce_XZ(
    size_t const size_received, size_t const stride_Z_received,
    T *const ptr_array_outputs_recieved,
    T const *const ptr_array_inputs_X_received,
    T const *const ptr_array_inputs_Z_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved) {
  switch (ptr_dimension_block_recieved->x) {
    case 1024:
      kernel__Reduce_XZ<T, 1024u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             1024u * sizeof(T)>>>(
              size_received, stride_Z_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X_received, ptr_array_inputs_Z_received);
      break;
    case 512:
      kernel__Reduce_XZ<T, 512u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             512u * sizeof(T)>>>(
              size_received, stride_Z_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X_received, ptr_array_inputs_Z_received);
      break;
    case 256:
      kernel__Reduce_XZ<T, 256u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             256u * sizeof(T)>>>(
              size_received, stride_Z_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X_received, ptr_array_inputs_Z_received);
      break;
    case 128:
      kernel__Reduce_XZ<T, 128u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             128u * sizeof(T)>>>(
              size_received, stride_Z_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X_received, ptr_array_inputs_Z_received);
      break;
    case 64:
      kernel__Reduce_XZ<T, 64u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             64u * sizeof(T)>>>(
              size_received, stride_Z_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X_received, ptr_array_inputs_Z_received);
      break;
    case 32:
      kernel__Reduce_XZ<T, 32u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (32u + 16u) * sizeof(T)>>>(
              size_received, stride_Z_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X_received, ptr_array_inputs_Z_received);
      break;
    case 16:
      kernel__Reduce_XZ<T, 16u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (16u + 8u) * sizeof(T)>>>(
              size_received, stride_Z_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X_received, ptr_array_inputs_Z_received);
      break;
    case 8:
      kernel__Reduce_XZ<T, 8u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (8u + 4u) * sizeof(T)>>>(
              size_received, stride_Z_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X_received, ptr_array_inputs_Z_received);
      break;
    case 4:
      kernel__Reduce_XZ<T, 4u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (4u + 2u) * sizeof(T)>>>(
              size_received, stride_Z_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X_received, ptr_array_inputs_Z_received);
      break;
    case 2:
      kernel__Reduce_XZ<T, 2u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (2u + 1u) * sizeof(T)>>>(
              size_received, stride_Z_received, ptr_array_outputs_recieved,
              ptr_array_inputs_X_received, ptr_array_inputs_Z_received);
      break;
    case 1:
      kernel__Reduce_XZ<T, 1u><<<*ptr_dimension_grid_recieved,
                                 *ptr_dimension_block_recieved, sizeof(T)>>>(
          size_received, stride_Z_received, ptr_array_outputs_recieved,
          ptr_array_inputs_X_received, ptr_array_inputs_Z_received);
      break;
  }
}

template <typename T>
__device__ void Reduce_XZ(
    size_t const size_received, size_t const stride_dim3_received,
    size_t const stride_Z_received, T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_X_received,
    T const *const ptr_array_inputs_Z_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved) {
  if (size_received > 1u) {
    if (USE_PARALLEL && size_received >= warpSize * 2u) {
      size_t tmp_iteration(0u), tmp_number_elements_to_reduce;

      Launch_Reduce_XZ<T>(
          size_received, stride_Z_received, ptr_array_outputs_received,
          ptr_array_inputs_X_received, ptr_array_inputs_Z_received,
          ptr_dimension_grid_recieved, ptr_dimension_block_recieved);

      tmp_number_elements_to_reduce = ptr_dimension_grid_recieved->x;

      while (tmp_number_elements_to_reduce != 1u) {
        if (tmp_number_elements_to_reduce >= warpSize * 2u) {
          ++tmp_iteration;

          Launch_Reduce<T>(tmp_number_elements_to_reduce,
                           ptr_array_outputs_received,
                           ptr_array_outputs_received,
                           ptr_dimension_grid_recieved +
                               tmp_iteration * stride_dim3_received,
                           ptr_dimension_block_recieved +
                               tmp_iteration * stride_dim3_received);

          tmp_number_elements_to_reduce =
              ptr_dimension_grid_recieved[tmp_iteration * stride_dim3_received]
                  .x;
        } else {
          Reduce_Loop(tmp_number_elements_to_reduce, ptr_array_outputs_received,
                      ptr_array_outputs_received);

          tmp_number_elements_to_reduce = 1u;
        }
      }
    } else {
      T tmp_summation(0);

      for (size_t i(0_UZ); i != size_received; ++i) {
        tmp_summation += ptr_array_inputs_X_received[i] *
                         ptr_array_inputs_Z_received[i * stride_Z_received];
      }

      *ptr_array_outputs_received = tmp_summation;
    }
  } else if (size_received == 1u) {
    *ptr_array_outputs_received =
        *ptr_array_inputs_X_received * *ptr_array_inputs_Z_received;
  } else {
    ERR(L"No element to reduce!",);
  }
}

DECLARE_EXTERN_SHARED_MEMORY_TEMPLATE(struct_kernel__Reduce_Z0Z1)

template <typename T, size_t BLOCK_SIZE>
__global__ void kernel__Reduce_Z0Z1(
    size_t size_received, size_t stride_Z0_received, size_t stride_Z1_received,
    T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_Z0_received,
    T const *const ptr_array_inputs_Z1_received) {
  // [0...1...1023]
  size_t const &tmp_thread_block_index(threadIdx.x),
      tmp_grid_stride(gridDim.x * BLOCK_SIZE * 2u);
  // 0 * 1024 * 2 + [0...1...1023] = 0 + [0...1...1023]
  // 1 * 1024 * 2 + [0...1...1023] = 2048 + [0...1...1023]
  // 2 * 1024 * 2 + [0...1...1023] = 4096 + [0...1...1023]
  // 3 * 1024 * 2 + [0...1...1023] = 6144 + [0...1...1023]
  size_t tmp_thread_global_index(blockIdx.x * BLOCK_SIZE * 2u +
                                 tmp_thread_block_index),
      tmp_thread_global_index_offset(tmp_thread_global_index + BLOCK_SIZE);

  T tmp_thread_reduced_value, *tmp_ptr_array_reduced;

  EXTERN_SHARED_MEMORY_TEMPLATE(T, tmp_ptr_array_reduced,
                                struct_kernel__Reduce_Z0Z1)

  tmp_ptr_array_reduced[tmp_thread_block_index] = T(0);

  // Add by two load from GMEM.
  do {
    // BlockIdx.x[0]: [0...1...1023] += [0...1...1023] + [0...1...1023] + 1024
    // BlockIdx.x[1]: [0...1...1023] += [2048...2049...3071] +
    // [2048...2049...3071] + 1024 BlockIdx.x[2]: [0...1...1023] +=
    // [4096...4097...5019] + [4096...4097...5019] + 1024 BlockIdx.x[3]:
    // [0...1...1023] += [6144...6145...7167] + [6144...6145...7167] + 1024
    tmp_ptr_array_reduced[tmp_thread_block_index] +=
        ptr_array_inputs_Z0_received[stride_Z0_received *
                                     tmp_thread_global_index] *
            ptr_array_inputs_Z1_received[stride_Z1_received *
                                         tmp_thread_global_index] +
        ptr_array_inputs_Z0_received[stride_Z0_received *
                                     tmp_thread_global_index_offset] *
            ptr_array_inputs_Z1_received[stride_Z1_received *
                                         tmp_thread_global_index_offset];

    // BlockIdx.x[0]: [0...1...1023] += 4 * 1024 * 2
    // BlockIdx.x[1]: [2048...2049...3071] += 4 * 1024 * 2
    // BlockIdx.x[2]: [4096...4097...5019] += 4 * 1024 * 2
    // BlockIdx.x[3]: [6144...6145...7167] += 4 * 1024 * 2
    tmp_thread_global_index += tmp_grid_stride;
    tmp_thread_global_index_offset += tmp_grid_stride;
  } while (tmp_thread_global_index_offset < size_received);

  // Add by one load from GMEM (Remaining elements).
  if (tmp_thread_global_index < size_received) {
    // BlockIdx.x[0]: [0...1...1023] += [0...1...1023] + [0...1...1023] + 1024
    // BlockIdx.x[1]: [0...1...1023] += [2048...2049...3071] +
    // [2048...2049...3071] + 1024 BlockIdx.x[2]: [0...1...1023] +=
    // [4096...4097...5019] + [4096...4097...5019] + 1024 BlockIdx.x[3]:
    // [0...1...1023] += [6144...6145...7167] + [6144...6145...7167] + 1024
    tmp_ptr_array_reduced[tmp_thread_block_index] +=
        ptr_array_inputs_Z0_received[stride_Z0_received *
                                     tmp_thread_global_index] *
        ptr_array_inputs_Z1_received[stride_Z1_received *
                                     tmp_thread_global_index];
  }

  __syncthreads();

  if (BLOCK_SIZE >= 1024u) {
    if (tmp_thread_block_index < 512u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 512u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 512u) {
    if (tmp_thread_block_index < 256u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 256u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 256u) {
    if (tmp_thread_block_index < 128u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 128u];
    }

    __syncthreads();
  }

  if (BLOCK_SIZE >= 128u) {
    if (tmp_thread_block_index < 64u) {
      tmp_ptr_array_reduced[tmp_thread_block_index] +=
          tmp_ptr_array_reduced[tmp_thread_block_index + 64u];
    }

    __syncthreads();
  }

  // Last warp.
  if (tmp_thread_block_index < warpSize) {
    switch (BLOCK_SIZE) {
      case 32:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 16);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 8);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 16:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 8u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 8:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 4u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 4:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 2u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 2:
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 1u];
        __syncwarp();
        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
      case 1:
        break;
      default:  // BLOCK_SIZE >= 64
        tmp_thread_reduced_value =
            tmp_ptr_array_reduced[tmp_thread_block_index] +
            tmp_ptr_array_reduced[tmp_thread_block_index + 32u];
        __syncwarp();

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 16);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 8);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 4);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 2);

        tmp_thread_reduced_value +=
            __shfl_down_sync(FULL_MASK, tmp_thread_reduced_value, 1);

        tmp_ptr_array_reduced[tmp_thread_block_index] =
            tmp_thread_reduced_value;
        break;
    }
  }

  if (tmp_thread_block_index == 0u) {
    ptr_array_outputs_received[blockIdx.x] = tmp_ptr_array_reduced[0];
  }
}

template <typename T>
__device__ void Launch_Reduce_Z0Z1(
    size_t const size_received, size_t const stride_Z0_received,
    size_t const stride_Z1_received, T *const ptr_array_outputs_recieved,
    T const *const ptr_array_inputs_Z0_received,
    T const *const ptr_array_inputs_Z1_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved) {
  switch (ptr_dimension_block_recieved->x) {
    case 1024:
      kernel__Reduce_Z0Z1<T, 1024u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             1024u * sizeof(T)>>>(
              size_received, stride_Z0_received, stride_Z1_received,
              ptr_array_outputs_recieved, ptr_array_inputs_Z0_received,
              ptr_array_inputs_Z1_received);
      break;
    case 512:
      kernel__Reduce_Z0Z1<T, 512u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             512u * sizeof(T)>>>(size_received, stride_Z0_received,
                                 stride_Z1_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_Z0_received,
                                 ptr_array_inputs_Z1_received);
      break;
    case 256:
      kernel__Reduce_Z0Z1<T, 256u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             256u * sizeof(T)>>>(size_received, stride_Z0_received,
                                 stride_Z1_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_Z0_received,
                                 ptr_array_inputs_Z1_received);
      break;
    case 128:
      kernel__Reduce_Z0Z1<T, 128u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             128u * sizeof(T)>>>(size_received, stride_Z0_received,
                                 stride_Z1_received, ptr_array_outputs_recieved,
                                 ptr_array_inputs_Z0_received,
                                 ptr_array_inputs_Z1_received);
      break;
    case 64:
      kernel__Reduce_Z0Z1<T, 64u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             64u * sizeof(T)>>>(size_received, stride_Z0_received,
                                stride_Z1_received, ptr_array_outputs_recieved,
                                ptr_array_inputs_Z0_received,
                                ptr_array_inputs_Z1_received);
      break;
    case 32:
      kernel__Reduce_Z0Z1<T, 32u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (32u + 16u) * sizeof(T)>>>(
              size_received, stride_Z0_received, stride_Z1_received,
              ptr_array_outputs_recieved, ptr_array_inputs_Z0_received,
              ptr_array_inputs_Z1_received);
      break;
    case 16:
      kernel__Reduce_Z0Z1<T, 16u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (16u + 8u) * sizeof(T)>>>(
              size_received, stride_Z0_received, stride_Z1_received,
              ptr_array_outputs_recieved, ptr_array_inputs_Z0_received,
              ptr_array_inputs_Z1_received);
      break;
    case 8:
      kernel__Reduce_Z0Z1<T, 8u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (8u + 4u) * sizeof(T)>>>(
              size_received, stride_Z0_received, stride_Z1_received,
              ptr_array_outputs_recieved, ptr_array_inputs_Z0_received,
              ptr_array_inputs_Z1_received);
      break;
    case 4:
      kernel__Reduce_Z0Z1<T, 4u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (4u + 2u) * sizeof(T)>>>(
              size_received, stride_Z0_received, stride_Z1_received,
              ptr_array_outputs_recieved, ptr_array_inputs_Z0_received,
              ptr_array_inputs_Z1_received);
      break;
    case 2:
      kernel__Reduce_Z0Z1<T, 2u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved,
             (2u + 1u) * sizeof(T)>>>(
              size_received, stride_Z0_received, stride_Z1_received,
              ptr_array_outputs_recieved, ptr_array_inputs_Z0_received,
              ptr_array_inputs_Z1_received);
      break;
    case 1:
      kernel__Reduce_Z0Z1<T, 1u><<<*ptr_dimension_grid_recieved,
                                   *ptr_dimension_block_recieved, sizeof(T)>>>(
          size_received, stride_Z0_received, stride_Z1_received,
          ptr_array_outputs_recieved, ptr_array_inputs_Z0_received,
          ptr_array_inputs_Z1_received);
      break;
  }
}

template <typename T>
__device__ void Reduce_Z0Z1(
    size_t const size_received, size_t const stride_dim3_received,
    size_t const stride_Z0_received, size_t const stride_Z1_received,
    T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_Z0_received,
    T const *const ptr_array_inputs_Z1_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved) {
  if (size_received > 1u) {
    if (USE_PARALLEL && size_received >= warpSize * 2u) {
      size_t tmp_iteration(0u), tmp_number_elements_to_reduce;

      Launch_Reduce_Z0Z1<T>(
          size_received, stride_Z0_received, stride_Z1_received,
          ptr_array_outputs_received, ptr_array_inputs_Z0_received,
          ptr_array_inputs_Z1_received, ptr_dimension_grid_recieved,
          ptr_dimension_block_recieved);

      tmp_number_elements_to_reduce = ptr_dimension_grid_recieved->x;

      while (tmp_number_elements_to_reduce != 1u) {
        if (tmp_number_elements_to_reduce >= warpSize * 2u) {
          ++tmp_iteration;

          Launch_Reduce<T>(tmp_number_elements_to_reduce,
                           ptr_array_outputs_received,
                           ptr_array_outputs_received,
                           ptr_dimension_grid_recieved +
                               tmp_iteration * stride_dim3_received,
                           ptr_dimension_block_recieved +
                               tmp_iteration * stride_dim3_received);

          tmp_number_elements_to_reduce =
              ptr_dimension_grid_recieved[tmp_iteration * stride_dim3_received]
                  .x;
        } else {
          Reduce_Loop(tmp_number_elements_to_reduce, ptr_array_outputs_received,
                      ptr_array_outputs_received);

          tmp_number_elements_to_reduce = 1u;
        }
      }
    } else {
      T tmp_summation(0);

      for (size_t i(0_UZ); i != size_received; ++i) {
        tmp_summation += ptr_array_inputs_Z0_received[i * stride_Z0_received] *
                         ptr_array_inputs_Z1_received[i * stride_Z1_received];
      }

      *ptr_array_outputs_received = tmp_summation;
    }
  } else if (size_received == 1u) {
    *ptr_array_outputs_received =
        *ptr_array_inputs_Z0_received * *ptr_array_inputs_Z1_received;
  } else {
    ERR(L"No element to reduce!",);
  }
}

template <typename T>
__device__ inline void Reduce_Array_Loop(
    size_t const size_received, size_t const size_array_received,
    size_t const stride_array_received, T *const ptr_array_IO_received,
    struct dim3 const *const ptr_dimension_grid_reduce_array_recieved,
    struct dim3 const *const ptr_dimension_block_reduce_array_recieved) {
  for (size_t i(1u); i != size_received; ++i) {
    Accumulate::Accumulate_X_X_1D(
        size_array_received, ptr_array_IO_received,
        ptr_array_IO_received + i * stride_array_received,
        ptr_dimension_grid_reduce_array_recieved,
        ptr_dimension_block_reduce_array_recieved);
  }
}

template <typename T, size_t BLOCK_SIZE>
__global__ void kernel__Reduce_Array(
    size_t size_received, size_t size_array_received,
    size_t stride_array_received, T *const ptr_array_IO_received,
    struct dim3 const *const ptr_dimension_grid_reduce_array_recieved,
    struct dim3 const *const ptr_dimension_block_reduce_array_recieved) {
  size_t const &tmp_thread_block_index(threadIdx.x),
      tmp_grid_stride(gridDim.x * BLOCK_SIZE);
  size_t tmp_thread_global_index(blockIdx.x * BLOCK_SIZE + threadIdx.x),
      tmp_thread_global_index_offset(tmp_thread_global_index + tmp_grid_stride);

  T *const tmp_ptr_array_IO(ptr_array_IO_received +
                            tmp_thread_global_index * stride_array_received);

  while (tmp_thread_global_index_offset < size_received) {
    Accumulate::Accumulate_X_X_1D(
        size_array_received, tmp_ptr_array_IO,
        ptr_array_IO_received +
            tmp_thread_global_index_offset * stride_array_received,
        ptr_dimension_grid_reduce_array_recieved,
        ptr_dimension_block_reduce_array_recieved);

    tmp_thread_global_index_offset += tmp_grid_stride;
  }

  if (BLOCK_SIZE >= 2u) {
    __syncthreads();

    if (BLOCK_SIZE >= 1024u) {
      if (tmp_thread_block_index < 512u) {
        Accumulate::Accumulate_X_X_1D(
            size_array_received, tmp_ptr_array_IO,
            tmp_ptr_array_IO + 512u * stride_array_received,
            ptr_dimension_grid_reduce_array_recieved,
            ptr_dimension_block_reduce_array_recieved);
      }

      __syncthreads();
    }

    if (BLOCK_SIZE >= 512u) {
      if (tmp_thread_block_index < 256u) {
        Accumulate::Accumulate_X_X_1D(
            size_array_received, tmp_ptr_array_IO,
            tmp_ptr_array_IO + 256u * stride_array_received,
            ptr_dimension_grid_reduce_array_recieved,
            ptr_dimension_block_reduce_array_recieved);
      }

      __syncthreads();
    }

    if (BLOCK_SIZE >= 256u) {
      if (tmp_thread_block_index < 128u) {
        Accumulate::Accumulate_X_X_1D(
            size_array_received, tmp_ptr_array_IO,
            tmp_ptr_array_IO + 128u * stride_array_received,
            ptr_dimension_grid_reduce_array_recieved,
            ptr_dimension_block_reduce_array_recieved);
      }

      __syncthreads();
    }

    if (BLOCK_SIZE >= 128u) {
      if (tmp_thread_block_index < 64u) {
        Accumulate::Accumulate_X_X_1D(
            size_array_received, tmp_ptr_array_IO,
            tmp_ptr_array_IO + 64u * stride_array_received,
            ptr_dimension_grid_reduce_array_recieved,
            ptr_dimension_block_reduce_array_recieved);
      }

      __syncthreads();
    }

    // Last warp.
    if (tmp_thread_block_index < 32u) {
      if (BLOCK_SIZE >= 64u) {
        Accumulate::Accumulate_X_X_1D(
            size_array_received, tmp_ptr_array_IO,
            tmp_ptr_array_IO + 32u * stride_array_received,
            ptr_dimension_grid_reduce_array_recieved,
            ptr_dimension_block_reduce_array_recieved);

        __syncwarp();
      }

      if (BLOCK_SIZE >= 32u) {
        if (tmp_thread_block_index < 16u) {
          Accumulate::Accumulate_X_X_1D(
              size_array_received, tmp_ptr_array_IO,
              tmp_ptr_array_IO + 16u * stride_array_received,
              ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
        }

        __syncwarp();
      }

      if (BLOCK_SIZE >= 16u) {
        if (tmp_thread_block_index < 8u) {
          Accumulate::Accumulate_X_X_1D(
              size_array_received, tmp_ptr_array_IO,
              tmp_ptr_array_IO + 8u * stride_array_received,
              ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
        }

        __syncwarp();
      }

      if (BLOCK_SIZE >= 8u) {
        if (tmp_thread_block_index < 4u) {
          Accumulate::Accumulate_X_X_1D(
              size_array_received, tmp_ptr_array_IO,
              tmp_ptr_array_IO + 4u * stride_array_received,
              ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
        }

        __syncwarp();
      }

      if (BLOCK_SIZE >= 4u) {
        if (tmp_thread_block_index < 2u) {
          Accumulate::Accumulate_X_X_1D(
              size_array_received, tmp_ptr_array_IO,
              tmp_ptr_array_IO + 2u * stride_array_received,
              ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
        }

        __syncwarp();
      }

      if (BLOCK_SIZE >= 2u) {
        if (tmp_thread_block_index == 0u) {
          Accumulate::Accumulate_X_X_1D(
              size_array_received, tmp_ptr_array_IO,
              tmp_ptr_array_IO + stride_array_received,
              ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
        }
      }
    }
  }
}

template <typename T>
__device__ void Launch_Reduce_Array(
    size_t const size_received, size_t const size_array_received,
    size_t const stride_array_received, T *const ptr_array_IO_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved,
    struct dim3 const *const ptr_dimension_grid_reduce_array_recieved,
    struct dim3 const *const ptr_dimension_block_reduce_array_recieved) {
  switch (ptr_dimension_block_recieved->x) {
    case 1024:
      kernel__Reduce_Array<T, 1024u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved>>>(
              size_received, size_array_received, stride_array_received,
              ptr_array_IO_received, ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
      break;
    case 512:
      kernel__Reduce_Array<T, 512u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved>>>(
              size_received, size_array_received, stride_array_received,
              ptr_array_IO_received, ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
      break;
    case 256:
      kernel__Reduce_Array<T, 256u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved>>>(
              size_received, size_array_received, stride_array_received,
              ptr_array_IO_received, ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
      break;
    case 128:
      kernel__Reduce_Array<T, 128u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved>>>(
              size_received, size_array_received, stride_array_received,
              ptr_array_IO_received, ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
      break;
    case 64:
      kernel__Reduce_Array<T, 64u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved>>>(
              size_received, size_array_received, stride_array_received,
              ptr_array_IO_received, ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
      break;
    case 32:
      kernel__Reduce_Array<T, 32u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved>>>(
              size_received, size_array_received, stride_array_received,
              ptr_array_IO_received, ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
      break;
    case 16:
      kernel__Reduce_Array<T, 16u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved>>>(
              size_received, size_array_received, stride_array_received,
              ptr_array_IO_received, ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
      break;
    case 8:
      kernel__Reduce_Array<T, 8u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved>>>(
              size_received, size_array_received, stride_array_received,
              ptr_array_IO_received, ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
      break;
    case 4:
      kernel__Reduce_Array<T, 4u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved>>>(
              size_received, size_array_received, stride_array_received,
              ptr_array_IO_received, ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
      break;
    case 2:
      kernel__Reduce_Array<T, 2u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved>>>(
              size_received, size_array_received, stride_array_received,
              ptr_array_IO_received, ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
      break;
    case 1:
      kernel__Reduce_Array<T, 1u>
          <<<*ptr_dimension_grid_recieved, *ptr_dimension_block_recieved>>>(
              size_received, size_array_received, stride_array_received,
              ptr_array_IO_received, ptr_dimension_grid_reduce_array_recieved,
              ptr_dimension_block_reduce_array_recieved);
      break;
  }
}

template <typename T>
__device__ void Reduce_Array(
    size_t const size_received, size_t const stride_array_received,
    size_t const stride_dim3_received, T *const ptr_array_IO_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved,
    struct dim3 const *const ptr_dimension_grid_reduce_array_recieved,
    struct dim3 const *const ptr_dimension_block_reduce_array_recieved) {
  if (size_received > 1u) {
    if (USE_PARALLEL && size_received > 3u) {
      size_t tmp_iteration(0u), tmp_number_elements_to_reduce,
          tmp_stride_array_access;

      Launch_Reduce_Array<T>(size_received, stride_array_received,
                             stride_array_received, ptr_array_IO_received,
                             ptr_dimension_grid_recieved,
                             ptr_dimension_block_recieved,
                             ptr_dimension_grid_reduce_array_recieved,
                             ptr_dimension_block_reduce_array_recieved);

      tmp_number_elements_to_reduce = ptr_dimension_grid_recieved->x;
      tmp_stride_array_access =
          stride_array_received * ptr_dimension_block_recieved->x;

      while (tmp_number_elements_to_reduce != 1u) {
        if (tmp_number_elements_to_reduce > 3u) {
          ++tmp_iteration;

          Launch_Reduce_Array<T>(tmp_number_elements_to_reduce,
                                 stride_array_received, tmp_stride_array_access,
                                 ptr_array_IO_received,
                                 ptr_dimension_grid_recieved +
                                     tmp_iteration * stride_dim3_received,
                                 ptr_dimension_block_recieved +
                                     tmp_iteration * stride_dim3_received,
                                 ptr_dimension_grid_reduce_array_recieved,
                                 ptr_dimension_block_reduce_array_recieved);

          tmp_number_elements_to_reduce =
              ptr_dimension_grid_recieved[tmp_iteration * stride_dim3_received]
                  .x;
          tmp_stride_array_access =
              tmp_stride_array_access *
              ptr_dimension_block_recieved[tmp_iteration * stride_dim3_received]
                  .x;
        } else {
          if (stride_array_received < warpSize) {
            CUDA__Check_Error();
          }

          Reduce_Array_Loop<T>(tmp_number_elements_to_reduce,
                               stride_array_received, tmp_stride_array_access,
                               ptr_array_IO_received,
                               ptr_dimension_grid_reduce_array_recieved,
                               ptr_dimension_block_reduce_array_recieved);

          tmp_number_elements_to_reduce = 1u;
        }
      }
    } else {
      Reduce_Array_Loop<T>(size_received, stride_array_received,
                           stride_array_received, ptr_array_IO_received,
                           ptr_dimension_grid_reduce_array_recieved,
                           ptr_dimension_block_reduce_array_recieved);
    }
  } else if (size_received == 0u) {
    ERR(L"No array to reduce!",);
  }
}
}  // namespace Reduce