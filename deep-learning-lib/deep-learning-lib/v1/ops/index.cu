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
__global__ void kernel__Memory_Initialize_Index(T *const ptr_array_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_received[tmp_thread_global_index] = T(tmp_thread_global_index);
}

template <typename T>
__global__ void kernel__Memory_Initialize_Index(size_t const size_received,
                                                T *const ptr_array_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    ptr_array_received[tmp_thread_global_index] = T(tmp_thread_global_index);
  }
}

template <typename T>
__global__ void kernel_while__Memory_Initialize_Index(
    size_t const size_received, T *const ptr_array_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_received[tmp_thread_global_index] = T(tmp_thread_global_index);

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__device__ void Memory_Initialize_Index(
    size_t const size_received, T *const ptr_array_outputs_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(Memory_Initialize_Index<T>,
                             ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             ptr_array_outputs_received)
  } else {
    for (size_t i(0_UZ); i != size_received; ++i) {
      ptr_array_outputs_received[i] = i;
    }
  }
}

template <typename T>
__global__ void kernel__Memory_Initialize_Index_Shift(
    size_t const size_received, size_t const shift,
    T *const ptr_array_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_received[tmp_thread_global_index] =
      T((tmp_thread_global_index + shift) % size_received);
}

template <typename T>
__global__ void kernel_if__Memory_Initialize_Index_Shift(
    size_t const size_received, size_t const shift,
    T *const ptr_array_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    ptr_array_received[tmp_thread_global_index] =
        T((tmp_thread_global_index + shift) % size_received);
  }
}

template <typename T>
__global__ void kernel_while__Memory_Initialize_Index_Shift(
    size_t const size_received, size_t const shift,
    T *const ptr_array_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_received[tmp_thread_global_index] =
        T((tmp_thread_global_index + shift) % size_received);

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__device__ void Memory_Initialize_Index_Shift(
    size_t const size_received, size_t const shift,
    T *const ptr_array_outputs_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    if (ptr_dimension_grid_received->x * ptr_dimension_block_received->x <
        size_received) {
      kernel_while__Memory_Initialize_Index_Shift<<<
          *ptr_dimension_grid_received, *ptr_dimension_block_received>>>(
          size_received, shift, ptr_array_outputs_received);
    } else if (ptr_dimension_grid_received->x *
                   ptr_dimension_block_received->x >
               size_received) {
      kernel_if__Memory_Initialize_Index_Shift<<<
          *ptr_dimension_grid_received, *ptr_dimension_block_received>>>(
          size_received, shift, ptr_array_outputs_received);
    } else {
      kernel__Memory_Initialize_Index_Shift<<<*ptr_dimension_grid_received,
                                              *ptr_dimension_block_received>>>(
          size_received, shift, ptr_array_outputs_received);
    }
  } else {
    for (size_t i(0_UZ); i != size_received; ++i) {
      ptr_array_outputs_received[i] = (i + shift) % size_received;
    }
  }
}

template <typename T>
__global__ void kernel__Memory_Initialize_Index_Offset(
    size_t const offSet__received, T *const ptr_array_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x +
                                       offSet__received);

  ptr_array_received[tmp_thread_global_index] = T(tmp_thread_global_index);
}

template <typename T>
__global__ void kernel__Memory_Initialize_Index_Offset(
    size_t const size_received, size_t const offSet__received,
    T *const ptr_array_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x +
                                       offSet__received);

  if (tmp_thread_global_index < size_received) {
    ptr_array_received[tmp_thread_global_index] = T(tmp_thread_global_index);
  }
}

template <typename T>
__global__ void kernel_while__Memory_Initialize_Index_Offset(
    size_t const size_received, size_t const offSet__received,
    T *const ptr_array_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x +
                                 offSet__received);

  do {
    ptr_array_received[tmp_thread_global_index] = T(tmp_thread_global_index);

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__device__ void Memory_Initialize_Index_Offset(
    size_t const size_received, size_t const offSet__received,
    T *const ptr_array_outputs_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received - offSet__received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(Memory_Initialize_Index_Offset<T>,
                             ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             offSet__received, ptr_array_outputs_received)
  } else {
    for (size_t i(offSet__received); i != size_received; ++i) {
      ptr_array_outputs_received[i] = i;
    }
  }
}

template <typename T>
__global__ void kernel__Memory_Initialize_Index_Transposed(
    size_t const size_received, size_t const width_received,
    T *const ptr_array_outputs_received) {
  size_t const &tmp_thread_block_index_x(threadIdx.x),
      &tmp_thread_block_index_y(threadIdx.y),
      tmp_grid_stride_x(gridDim.x * blockDim.x * 2u),
      tmp_grid_stride_y(gridDim.y * blockDim.y * 2u);
  size_t tmp_thread_global_index_block_x(blockIdx.x * blockDim.x * 2u),
      tmp_thread_global_index_block_y(blockIdx.y * blockDim.y * 2u),
      tmp_thread_global_index_x, tmp_thread_global_index_y,
      tmp_thread_global_index_offSet__x, tmp_thread_global_index_offSet__y;

  while (tmp_thread_global_index_block_y < width_received) {
    while (tmp_thread_global_index_block_x < width_received) {
      tmp_thread_global_index_x =
          tmp_thread_global_index_block_x + tmp_thread_block_index_x;
      tmp_thread_global_index_offSet__x =
          tmp_thread_global_index_x + blockDim.x;

      tmp_thread_global_index_y =
          tmp_thread_global_index_block_y + tmp_thread_block_index_y;
      tmp_thread_global_index_offSet__y =
          tmp_thread_global_index_y + blockDim.y;

      if (tmp_thread_global_index_offSet__x < width_received &&
          tmp_thread_global_index_offSet__y < width_received) {
        ptr_array_outputs_received[tmp_thread_global_index_y * width_received +
                                   tmp_thread_global_index_x] =
            tmp_thread_global_index_x * width_received +
            tmp_thread_global_index_y;
        ptr_array_outputs_received[tmp_thread_global_index_y * width_received +
                                   tmp_thread_global_index_offSet__x] =
            tmp_thread_global_index_offSet__x * width_received +
            tmp_thread_global_index_y;
        ptr_array_outputs_received[tmp_thread_global_index_offSet__y *
                                       width_received +
                                   tmp_thread_global_index_x] =
            tmp_thread_global_index_x * width_received +
            tmp_thread_global_index_offSet__y;
        ptr_array_outputs_received[tmp_thread_global_index_offSet__y *
                                       width_received +
                                   tmp_thread_global_index_offSet__x] =
            tmp_thread_global_index_offSet__x * width_received +
            tmp_thread_global_index_offSet__y;
      } else if (tmp_thread_global_index_offSet__x < width_received &&
                 tmp_thread_global_index_y < width_received) {
        ptr_array_outputs_received[tmp_thread_global_index_y * width_received +
                                   tmp_thread_global_index_x] =
            tmp_thread_global_index_x * width_received +
            tmp_thread_global_index_y;
        ptr_array_outputs_received[tmp_thread_global_index_y * width_received +
                                   tmp_thread_global_index_offSet__x] =
            tmp_thread_global_index_offSet__x * width_received +
            tmp_thread_global_index_y;
      } else if (tmp_thread_global_index_x < width_received &&
                 tmp_thread_global_index_offSet__y < width_received) {
        ptr_array_outputs_received[tmp_thread_global_index_y * width_received +
                                   tmp_thread_global_index_x] =
            tmp_thread_global_index_x * width_received +
            tmp_thread_global_index_y;
        ptr_array_outputs_received[tmp_thread_global_index_offSet__y *
                                       width_received +
                                   tmp_thread_global_index_x] =
            tmp_thread_global_index_x * width_received +
            tmp_thread_global_index_offSet__y;
      } else if (tmp_thread_global_index_x < width_received &&
                 tmp_thread_global_index_y < width_received) {
        ptr_array_outputs_received[tmp_thread_global_index_y * width_received +
                                   tmp_thread_global_index_x] =
            tmp_thread_global_index_x * width_received +
            tmp_thread_global_index_y;
      }

      // Increment X.
      tmp_thread_global_index_block_x += tmp_grid_stride_x;
    }

    // reset X.
    tmp_thread_global_index_block_x = blockIdx.x * blockDim.x * 2u;

    // Increment Y.
    tmp_thread_global_index_block_y += tmp_grid_stride_y;
  }
}

template <typename T>
__device__ void Memory_Initialize_Index_Transposed(
    size_t const size_received, T *const ptr_array_outputs_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  size_t const tmp_width(
      static_cast<size_t>(floor(sqrt(static_cast<double>(size_received)))));

  if (USE_PARALLEL && tmp_width * tmp_width >= warpSize) {
    kernel__Memory_Initialize_Index_Transposed<<<
        *ptr_dimension_grid_received, *ptr_dimension_block_received>>>(
        tmp_width * tmp_width, tmp_width, ptr_array_outputs_received);
  } else {
    for (size_t row(0u), column(0u); column != tmp_width; ++column) {
      for (row = 0u; row != tmp_width; ++row) {
        ptr_array_outputs_received[column * tmp_width + row] =
            row * tmp_width + column;
      }
    }
  }

  DL::Memory::Memory_Initialize_Index_Offset<T>(
      size_received, tmp_width * tmp_width, ptr_array_outputs_received,
      ptr_dimension_grid_received + 1, ptr_dimension_block_received + 1);
}
}  // namespace DL::Memory
