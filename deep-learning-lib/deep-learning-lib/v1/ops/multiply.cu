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
__global__ void kernel__Multiply_Z_Y_1D(size_t const stride_Z_received,
                                        T const constant_received,
                                        T *const ptr_array_Z_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_Z_received[stride_Z_received * tmp_thread_global_index] *=
      constant_received;
}

template <typename T>
__global__ void kernel__Multiply_Z_Y_1D(size_t const size_received,
                                        size_t const stride_Z_received,
                                        T const constant_received,
                                        T *const ptr_array_Z_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    ptr_array_Z_received[stride_Z_received * tmp_thread_global_index] *=
        constant_received;
  }
}

template <typename T>
__global__ void kernel_while__Multiply_Z_Y_1D(size_t const size_received,
                                              size_t const stride_Z_received,
                                              T const constant_received,
                                              T *const ptr_array_Z_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_Z_received[stride_Z_received * tmp_thread_global_index] *=
        constant_received;

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__device__ void Multiply_Z_Y_1D(
    size_t const size_received, size_t const stride_Z_received,
    T const constant_received, T *ptr_array_Z_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(Multiply_Z_Y_1D<T>, ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             stride_Z_received, constant_received,
                             ptr_array_Z_received)
  } else {
    for (T const *const ptr_Z_end(ptr_array_Z_received +
                                  stride_Z_received * size_received);
         ptr_array_Z_received != ptr_Z_end;
         ptr_array_Z_received += stride_Z_received) {
      *ptr_array_Z_received *= constant_received;
    }
  }
}

template <typename T>
__global__ void kernel__Multiply_X_Y_1D(T const constant_received,
                                        T *const ptr_array_X_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_X_received[tmp_thread_global_index] *= constant_received;
}

template <typename T>
__global__ void kernel__Multiply_X_Y_1D(size_t const size_received,
                                        T const constant_received,
                                        T *const ptr_array_X_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    ptr_array_X_received[tmp_thread_global_index] *= constant_received;
  }
}

template <typename T>
__global__ void kernel_while__Multiply_X_Y_1D(size_t const size_received,
                                              T const constant_received,
                                              T *const ptr_array_X_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_X_received[tmp_thread_global_index] *= constant_received;

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__device__ void Multiply_X_Y_1D(
    size_t const size_received, T const constant_received,
    T *ptr_array_X_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(Multiply_X_Y_1D<T>, ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             constant_received, ptr_array_X_received)
  } else {
    for (T const *const ptr_X_end(ptr_array_X_received + size_received);
         ptr_array_X_received != ptr_X_end; ++ptr_array_X_received) {
      *ptr_array_X_received *= constant_received;
    }
  }
}

template <typename T>
__device__ void Multiply_X_Y_1D(
    bool &ref_synchronized_received, size_t const size_received,
    T const constant_received, T *ptr_array_X_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    ref_synchronized_received = false;

    LAUNCH_KERNEL_POINTER_1D(Multiply_X_Y_1D<T>, ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             constant_received, ptr_array_X_received)
  } else {
    for (T const *const ptr_X_end(ptr_array_X_received + size_received);
         ptr_array_X_received != ptr_X_end; ++ptr_array_X_received) {
      *ptr_array_X_received *= constant_received;
    }
  }
}

template <typename T>
__global__ void kernel__FMAC_Z_YX_1D(size_t const stride_Z_received,
                                     T *const ptr_array_Z_received,
                                     T const constant_received,
                                     T const *const ptr_array_X_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_Z_received[tmp_thread_global_index * stride_Z_received] +=
      constant_received * ptr_array_X_received[tmp_thread_global_index];
}

template <typename T>
__global__ void kernel__FMAC_Z_YX_1D(size_t const size_received,
                                     size_t const stride_Z_received,
                                     T *const ptr_array_Z_received,
                                     T const constant_received,
                                     T const *const ptr_array_X_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    ptr_array_Z_received[tmp_thread_global_index * stride_Z_received] +=
        constant_received * ptr_array_X_received[tmp_thread_global_index];
  }
}

template <typename T>
__global__ void kernel_while__FMAC_Z_YX_1D(
    size_t const size_received, size_t const stride_Z_received,
    T *const ptr_array_Z_received, T const constant_received,
    T const *const ptr_array_X_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_Z_received[tmp_thread_global_index * stride_Z_received] +=
        constant_received * ptr_array_X_received[tmp_thread_global_index];

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__device__ void FMAC_Z_YX_1D(
    size_t const size_received, size_t const stride_Z_received,
    T *ptr_array_Z_received, T const constant_received,
    T const *ptr_array_X_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(FMAC_Z_YX_1D<T>, ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             stride_Z_received, ptr_array_Z_received,
                             constant_received, ptr_array_X_received)
  } else {
    for (T const *const tmp_ptr_Z_end(ptr_array_Z_received +
                                      stride_Z_received * size_received);
         ptr_array_Z_received != tmp_ptr_Z_end;
         ptr_array_Z_received += stride_Z_received, ++ptr_array_X_received) {
      *ptr_array_Z_received += constant_received * *ptr_array_X_received;
    }
  }
}

template <typename T>
__global__ void kernel__FMAC_X_YZ_1D(size_t const stride_Z_received,
                                     T *const ptr_array_X_received,
                                     T const constant_received,
                                     T const *const ptr_array_Z_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_X_received[tmp_thread_global_index] +=
      constant_received *
      ptr_array_Z_received[tmp_thread_global_index * stride_Z_received];
}

template <typename T>
__global__ void kernel__FMAC_X_YZ_1D(size_t const size_received,
                                     size_t const stride_Z_received,
                                     T *const ptr_array_X_received,
                                     T const constant_received,
                                     T const *const ptr_array_Z_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    ptr_array_X_received[tmp_thread_global_index] +=
        constant_received *
        ptr_array_Z_received[tmp_thread_global_index * stride_Z_received];
  }
}

template <typename T>
__global__ void kernel_while__FMAC_X_YZ_1D(
    size_t const size_received, size_t const stride_Z_received,
    T *const ptr_array_X_received, T const constant_received,
    T const *const ptr_array_Z_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_X_received[tmp_thread_global_index] +=
        constant_received *
        ptr_array_Z_received[tmp_thread_global_index * stride_Z_received];

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__device__ void FMAC_X_YZ_1D(
    size_t const size_received, size_t const stride_Z_received,
    T *ptr_array_X_received, T const constant_received,
    T const *ptr_array_Z_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(FMAC_X_YZ_1D<T>, ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             stride_Z_received, ptr_array_X_received,
                             constant_received, ptr_array_Z_received)
  } else {
    for (T const *const tmp_ptr_X_end(ptr_array_X_received + size_received);
         ptr_array_X_received != tmp_ptr_X_end;
         ++ptr_array_X_received, ptr_array_Z_received += stride_Z_received) {
      *ptr_array_X_received += constant_received * *ptr_array_Z_received;
    }
  }
}

template <typename T>
__global__ void kernel__FMAC_X_YX_1D(
    T *const ptr_array_outputs_X_received, T const constant_received,
    T const *const ptr_array_inputs_X_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_outputs_X_received[tmp_thread_global_index] +=
      constant_received * ptr_array_inputs_X_received[tmp_thread_global_index];
}

template <typename T>
__global__ void kernel__FMAC_X_YX_1D(
    size_t const size_received, T *const ptr_array_outputs_X_received,
    T const constant_received, T const *const ptr_array_inputs_X_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    ptr_array_outputs_X_received[tmp_thread_global_index] +=
        constant_received *
        ptr_array_inputs_X_received[tmp_thread_global_index];
  }
}

template <typename T>
__global__ void kernel_while__FMAC_X_YX_1D(
    size_t const size_received, T *const ptr_array_outputs_X_received,
    T const constant_received, T const *const ptr_array_inputs_X_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_outputs_X_received[tmp_thread_global_index] +=
        constant_received *
        ptr_array_inputs_X_received[tmp_thread_global_index];

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__device__ void FMAC_X_YX_1D(
    size_t const size_received, T *ptr_array_outputs_X_received,
    T const constant_received, T const *ptr_array_inputs_X_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(FMAC_X_YX_1D<T>, ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             ptr_array_outputs_X_received, constant_received,
                             ptr_array_inputs_X_received)
  } else {
    for (T const *const tmp_ptr_X_end(ptr_array_outputs_X_received +
                                      size_received);
         ptr_array_outputs_X_received != tmp_ptr_X_end;
         ++ptr_array_outputs_X_received, ++ptr_array_inputs_X_received) {
      *ptr_array_outputs_X_received +=
          constant_received * *ptr_array_inputs_X_received;
    }
  }
}

template <typename T>
__global__ void kernel__FMAC_X_YX_1D__atomic(
    T *const ptr_array_outputs_X_received, T const constant_received,
    T const *const ptr_array_inputs_X_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  atomicAdd(
      ptr_array_outputs_X_received + tmp_thread_global_index,
      constant_received * ptr_array_inputs_X_received[tmp_thread_global_index]);
}

template <typename T>
__global__ void kernel__FMAC_X_YX_1D__atomic(
    size_t const size_received, T *const ptr_array_outputs_X_received,
    T const constant_received, T const *const ptr_array_inputs_X_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    atomicAdd(ptr_array_outputs_X_received + tmp_thread_global_index,
              constant_received *
                  ptr_array_inputs_X_received[tmp_thread_global_index]);
  }
}

template <typename T>
__global__ void kernel_while__FMAC_X_YX_1D__atomic(
    size_t const size_received, T *const ptr_array_outputs_X_received,
    T const constant_received, T const *const ptr_array_inputs_X_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    atomicAdd(ptr_array_outputs_X_received + tmp_thread_global_index,
              constant_received *
                  ptr_array_inputs_X_received[tmp_thread_global_index]);

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__device__ void FMAC_X_YX_1D__atomic(
    size_t const size_received, T *ptr_array_outputs_X_received,
    T const constant_received, T const *ptr_array_inputs_X_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(FMAC_X_YX_1D__atomic<T>,
                             ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             ptr_array_outputs_X_received, constant_received,
                             ptr_array_inputs_X_received)
  } else {
    for (T const *const tmp_ptr_X_end(ptr_array_outputs_X_received +
                                      size_received);
         ptr_array_outputs_X_received != tmp_ptr_X_end;
         ++ptr_array_outputs_X_received, ++ptr_array_inputs_X_received) {
      atomicAdd(ptr_array_outputs_X_received,
                constant_received * *ptr_array_inputs_X_received);
    }
  }
}
}  // namespace Multiply