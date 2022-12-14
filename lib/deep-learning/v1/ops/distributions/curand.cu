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

#include "deep-learning/v1/ops/distributions/curand.cuh"

#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#include <chrono>

__global__ void kernel__cuRAND__Memcpy_cuRAND_State_MTGP32(
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_destination_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_source_received,
    struct mtgp32_kernel_params
        *const ptr_array_mtgp32_kernel_params_t_source_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index]
      .k = ptr_array_mtgp32_kernel_params_t_source_received +
           tmp_thread_global_index;

  *ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index]
       .k =
      *ptr_array_cuRAND_State_MTGP32_source_received[tmp_thread_global_index].k;
}

__global__ void kernel__cuRAND__Memcpy_cuRAND_State_MTGP32(
    size_t const size_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_destination_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_source_received,
    struct mtgp32_kernel_params
        *const ptr_array_mtgp32_kernel_params_t_source_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index]
        .k = ptr_array_mtgp32_kernel_params_t_source_received +
             tmp_thread_global_index;

    *ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index]
         .k =
        *ptr_array_cuRAND_State_MTGP32_source_received[tmp_thread_global_index]
             .k;
  }
}

__global__ void kernel_while__cuRAND__Memcpy_cuRAND_State_MTGP32(
    size_t const size_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_destination_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_source_received,
    struct mtgp32_kernel_params
        *const ptr_array_mtgp32_kernel_params_t_source_received) {
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index]
        .k = ptr_array_mtgp32_kernel_params_t_source_received +
             tmp_thread_global_index;

    *ptr_array_cuRAND_State_MTGP32_destination_received[tmp_thread_global_index]
         .k =
        *ptr_array_cuRAND_State_MTGP32_source_received[tmp_thread_global_index]
             .k;

    tmp_thread_global_index += gridDim.x * blockDim.x;
  } while (tmp_thread_global_index < size_received);
}

__device__ void cuRAND__Memcpy_cuRAND_State_MTGP32(
    size_t const size_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_destination_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_source_received,
    struct mtgp32_kernel_params
        *const ptr_array_mtgp32_kernel_params_t_source_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(cuRAND__Memcpy_cuRAND_State_MTGP32,
                             ptr_dimension_grid_received,
                             ptr_dimension_block_received, 0_UZ, size_received,
                             ptr_array_cuRAND_State_MTGP32_destination_received,
                             ptr_array_cuRAND_State_MTGP32_source_received,
                             ptr_array_mtgp32_kernel_params_t_source_received)
  } else {
    for (size_t i(0_UZ); i != size_received; ++i) {
      ptr_array_cuRAND_State_MTGP32_destination_received[i].k =
          ptr_array_mtgp32_kernel_params_t_source_received + i;

      *ptr_array_cuRAND_State_MTGP32_destination_received[i].k =
          *ptr_array_cuRAND_State_MTGP32_source_received[i].k;
    }
  }
}

__host__ bool Allocate_cuRAND_MTGP32(
    int const number_states_MTGP32_received, size_t seed,
    struct mtgp32_kernel_params *&ptr_mtgp32_kernel_params_received,
    struct curandStateMtgp32 *&ptr_curandStateMtgp32_t_received) {
  if (number_states_MTGP32_received == 0) {
    ERR(L"Number of states for MTGP32 equal zero.",);

    return false;
  }

  CUDA__Safe_Call(cudaMalloc((void **)&ptr_mtgp32_kernel_params_received,
                             sizeof(struct mtgp32_kernel_params)));

  curandStatus_t tmp_curandStatus_t(curandMakeMTGP32Constants(
      mtgp32dc_params_fast_11213, ptr_mtgp32_kernel_params_received));

  if (tmp_curandStatus_t != curandStatus::CURAND_STATUS_SUCCESS) {
    ERR(
        L"curandMakeMTGP32Constants failed at %ls:%i: _%d", __FILE__, __LINE__, tmp_curandStatus_t);

    CUDA__Safe_Call(cudaFree(ptr_mtgp32_kernel_params_received));

    return false;
  }

  CUDA__Safe_Call(
      cudaMalloc((void **)&ptr_curandStateMtgp32_t_received,
                 static_cast<size_t>(number_states_MTGP32_received) *
                     sizeof(struct curandStateMtgp32)));

  for (int tmp_number_states_MTGP32_allocate,
       tmp_number_states_MTGP32_offset(0),
       tmp_length_i(static_cast<int>(
           ceil(static_cast<double>(number_states_MTGP32_received) / 200.0))),
       i(0);
       i != tmp_length_i;
       ++i, tmp_number_states_MTGP32_offset +=
            static_cast<size_t>(tmp_number_states_MTGP32_allocate)) {
    if (i + 1 != tmp_length_i) {
      tmp_number_states_MTGP32_allocate = 200;
    } else {
      tmp_number_states_MTGP32_allocate =
          number_states_MTGP32_received - 200 * i;
    }

    tmp_curandStatus_t = curandMakeMTGP32KernelState(
        ptr_curandStateMtgp32_t_received + tmp_number_states_MTGP32_offset,
        mtgp32dc_params_fast_11213, ptr_mtgp32_kernel_params_received,
        tmp_number_states_MTGP32_allocate,  // 200 Maximum states
        seed);

    if (tmp_curandStatus_t != curandStatus::CURAND_STATUS_SUCCESS) {
      ERR(
          L"curandMakeMTGP32KernelState(ptr + %d, args, args, %d, "
          "%zu) failed at %ls:%i: _%d", tmp_number_states_MTGP32_offset,
          tmp_number_states_MTGP32_allocate, seed, __FILE__, __LINE__,
          tmp_curandStatus_t);

      CUDA__Safe_Call(cudaFree(ptr_mtgp32_kernel_params_received));

      CUDA__Safe_Call(cudaFree(ptr_curandStateMtgp32_t_received));

      return false;
    }

    seed = seed == 0_UZ
                        ? static_cast<unsigned int>(
                              std::chrono::high_resolution_clock::now()
                                  .time_since_epoch()
                                  .count())
                        : seed - 1_UZ;
  }

  return true;
}

__host__ void Cleanup_cuRAND_MTGP32(
    struct mtgp32_kernel_params *&ptr_mtgp32_kernel_params_received,
    struct curandStateMtgp32 *&ptr_curandStateMtgp32_t_received) {
  CUDA__Safe_Call(cudaFree(ptr_mtgp32_kernel_params_received));

  CUDA__Safe_Call(cudaFree(ptr_curandStateMtgp32_t_received));
}

__device__ bool cuRAND_Bernoulli(float const probability_received,
                                 float const curand_uniform_received) {
  return ((probability_received == 1.0f)
              ? true
              : ((probability_received == 0.0f)
                     ? false
                     : ((curand_uniform_received <= probability_received)
                            ? true
                            : false)));
}

__global__ void kernel__cuModel__Total_Blocks_cuRAND_MTGP32(
    int *const ptr_number_states_MTGP32_received,
    enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received,
    class cuModel *const ptr_cuModel_received) {
  *ptr_number_states_MTGP32_received =
      ptr_cuModel_received->Total_Blocks_cuRAND_MTGP32(
          type_curand_generator_received);
}

__device__ int cuModel::Total_Blocks_cuRAND_MTGP32(
    enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received) {
  class cuDeviceProp *const tmp_ptr_CUDA_Device(
      this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

  size_t const tmp_maximum_states_usable(static_cast<size_t>(
      ceil(static_cast<double>(tmp_ptr_CUDA_Device->Get__Maximum_Threads()) /
           256.0)));
  size_t tmp_number_blocks;

  switch (type_curand_generator_received) {
    case ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_WEIGHTS:
      tmp_number_blocks = static_cast<size_t>(
          ceil(static_cast<double>(this->total_weights_allocated) / 256.0));
      break;
    case ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_BERNOULLI:
      for (size_t tmp_number_blocks = static_cast<size_t>(ceil(
                      static_cast<double>(
                          this->ptr_array_number_neurons_by_layer[0] - 1u) /
                      256.0)),
                  tmp_number_blocks_in_layer = 0, i = 1u;
           i != this->total_layers; ++i) {
        tmp_number_blocks_in_layer = static_cast<size_t>(
            ceil(static_cast<double>(
                     this->ptr_array_number_neurons_by_layer[i] - 1u) /
                 256.0));

        tmp_number_blocks = std::max<size_t>(
            tmp_number_blocks, tmp_number_blocks_in_layer);
      }
      break;
    default:
      return (0);
  }

  tmp_number_blocks =
      std::min<size_t>(tmp_number_blocks, tmp_maximum_states_usable);

  if (tmp_number_blocks > (std::numeric_limits<int>::max)()) {
    ERR(
        L"Overflow conversion (%zu) to int (%d). At line "
        "%d.", tmp_number_blocks, (std::numeric_limits<int>::max)(),
        __LINE__);
  }

  return (static_cast<int>(tmp_number_blocks));
}

__global__ void kernel__cuModel__Initialize_cuRAND_MTGP32(
    int const size_received,
    enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received,
    struct curandStateMtgp32 *const ptr_curandStateMtgp32_received,
    class cuModel *const ptr_cuModel_received) {
  if (ptr_cuModel_received->Initialize_cuRAND_MTGP32(
          size_received, type_curand_generator_received,
          ptr_curandStateMtgp32_received) == false) {
    ERR(L"From \"Initialize_cuRAND_MTGP32\".",);
  }
}

__device__ bool cuModel::Initialize_cuRAND_MTGP32(
    int const size_received,
    enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received,
    struct curandStateMtgp32 *const ptr_curandStateMtgp32_received) {
  if (size_received == 0) {
    ERR(
        L"Can not initialize cuRAND. Size of the array equal "
        "zero.",);

    return false;
  }

  struct mtgp32_kernel_params *tmp_ptr_array_mtgp32_kernel_params_t;

  struct dim3 tmp_dim3_grid(1, 1, 1u), tmp_dim3_block(1, 1, 1u);

  switch (type_curand_generator_received) {
    case ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_WEIGHTS: {
      // allocate cuRAND State MTGP32 parametred.
      struct curandStateMtgp32 *tmp_ptr_array_cuRAND_State_MTGP32_weighted(
          new struct curandStateMtgp32[size_received]);
      if (tmp_ptr_array_cuRAND_State_MTGP32_weighted == nullptr) {
        ERR(
            L"Can not allocate memory. new struct "
            "curandStateMtgp32(%u)[size_received(%d)]", sizeof(struct curandStateMtgp32), size_received);

        return false;
      }
      this->ptr_array_cuRAND_State_MTGP32_weighted =
          tmp_ptr_array_cuRAND_State_MTGP32_weighted;
      // |END| allocate cuRAND State MTGP32 parametred. |END|

      // copy cuRAND State MTGP32 parametred.
      Memory::Copy_Loop<struct curandStateMtgp32>(
          ptr_curandStateMtgp32_received,
          ptr_curandStateMtgp32_received + size_received,
          this->ptr_array_cuRAND_State_MTGP32_weighted);
      // |END| copy cuRAND State MTGP32 parametred. |END|

      // allocate tmp_ptr_array_mtgp32_kernel_params_t.
      tmp_ptr_array_mtgp32_kernel_params_t =
          new struct mtgp32_kernel_params[size_received];
      if (tmp_ptr_array_mtgp32_kernel_params_t == nullptr) {
        ERR(
            L"Can not allocate memory. new struct "
            "mtgp32_kernel_params(%u)[size_received(%d)]", sizeof(struct mtgp32_kernel_params), size_received);

        return false;
      }
      // |END| allocate tmp_ptr_array_mtgp32_kernel_params_t. |END|

      // Assign cuRAND State MTGP32 parametred variable.
      if (USE_PARALLEL && size_received >= warpSize) {
        this->Get__Class_Device_Information_Array()
            ->Get__CUDA_Device()
            ->Grid_Block_1Dimensions(static_cast<size_t>(size_received), 0_UZ,
                                     tmp_dim3_grid, tmp_dim3_block);
      }

      cuRAND__Memcpy_cuRAND_State_MTGP32(
          size_received, tmp_ptr_array_cuRAND_State_MTGP32_weighted,
          ptr_curandStateMtgp32_received, tmp_ptr_array_mtgp32_kernel_params_t,
          &tmp_dim3_grid, &tmp_dim3_block);

      this->number_cuRAND_State_MTGP32_weighted = size_received;
      // |END| Assign cuRAND State MTGP32 parametred variable. |END|
    } break;
    case ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_BERNOULLI: {
      // allocate cuRAND State MTGP32 neuroyed.
      struct curandStateMtgp32 *tmp_ptr_array_cuRAND_State_MTGP32_neuroyed(
          new struct curandStateMtgp32[size_received]);
      if (tmp_ptr_array_cuRAND_State_MTGP32_neuroyed == nullptr) {
        ERR(
            L"Can not allocate memory. new struct "
            "curandStateMtgp32(%u)[size_received(%d)]", sizeof(struct curandStateMtgp32), size_received);

        return false;
      }
      this->ptr_array_cuRAND_State_MTGP32_neuroyed =
          tmp_ptr_array_cuRAND_State_MTGP32_neuroyed;
      // |END| allocate cuRAND State MTGP32 neuroyed. |END|

      // copy cuRAND State MTGP32 neuroyed.
      Memory::Copy_Loop<struct curandStateMtgp32>(
          ptr_curandStateMtgp32_received,
          ptr_curandStateMtgp32_received + size_received,
          this->ptr_array_cuRAND_State_MTGP32_neuroyed);
      // |END| copy cuRAND State MTGP32 neuroyed. |END|

      // allocate tmp_ptr_array_mtgp32_kernel_params_t.
      tmp_ptr_array_mtgp32_kernel_params_t =
          new struct mtgp32_kernel_params[size_received];
      if (tmp_ptr_array_mtgp32_kernel_params_t == nullptr) {
        ERR(
            L"Can not allocate memory. new struct "
            "mtgp32_kernel_params(%u)[size_received(%d)]", sizeof(struct mtgp32_kernel_params), size_received);

        return false;
      }
      // |END| allocate tmp_ptr_array_mtgp32_kernel_params_t. |END|

      // Assign cuRAND State MTGP32 neuroyed variable.
      if (USE_PARALLEL && size_received >= warpSize) {
        this->Get__Class_Device_Information_Array()
            ->Get__CUDA_Device()
            ->Grid_Block_1Dimensions(static_cast<size_t>(size_received), 0_UZ,
                                     tmp_dim3_grid, tmp_dim3_block);
      }

      cuRAND__Memcpy_cuRAND_State_MTGP32(
          size_received, this->ptr_array_cuRAND_State_MTGP32_neuroyed,
          ptr_curandStateMtgp32_received, tmp_ptr_array_mtgp32_kernel_params_t,
          &tmp_dim3_grid, &tmp_dim3_block);

      this->number_cuRAND_State_MTGP32_neuroyed = size_received;
      // |END| Assign cuRAND State MTGP32 neuroyed variable. |END|
    } break;
    default:
      return false;
  }

  return true;
}

__host__ bool cuModel::Initialize_cuRAND(
    size_t const seed) {
  int tmp_number_states_MTGP32, *tmp_ptr_device_number_states_MTGP32(nullptr);

  CUDA__Safe_Call(
      cudaMalloc((void **)&tmp_ptr_device_number_states_MTGP32, sizeof(int)));

  // Weights
  kernel__cuModel__Total_Blocks_cuRAND_MTGP32<<<1, 1u>>>(
      tmp_ptr_device_number_states_MTGP32,
      ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_WEIGHTS, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&tmp_number_states_MTGP32,
                             tmp_ptr_device_number_states_MTGP32, sizeof(int),
                             cudaMemcpyKind::cudaMemcpyDeviceToHost));

  if (tmp_number_states_MTGP32 != 0) {
    struct mtgp32_kernel_params *tmp_ptr_mtgp32_kernel_params(NULL);

    struct curandStateMtgp32 *tmp_ptr_curandStateMtgp32_t(NULL);

    if (Allocate_cuRAND_MTGP32(tmp_number_states_MTGP32, seed,
                               tmp_ptr_mtgp32_kernel_params,
                               tmp_ptr_curandStateMtgp32_t) == false) {
      ERR(L"From \"Allocate_cuRAND_MTGP32\".",);

      CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));

      return false;
    }

    kernel__cuModel__Initialize_cuRAND_MTGP32<<<1, 1u>>>(
        tmp_number_states_MTGP32,
        ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_WEIGHTS,
        tmp_ptr_curandStateMtgp32_t, this);

    CUDA__Check_Error();

    Cleanup_cuRAND_MTGP32(tmp_ptr_mtgp32_kernel_params,
                          tmp_ptr_curandStateMtgp32_t);
  }
  // |END| Weights |END|

  // Dropout bernoulli
  kernel__cuModel__Total_Blocks_cuRAND_MTGP32<<<1, 1u>>>(
      tmp_ptr_device_number_states_MTGP32,
      ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_BERNOULLI, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&tmp_number_states_MTGP32,
                             tmp_ptr_device_number_states_MTGP32, sizeof(int),
                             cudaMemcpyKind::cudaMemcpyDeviceToHost));

  if (tmp_number_states_MTGP32 != 0) {
    struct mtgp32_kernel_params *tmp_ptr_mtgp32_kernel_params(NULL);

    struct curandStateMtgp32 *tmp_ptr_curandStateMtgp32_t(NULL);

    if (Allocate_cuRAND_MTGP32(tmp_number_states_MTGP32, seed,
                               tmp_ptr_mtgp32_kernel_params,
                               tmp_ptr_curandStateMtgp32_t) == false) {
      ERR(L"From \"Allocate_cuRAND_MTGP32\".",);

      CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));

      return false;
    }

    kernel__cuModel__Initialize_cuRAND_MTGP32<<<1, 1u>>>(
        tmp_number_states_MTGP32,
        ENUM_TYPE_CURAND_GENERATOR::TYPE_CURAND_BERNOULLI,
        tmp_ptr_curandStateMtgp32_t, this);

    CUDA__Check_Error();

    Cleanup_cuRAND_MTGP32(tmp_ptr_mtgp32_kernel_params,
                          tmp_ptr_curandStateMtgp32_t);
  }
  // |END| Dropout bernoulli |END|

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));

  return true;
}