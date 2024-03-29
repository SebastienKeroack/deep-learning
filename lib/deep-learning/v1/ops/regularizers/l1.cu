﻿/* Copyright 2016, 2019 Sébastien Kéroack. All Rights Reserved.

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

#include "deep-learning/v1/learner/model.cuh"
#include "deep-learning/ops/math.hpp"

__global__ void kernel__cuModel__Set__Regularization__L1(
    var const regularization__l1_received,
    class cuModel *const ptr_cuModel_received) {
  ptr_cuModel_received->set_l1(
      regularization__l1_received);
}

__host__ __device__ bool cuModel::set_l1(
    var const regularization__l1_received) {
#ifndef __CUDA_ARCH__
  kernel__cuModel__Set__Regularization__L1<<<1, 1u>>>(
      regularization__l1_received, this);

  CUDA__Check_Error();
#else
  if (this->regularization__l1 != regularization__l1_received) {
    bool const tmp_use_regularization(this->Use__Regularization_Parameter()),
        tmp_not_initialized_regularization(
            this->ptr_array_mask_regularized_parameters == nullptr);

    this->regularization__l1 = regularization__l1_received;

    if (tmp_use_regularization == false && regularization__l1_received != 0_r) {
      if (this->Allocate__Parameter__Regularization() == false) {
        ERR(
            L"Can not allocate regularization connections!",);

        return false;
      }

      if (tmp_not_initialized_regularization) {
        this->Indexing_Regularization_Parameters();
      }
    }

    if (this->Use__Regularization_Parameter() == false) {
      this->Deallocate__Parameter__Regularization();
    }
  }
#endif

  return true;
}

template <typename T>
__global__ void kernel__Update_Derivative_Weight__Regularization__L1(
    T const regularization__l1_received, T *const ptr_array_gradients_received,
    T const *const ptr_array_parameters_received,
    T const *const ptr_array_connections_mask_regularization_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_gradients_received[tmp_thread_global_index] +=
      ptr_array_connections_mask_regularization_received
          [tmp_thread_global_index] *
      DL::Math::sign<var>(
          ptr_array_parameters_received[tmp_thread_global_index]) *
      regularization__l1_received;
}

template <typename T>
__global__ void kernel__Update_Derivative_Weight__Regularization__L1(
    size_t const size_received, T const regularization__l1_received,
    T *const ptr_array_gradients_received,
    T const *const ptr_array_parameters_received,
    T const *const ptr_array_connections_mask_regularization_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    ptr_array_gradients_received[tmp_thread_global_index] +=
        ptr_array_connections_mask_regularization_received
            [tmp_thread_global_index] *
        DL::Math::sign<var>(
            ptr_array_parameters_received[tmp_thread_global_index]) *
        regularization__l1_received;
  }
}

template <typename T>
__global__ void kernel_while__Update_Derivative_Weight__Regularization__L1(
    size_t const size_received, T const regularization__l1_received,
    T *const ptr_array_gradients_received,
    T const *const ptr_array_parameters_received,
    T const *const ptr_array_connections_mask_regularization_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_gradients_received[tmp_thread_global_index] +=
        ptr_array_connections_mask_regularization_received
            [tmp_thread_global_index] *
        DL::Math::sign<var>(
            ptr_array_parameters_received[tmp_thread_global_index]) *
        regularization__l1_received;

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

__device__ void
cuModel::Update_Derivative_Weight__Regularization__L1(
    size_t const batch_size) {
  var *tmp_ptr_gradient_it(this->ptr_array_derivatives_parameters);
  var const tmp_regularization__l1(this->regularization__l1 *
                                  batch_size),
      *tmp_ptr_weight_it(this->ptr_array_parameters),
      *tmp_ptr_connections_mask_regularization_it(
          this->ptr_array_mask_regularized_parameters);

  if (USE_PARALLEL && this->total_weights >= warpSize) {
    LAUNCH_KERNEL_1D(
        Update_Derivative_Weight__Regularization__L1<var>,
        this->ptr_array_dim3_grid[2], this->ptr_array_dim3_block[2], 0_UZ,
        this->total_weights, tmp_regularization__l1, tmp_ptr_gradient_it,
        tmp_ptr_weight_it, tmp_ptr_connections_mask_regularization_it)

    CUDA__Check_Error();
  } else {
    for (var const *const tmp_ptr_last_gradient(tmp_ptr_gradient_it +
                                               this->total_weights);
         tmp_ptr_gradient_it != tmp_ptr_last_gradient;
         ++tmp_ptr_gradient_it, ++tmp_ptr_weight_it,
         ++tmp_ptr_connections_mask_regularization_it) {
      *tmp_ptr_gradient_it += *tmp_ptr_connections_mask_regularization_it *
                              DL::Math::sign<var>(*tmp_ptr_weight_it) *
                              tmp_regularization__l1;
    }
  }
}

__host__ __device__ var
cuModel::get_l1(void) const {
  return (this->regularization__l1);
}
