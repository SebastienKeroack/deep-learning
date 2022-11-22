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

#include "deep-learning-lib/v1/learner/model.cuh"
#include "deep-learning-lib/ops/math.hpp"

__device__ void cuModel::Update_Parameter__iRPROP_plus(
    size_t const start_index_received, size_t const end_index_received) {
  if (this->use_Dropout) {
    this->Update_Parameter__iRPROP_plus__CUDA__Dropout(start_index_received,
                                                       end_index_received);
  } else {
    this->Update_Parameter__iRPROP_plus__CUDA(start_index_received,
                                              end_index_received);
  }
}

template <typename T>
__global__ void kernel__Update_Parameter__iRPROP_plus(
    bool const error_is_worst_received, T const increase_factor_received,
    T const decrease_factor_received, T const minimum_delta_received,
    T const maximum_delta_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_weight_received,
    T *const ptr_array_previous_delta_weight_received,
    T *const ptr_array_previous_steps_received,
    T *const ptr_array_previous_partial_derivative_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  extern __shared__ T
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative
          [];
  /* Index map:
      0: delta_step
      1: partial_derivative
      2: delta_weight */
  T(&tmp_ptr_array_smem)
  [] =
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;

  tmp_ptr_array_smem[threadIdx.x + blockDim.x] =
      ptr_array_partial_derivative_received[tmp_thread_global_index];

  T const tmp_sign(
      ptr_array_previous_partial_derivative_received[tmp_thread_global_index] *
      tmp_ptr_array_smem[threadIdx.x + blockDim.x]);

  if (tmp_sign > T(0)) {
    tmp_ptr_array_smem[threadIdx.x] =
        ptr_array_previous_steps_received[tmp_thread_global_index] *
        increase_factor_received;
    ptr_array_previous_steps_received[tmp_thread_global_index] =
        tmp_ptr_array_smem[threadIdx.x] = std::min<var>(
            tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);

    ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
        tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
            -DL::Math::sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
            tmp_ptr_array_smem[threadIdx.x];

    ptr_array_weight_received[tmp_thread_global_index] +=
        tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

    ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
        tmp_ptr_array_smem[threadIdx.x + blockDim.x];
  } else if (tmp_sign < T(0)) {
    tmp_ptr_array_smem[threadIdx.x] =
        ptr_array_previous_steps_received[tmp_thread_global_index] *
        decrease_factor_received;
    ptr_array_previous_steps_received[tmp_thread_global_index] =
        std::max<var>(tmp_ptr_array_smem[threadIdx.x],
                                minimum_delta_received);

    if (error_is_worst_received) {
      ptr_array_weight_received[tmp_thread_global_index] -=
          ptr_array_previous_delta_weight_received[tmp_thread_global_index];
    }

    ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
        T(0);
  } else  // if(tmp_sign == T(0))
  {
    ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
        tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
            -DL::Math::sign<T>(tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
            ptr_array_previous_steps_received[tmp_thread_global_index];

    ptr_array_weight_received[tmp_thread_global_index] +=
        tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

    ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
        tmp_ptr_array_smem[threadIdx.x + blockDim.x];
  }

  ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template <typename T>
__global__ void kernel__Update_Parameter__iRPROP_plus(
    size_t const size_received, bool const error_is_worst_received,
    T const increase_factor_received, T const decrease_factor_received,
    T const minimum_delta_received, T const maximum_delta_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_weight_received,
    T *const ptr_array_previous_delta_weight_received,
    T *const ptr_array_previous_steps_received,
    T *const ptr_array_previous_partial_derivative_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  extern __shared__ T
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative
          [];
  /* Index map:
      0: delta_step
      1: partial_derivative
      2: delta_weight */
  T(&tmp_ptr_array_smem)
  [] =
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;

  if (tmp_thread_global_index < size_received) {
    tmp_ptr_array_smem[threadIdx.x + blockDim.x] =
        ptr_array_partial_derivative_received[tmp_thread_global_index];

    T const tmp_sign(ptr_array_previous_partial_derivative_received
                         [tmp_thread_global_index] *
                     tmp_ptr_array_smem[threadIdx.x + blockDim.x]);

    if (tmp_sign > T(0)) {
      tmp_ptr_array_smem[threadIdx.x] =
          ptr_array_previous_steps_received[tmp_thread_global_index] *
          increase_factor_received;
      ptr_array_previous_steps_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x] = std::min<var>(
              tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);

      ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
              -DL::Math::sign<T>(
                  tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
              tmp_ptr_array_smem[threadIdx.x];

      ptr_array_weight_received[tmp_thread_global_index] +=
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

      ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + blockDim.x];
    } else if (tmp_sign < T(0)) {
      tmp_ptr_array_smem[threadIdx.x] =
          ptr_array_previous_steps_received[tmp_thread_global_index] *
          decrease_factor_received;
      ptr_array_previous_steps_received[tmp_thread_global_index] =
          std::max<var>(tmp_ptr_array_smem[threadIdx.x],
                                  minimum_delta_received);

      if (error_is_worst_received) {
        ptr_array_weight_received[tmp_thread_global_index] -=
            ptr_array_previous_delta_weight_received[tmp_thread_global_index];
      }

      ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
          T(0);
    } else  // if(tmp_sign == T(0))
    {
      ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
              -DL::Math::sign<T>(
                  tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
              ptr_array_previous_steps_received[tmp_thread_global_index];

      ptr_array_weight_received[tmp_thread_global_index] +=
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

      ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + blockDim.x];
    }

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
  }
}

template <typename T>
__global__ void kernel_while__Update_Parameter__iRPROP_plus(
    size_t const size_received, bool const error_is_worst_received,
    T const increase_factor_received, T const decrease_factor_received,
    T const minimum_delta_received, T const maximum_delta_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_weight_received,
    T *const ptr_array_previous_delta_weight_received,
    T *const ptr_array_previous_steps_received,
    T *const ptr_array_previous_partial_derivative_received) {
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  extern __shared__ T
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative
          [];
  /* Index map:
      0: delta_step
      1: partial_derivative
      2: delta_weight */
  T(&tmp_ptr_array_smem)
  [] =
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;
  T tmp_sign;

  do {
    tmp_ptr_array_smem[threadIdx.x + blockDim.x] =
        ptr_array_partial_derivative_received[tmp_thread_global_index];

    tmp_sign = ptr_array_previous_partial_derivative_received
                   [tmp_thread_global_index] *
               tmp_ptr_array_smem[threadIdx.x + blockDim.x];

    if (tmp_sign > T(0)) {
      tmp_ptr_array_smem[threadIdx.x] =
          ptr_array_previous_steps_received[tmp_thread_global_index] *
          increase_factor_received;
      ptr_array_previous_steps_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x] = std::min<var>(
              tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);

      ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
              -DL::Math::sign<T>(
                  tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
              tmp_ptr_array_smem[threadIdx.x];

      ptr_array_weight_received[tmp_thread_global_index] +=
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

      ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + blockDim.x];
    } else if (tmp_sign < T(0)) {
      tmp_ptr_array_smem[threadIdx.x] =
          ptr_array_previous_steps_received[tmp_thread_global_index] *
          decrease_factor_received;
      ptr_array_previous_steps_received[tmp_thread_global_index] =
          std::max<var>(tmp_ptr_array_smem[threadIdx.x],
                                  minimum_delta_received);

      if (error_is_worst_received) {
        ptr_array_weight_received[tmp_thread_global_index] -=
            ptr_array_previous_delta_weight_received[tmp_thread_global_index];
      }

      ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
          T(0);
    } else  // if(tmp_sign == T(0))
    {
      ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
              -DL::Math::sign<T>(
                  tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
              ptr_array_previous_steps_received[tmp_thread_global_index];

      ptr_array_weight_received[tmp_thread_global_index] +=
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

      ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + blockDim.x];
    }

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

    tmp_thread_global_index += gridDim.x * blockDim.x;
  } while (tmp_thread_global_index < size_received);
}

__device__ void cuModel::Update_Parameter__iRPROP_plus__CUDA(
    size_t const start_index_received, size_t const end_index_received) {
  bool const tmp_error_is_worst(this->loss_rprop > this->loss_rprop_tm1);

  float const tmp_increase_factor(this->rprop_increase_factor),  // 1.2
      tmp_decrease_factor(this->rprop_decrease_factor),          // 0.5
      tmp_delta_minimum(this->rprop_delta_min),                  // 1e-6
      tmp_delta_maximum(this->rprop_delta_max);                  // 50.0

  var *const tmp_ptr_array_partial_derivative(
      this->ptr_array_derivatives_parameters),
      *const tmp_ptr_array_parameters(this->ptr_array_parameters),
          *const tmp_ptr_array_previous_delta_weight(
              this->ptr_array_previous_delta_parameters),
              *const tmp_ptr_array_previous_step(
                  this->ptr_array_previous_steps),
                  *const tmp_ptr_array_previous_partial_derivative(
                      this->ptr_array_previous_derivatives_parameters),
      tmp_partial_derivative, tmp_delta_weight, tmp_delta_step;

  if (USE_PARALLEL && end_index_received - start_index_received >= warpSize) {
    LAUNCH_KERNEL_1D(
        Update_Parameter__iRPROP_plus<var>, this->ptr_array_dim3_grid[1],
        this->ptr_array_dim3_block[1],
        this->ptr_array_dim3_block[1].x * 3u * sizeof(var),
        end_index_received - start_index_received, tmp_error_is_worst,
        tmp_increase_factor, tmp_decrease_factor, tmp_delta_minimum,
        tmp_delta_maximum,
        tmp_ptr_array_partial_derivative + start_index_received,
        tmp_ptr_array_parameters + start_index_received,
        tmp_ptr_array_previous_delta_weight + start_index_received,
        tmp_ptr_array_previous_step + start_index_received,
        tmp_ptr_array_previous_partial_derivative + start_index_received)

    CUDA__Check_Error();
  } else {
    for (size_t i(start_index_received); i != end_index_received; ++i) {
      tmp_partial_derivative =
          tmp_ptr_array_partial_derivative[i];  // Gradient descent

      if (tmp_ptr_array_previous_partial_derivative[i] *
              tmp_partial_derivative >
          0_r) {
        tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_increase_factor;
        tmp_ptr_array_previous_step[i] = tmp_delta_step =
            std::min<var>(tmp_delta_step, tmp_delta_maximum);

        tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight =
            -DL::Math::sign<var>(tmp_partial_derivative) * tmp_delta_step;

        tmp_ptr_array_parameters[i] += tmp_delta_weight;

        tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
      } else if (tmp_ptr_array_previous_partial_derivative[i] *
                     tmp_partial_derivative <
                 0_r) {
        tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_decrease_factor;
        tmp_ptr_array_previous_step[i] =
            std::max<var>(tmp_delta_step, tmp_delta_minimum);

        if (tmp_error_is_worst) {
          tmp_ptr_array_parameters[i] -= tmp_ptr_array_previous_delta_weight[i];
        }

        tmp_ptr_array_previous_partial_derivative[i] = 0_r;
      } else  // if(tmp_ptr_array_previous_partial_derivative[i] *
              // tmp_partial_derivative == 0_r)
      {
        tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight =
            -DL::Math::sign<var>(tmp_partial_derivative) *
            tmp_ptr_array_previous_step[i];

        tmp_ptr_array_parameters[i] += tmp_delta_weight;

        tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
      }

      tmp_ptr_array_partial_derivative[i] = 0_r;  // tmp_partial_derivative
    }
  }
}

template <typename T>
__global__ void kernel__Update_Parameter__iRPROP_plus__Dropout(
    bool const error_is_worst_received, T const increase_factor_received,
    T const decrease_factor_received, T const minimum_delta_received,
    T const maximum_delta_received,
    T const *const ptr_array_mask_dropout_parameters_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_weight_received,
    T *const ptr_array_previous_delta_weight_received,
    T *const ptr_array_previous_steps_received,
    T *const ptr_array_previous_partial_derivative_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  extern __shared__ T
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative
          [];
  /* Index map:
      0: delta_step
      1: partial_derivative
      2: delta_weight */
  T(&tmp_ptr_array_smem)
  [] =
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;

  tmp_ptr_array_smem[threadIdx.x + blockDim.x] =
      ptr_array_partial_derivative_received[tmp_thread_global_index];

  T const tmp_sign(
      ptr_array_previous_partial_derivative_received[tmp_thread_global_index] *
      tmp_ptr_array_smem[threadIdx.x + blockDim.x]);

  if (ptr_array_mask_dropout_parameters_received[tmp_thread_global_index] ==
      T(0)) {
    ptr_array_previous_delta_weight_received[tmp_thread_global_index] = T(0);
  } else {
    if (tmp_sign > T(0)) {
      tmp_ptr_array_smem[threadIdx.x] =
          ptr_array_previous_steps_received[tmp_thread_global_index] *
          increase_factor_received;
      ptr_array_previous_steps_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x] = std::min<var>(
              tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);

      ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
              -DL::Math::sign<T>(
                  tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
              tmp_ptr_array_smem[threadIdx.x];

      ptr_array_weight_received[tmp_thread_global_index] +=
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

      ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + blockDim.x];
    } else if (tmp_sign < T(0)) {
      tmp_ptr_array_smem[threadIdx.x] =
          ptr_array_previous_steps_received[tmp_thread_global_index] *
          decrease_factor_received;
      ptr_array_previous_steps_received[tmp_thread_global_index] =
          std::max<var>(tmp_ptr_array_smem[threadIdx.x],
                                  minimum_delta_received);

      if (error_is_worst_received) {
        ptr_array_weight_received[tmp_thread_global_index] -=
            ptr_array_previous_delta_weight_received[tmp_thread_global_index];
      }

      ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
          T(0);
    } else if (ptr_array_previous_partial_derivative_received
                   [tmp_thread_global_index] == T(0)) {
      ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
              -DL::Math::sign<T>(
                  tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
              ptr_array_previous_steps_received[tmp_thread_global_index];

      ptr_array_weight_received[tmp_thread_global_index] +=
          tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

      ptr_array_previous_partial_derivative_received[tmp_thread_global_index] =
          tmp_ptr_array_smem[threadIdx.x + blockDim.x];
    } else {
      ptr_array_previous_delta_weight_received[tmp_thread_global_index] = T(0);
    }
  }

  ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template <typename T>
__global__ void kernel__Update_Parameter__iRPROP_plus__Dropout(
    size_t const size_received, bool const error_is_worst_received,
    T const increase_factor_received, T const decrease_factor_received,
    T const minimum_delta_received, T const maximum_delta_received,
    T const *const ptr_array_mask_dropout_parameters_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_weight_received,
    T *const ptr_array_previous_delta_weight_received,
    T *const ptr_array_previous_steps_received,
    T *const ptr_array_previous_partial_derivative_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  extern __shared__ T
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative
          [];
  /* Index map:
      0: delta_step
      1: partial_derivative
      2: delta_weight */
  T(&tmp_ptr_array_smem)
  [] =
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;

  if (tmp_thread_global_index < size_received) {
    tmp_ptr_array_smem[threadIdx.x + blockDim.x] =
        ptr_array_partial_derivative_received[tmp_thread_global_index];

    T const tmp_sign(ptr_array_previous_partial_derivative_received
                         [tmp_thread_global_index] *
                     tmp_ptr_array_smem[threadIdx.x + blockDim.x]);

    if (ptr_array_mask_dropout_parameters_received[tmp_thread_global_index] ==
        T(0)) {
      ptr_array_previous_delta_weight_received[tmp_thread_global_index] = T(0);
    } else {
      if (tmp_sign > T(0)) {
        tmp_ptr_array_smem[threadIdx.x] =
            ptr_array_previous_steps_received[tmp_thread_global_index] *
            increase_factor_received;
        ptr_array_previous_steps_received[tmp_thread_global_index] =
            tmp_ptr_array_smem[threadIdx.x] = std::min<var>(
                tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);

        ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
                -DL::Math::sign<T>(
                    tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
                tmp_ptr_array_smem[threadIdx.x];

        ptr_array_weight_received[tmp_thread_global_index] +=
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

        ptr_array_previous_partial_derivative_received
            [tmp_thread_global_index] =
                tmp_ptr_array_smem[threadIdx.x + blockDim.x];
      } else if (tmp_sign < T(0)) {
        tmp_ptr_array_smem[threadIdx.x] =
            ptr_array_previous_steps_received[tmp_thread_global_index] *
            decrease_factor_received;
        ptr_array_previous_steps_received[tmp_thread_global_index] =
            std::max<var>(tmp_ptr_array_smem[threadIdx.x],
                                    minimum_delta_received);

        if (error_is_worst_received) {
          ptr_array_weight_received[tmp_thread_global_index] -=
              ptr_array_previous_delta_weight_received[tmp_thread_global_index];
        }

        ptr_array_previous_partial_derivative_received
            [tmp_thread_global_index] = T(0);
      } else if (ptr_array_previous_partial_derivative_received
                     [tmp_thread_global_index] == T(0)) {
        ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
                -DL::Math::sign<T>(
                    tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
                ptr_array_previous_steps_received[tmp_thread_global_index];

        ptr_array_weight_received[tmp_thread_global_index] +=
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

        ptr_array_previous_partial_derivative_received
            [tmp_thread_global_index] =
                tmp_ptr_array_smem[threadIdx.x + blockDim.x];
      } else {
        ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
            T(0);
      }
    }

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
  }
}

template <typename T>
__global__ void kernel_while__Update_Parameter__iRPROP_plus__Dropout(
    size_t const size_received, bool const error_is_worst_received,
    T const increase_factor_received, T const decrease_factor_received,
    T const minimum_delta_received, T const maximum_delta_received,
    T const *const ptr_array_mask_dropout_parameters_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_weight_received,
    T *const ptr_array_previous_delta_weight_received,
    T *const ptr_array_previous_steps_received,
    T *const ptr_array_previous_partial_derivative_received) {
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  extern __shared__ T
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative
          [];
  /* Index map:
      0: delta_step
      1: partial_derivative
      2: delta_weight */
  T(&tmp_ptr_array_smem)
  [] =
      tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;

  do {
    tmp_ptr_array_smem[threadIdx.x + blockDim.x] =
        ptr_array_partial_derivative_received[tmp_thread_global_index];

    T const tmp_sign(ptr_array_previous_partial_derivative_received
                         [tmp_thread_global_index] *
                     tmp_ptr_array_smem[threadIdx.x + blockDim.x]);

    if (ptr_array_mask_dropout_parameters_received[tmp_thread_global_index] ==
        T(0)) {
      ptr_array_previous_delta_weight_received[tmp_thread_global_index] = T(0);
    } else {
      if (tmp_sign > T(0)) {
        tmp_ptr_array_smem[threadIdx.x] =
            ptr_array_previous_steps_received[tmp_thread_global_index] *
            increase_factor_received;
        ptr_array_previous_steps_received[tmp_thread_global_index] =
            tmp_ptr_array_smem[threadIdx.x] = std::min<var>(
                tmp_ptr_array_smem[threadIdx.x], maximum_delta_received);

        ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
                -DL::Math::sign<T>(
                    tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
                tmp_ptr_array_smem[threadIdx.x];

        ptr_array_weight_received[tmp_thread_global_index] +=
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

        ptr_array_previous_partial_derivative_received
            [tmp_thread_global_index] =
                tmp_ptr_array_smem[threadIdx.x + blockDim.x];
      } else if (tmp_sign < T(0)) {
        tmp_ptr_array_smem[threadIdx.x] =
            ptr_array_previous_steps_received[tmp_thread_global_index] *
            decrease_factor_received;
        ptr_array_previous_steps_received[tmp_thread_global_index] =
            std::max<var>(tmp_ptr_array_smem[threadIdx.x],
                                    minimum_delta_received);

        if (error_is_worst_received) {
          ptr_array_weight_received[tmp_thread_global_index] -=
              ptr_array_previous_delta_weight_received[tmp_thread_global_index];
        }

        ptr_array_previous_partial_derivative_received
            [tmp_thread_global_index] = T(0);
      } else if (ptr_array_previous_partial_derivative_received
                     [tmp_thread_global_index] == T(0)) {
        ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] =
                -DL::Math::sign<T>(
                    tmp_ptr_array_smem[threadIdx.x + blockDim.x]) *
                ptr_array_previous_steps_received[tmp_thread_global_index];

        ptr_array_weight_received[tmp_thread_global_index] +=
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x];

        ptr_array_previous_partial_derivative_received
            [tmp_thread_global_index] =
                tmp_ptr_array_smem[threadIdx.x + blockDim.x];
      } else {
        ptr_array_previous_delta_weight_received[tmp_thread_global_index] =
            T(0);
      }
    }

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

    tmp_thread_global_index += gridDim.x * blockDim.x;
  } while (tmp_thread_global_index < size_received);
}

__device__ void
cuModel::Update_Parameter__iRPROP_plus__CUDA__Dropout(
    size_t const start_index_received, size_t const end_index_received) {
  bool const tmp_error_is_worst(this->loss_rprop > this->loss_rprop_tm1);

  float const tmp_increase_factor(this->rprop_increase_factor),  // 1.2
      tmp_decrease_factor(this->rprop_decrease_factor),          // 0.5
      tmp_delta_minimum(this->rprop_delta_min),                  // 1e-6
      tmp_delta_maximum(this->rprop_delta_max);                  // 50.0

  var const *const tmp_ptr_array_mask_dropout_parameters(
      this->ptr_array_mask_dropout_parameters);
  var *const tmp_ptr_array_partial_derivative(
      this->ptr_array_derivatives_parameters),
      *const tmp_ptr_array_parameters(this->ptr_array_parameters),
          *const tmp_ptr_array_previous_delta_weight(
              this->ptr_array_previous_delta_parameters),
              *const tmp_ptr_array_previous_step(
                  this->ptr_array_previous_steps),
                  *const tmp_ptr_array_previous_partial_derivative(
                      this->ptr_array_previous_derivatives_parameters),
      tmp_partial_derivative, tmp_delta_weight, tmp_delta_step;

  if (USE_PARALLEL && end_index_received - start_index_received >= warpSize) {
    LAUNCH_KERNEL_1D(
        Update_Parameter__iRPROP_plus__Dropout<var>,
        this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1],
        this->ptr_array_dim3_block[1].x * 3u * sizeof(var),
        end_index_received - start_index_received, tmp_error_is_worst,
        tmp_increase_factor, tmp_decrease_factor, tmp_delta_minimum,
        tmp_delta_maximum,
        tmp_ptr_array_mask_dropout_parameters + start_index_received,
        tmp_ptr_array_partial_derivative + start_index_received,
        tmp_ptr_array_parameters + start_index_received,
        tmp_ptr_array_previous_delta_weight + start_index_received,
        tmp_ptr_array_previous_step + start_index_received,
        tmp_ptr_array_previous_partial_derivative + start_index_received)

    CUDA__Check_Error();
  } else {
    for (size_t i(start_index_received); i != end_index_received; ++i) {
      tmp_partial_derivative =
          tmp_ptr_array_partial_derivative[i];  // Gradient descent

      if (tmp_ptr_array_mask_dropout_parameters[i] == 0_r) {
        tmp_ptr_array_previous_delta_weight[i] = 0_r;
      } else {
        if (tmp_ptr_array_previous_partial_derivative[i] *
                tmp_partial_derivative >
            0_r) {
          tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_increase_factor;
          tmp_ptr_array_previous_step[i] = tmp_delta_step =
              std::min<var>(tmp_delta_step, tmp_delta_maximum);

          tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight =
              -DL::Math::sign<var>(tmp_partial_derivative) * tmp_delta_step;

          tmp_ptr_array_parameters[i] += tmp_delta_weight;

          tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
        } else if (tmp_ptr_array_previous_partial_derivative[i] *
                       tmp_partial_derivative <
                   0_r) {
          tmp_delta_step = tmp_ptr_array_previous_step[i] * tmp_decrease_factor;
          tmp_ptr_array_previous_step[i] =
              std::max<var>(tmp_delta_step, tmp_delta_minimum);

          if (tmp_error_is_worst) {
            tmp_ptr_array_parameters[i] -=
                tmp_ptr_array_previous_delta_weight[i];
          }

          tmp_ptr_array_previous_partial_derivative[i] = 0_r;
        } else if (tmp_ptr_array_previous_partial_derivative[i] == 0_r) {
          tmp_ptr_array_previous_delta_weight[i] = tmp_delta_weight =
              -DL::Math::sign<var>(tmp_partial_derivative) *
              tmp_ptr_array_previous_step[i];

          tmp_ptr_array_parameters[i] += tmp_delta_weight;

          tmp_ptr_array_previous_partial_derivative[i] = tmp_partial_derivative;
        } else {
          tmp_ptr_array_previous_delta_weight[i] = 0_r;
        }
      }

      tmp_ptr_array_partial_derivative[i] = 0_r;  // tmp_partial_derivative
    }
  }
}
