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

template <typename T>
__global__ void kernel__Update_Parameter__Gradient_Descent_Momentum(
    T const learning_rate_received, T const learning_momentum_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_delta_weights_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_delta_weigth;

  ptr_array_previous_delta_weights_received[tmp_thread_global_index] =
      tmp_delta_weigth =
          learning_rate_received *
              ptr_array_partial_derivative_received[tmp_thread_global_index] +
          learning_momentum_received * ptr_array_previous_delta_weights_received
                                           [tmp_thread_global_index];

  ptr_array_parameters_received[tmp_thread_global_index] -= tmp_delta_weigth;

  ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template <typename T>
__global__ void kernel__Update_Parameter__Gradient_Descent_Momentum(
    size_t const size_received, T const learning_rate_received,
    T const learning_momentum_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_delta_weights_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_delta_weigth;

  if (tmp_thread_global_index < size_received) {
    ptr_array_previous_delta_weights_received[tmp_thread_global_index] =
        tmp_delta_weigth =
            learning_rate_received *
                ptr_array_partial_derivative_received[tmp_thread_global_index] +
            learning_momentum_received *
                ptr_array_previous_delta_weights_received
                    [tmp_thread_global_index];

    ptr_array_parameters_received[tmp_thread_global_index] -= tmp_delta_weigth;

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
  }
}

template <typename T>
__global__ void kernel_while__Update_Parameter__Gradient_Descent_Momentum(
    size_t const size_received, T const learning_rate_received,
    T const learning_momentum_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_delta_weights_received) {
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_delta_weigth;

  do {
    ptr_array_previous_delta_weights_received[tmp_thread_global_index] =
        tmp_delta_weigth =
            learning_rate_received *
                ptr_array_partial_derivative_received[tmp_thread_global_index] +
            learning_momentum_received *
                ptr_array_previous_delta_weights_received
                    [tmp_thread_global_index];

    ptr_array_parameters_received[tmp_thread_global_index] -= tmp_delta_weigth;

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

    tmp_thread_global_index += gridDim.x * blockDim.x;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__global__ void kernel__Update_Parameter__Gradient_Descent_Momentum(
    T const weight_decay_received, T const learning_rate_received,
    T const learning_momentum_received,
    T const *const ptr_array_connections_mask_rergularization_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_delta_weights_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_delta_weigth;

  ptr_array_previous_delta_weights_received[tmp_thread_global_index] =
      tmp_delta_weigth =
          learning_rate_received *
              ptr_array_partial_derivative_received[tmp_thread_global_index] +
          learning_momentum_received * ptr_array_previous_delta_weights_received
                                           [tmp_thread_global_index];

  ptr_array_parameters_received[tmp_thread_global_index] -=
      tmp_delta_weigth +
      ptr_array_connections_mask_rergularization_received
              [tmp_thread_global_index] *
          weight_decay_received *
          ptr_array_parameters_received[tmp_thread_global_index];

  ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template <typename T>
__global__ void kernel__Update_Parameter__Gradient_Descent_Momentum(
    size_t const size_received, T const weight_decay_received,
    T const learning_rate_received, T const learning_momentum_received,
    T const *const ptr_array_connections_mask_rergularization_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_delta_weights_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_delta_weigth;

  if (tmp_thread_global_index < size_received) {
    ptr_array_previous_delta_weights_received[tmp_thread_global_index] =
        tmp_delta_weigth =
            learning_rate_received *
                ptr_array_partial_derivative_received[tmp_thread_global_index] +
            learning_momentum_received *
                ptr_array_previous_delta_weights_received
                    [tmp_thread_global_index];

    ptr_array_parameters_received[tmp_thread_global_index] -=
        tmp_delta_weigth +
        ptr_array_connections_mask_rergularization_received
                [tmp_thread_global_index] *
            weight_decay_received *
            ptr_array_parameters_received[tmp_thread_global_index];

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
  }
}

template <typename T>
__global__ void kernel_while__Update_Parameter__Gradient_Descent_Momentum(
    size_t const size_received, T const weight_decay_received,
    T const learning_rate_received, T const learning_momentum_received,
    T const *const ptr_array_connections_mask_rergularization_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_delta_weights_received) {
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_delta_weigth;

  do {
    ptr_array_previous_delta_weights_received[tmp_thread_global_index] =
        tmp_delta_weigth =
            learning_rate_received *
                ptr_array_partial_derivative_received[tmp_thread_global_index] +
            learning_momentum_received *
                ptr_array_previous_delta_weights_received
                    [tmp_thread_global_index];

    ptr_array_parameters_received[tmp_thread_global_index] -=
        tmp_delta_weigth +
        ptr_array_connections_mask_rergularization_received
                [tmp_thread_global_index] *
            weight_decay_received *
            ptr_array_parameters_received[tmp_thread_global_index];

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

    tmp_thread_global_index += gridDim.x * blockDim.x;
  } while (tmp_thread_global_index < size_received);
}

__device__ void
cuModel::Update_Parameter__Gradient_Descent_Momentum__CUDA(
    size_t const batch_size, size_t const training_size,
    size_t const start_index_received, size_t const end_index_received) {
  var const lr_scale(this->use_warm_restarts
                                       ? this->warm_restarts_decay() /
                                             this->learning_rate
                                       : 1_r);

  ++this->optimizer_time_step;

  var const *const mask(
      this->ptr_array_mask_regularized_parameters),
      lr(lr_scale * this->learning_rate),
      lr_mom(this->learning_momentum),
      wd(lr_scale *
                       (this->use_normalized_weight_decay
                            ? this->normalized_wd(
                                  batch_size, training_size)
                            : this->weight_decay));
  var *const tmp_ptr_array_partial_derivative(
      this->ptr_array_derivatives_parameters),
      *const tmp_ptr_array_parameters(this->ptr_array_parameters),
          *const tmp_ptr_array_previous_delta_weights(
              this->ptr_array_previous_delta_parameters),
      tmp_delta_weigth;

  if (wd != 0_r) {
    if (USE_PARALLEL && end_index_received - start_index_received >= warpSize) {
      // KERNEL LAUNCH
      //    1: Launching do-while elements.
      if (this->ptr_array_dim3_grid[1].x * this->ptr_array_dim3_block[1].x <
          end_index_received - start_index_received) {
        kernel_while__Update_Parameter__Gradient_Descent_Momentum<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                end_index_received - start_index_received, wd,
                lr, lr_mom,
                mask +
                    start_index_received,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_delta_weights + start_index_received);
      }
      //    2: Launching size condition.
      else if (this->ptr_array_dim3_grid[1].x *
                   this->ptr_array_dim3_block[1].x >
               end_index_received - start_index_received) {
        kernel__Update_Parameter__Gradient_Descent_Momentum<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                end_index_received - start_index_received, wd,
                lr, lr_mom,
                mask +
                    start_index_received,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_delta_weights + start_index_received);
      }
      //    3: Standard.
      else {
        kernel__Update_Parameter__Gradient_Descent_Momentum<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                wd, lr, lr_mom,
                mask +
                    start_index_received,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_delta_weights + start_index_received);
      }
      // |END| KERNEL LAUNCH |END|

      CUDA__Check_Error();
    } else {
      for (size_t tmp_thread_global_index(start_index_received);
           tmp_thread_global_index != end_index_received;
           ++tmp_thread_global_index) {
        tmp_ptr_array_previous_delta_weights[tmp_thread_global_index] =
            tmp_delta_weigth =
                lr *
                    tmp_ptr_array_partial_derivative[tmp_thread_global_index] +
                lr_mom * tmp_ptr_array_previous_delta_weights
                                            [tmp_thread_global_index];

        tmp_ptr_array_parameters[tmp_thread_global_index] -=
            tmp_delta_weigth -
            mask
                    [tmp_thread_global_index] *
                wd *
                tmp_ptr_array_parameters[tmp_thread_global_index];

        tmp_ptr_array_partial_derivative[tmp_thread_global_index] = 0_r;
      }
    }
  } else {
    if (USE_PARALLEL && end_index_received - start_index_received >= warpSize) {
      // KERNEL LAUNCH
      //    1: Launching do-while elements.
      if (this->ptr_array_dim3_grid[1].x * this->ptr_array_dim3_block[1].x <
          end_index_received - start_index_received) {
        kernel_while__Update_Parameter__Gradient_Descent_Momentum<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                end_index_received - start_index_received, lr,
                lr_mom,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_delta_weights + start_index_received);
      }
      //    2: Launching size condition.
      else if (this->ptr_array_dim3_grid[1].x *
                   this->ptr_array_dim3_block[1].x >
               end_index_received - start_index_received) {
        kernel__Update_Parameter__Gradient_Descent_Momentum<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                end_index_received - start_index_received, lr,
                lr_mom,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_delta_weights + start_index_received);
      }
      //    3: Standard.
      else {
        kernel__Update_Parameter__Gradient_Descent_Momentum<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                lr, lr_mom,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_delta_weights + start_index_received);
      }
      // |END| KERNEL LAUNCH |END|

      CUDA__Check_Error();
    } else {
      for (size_t tmp_thread_global_index(start_index_received);
           tmp_thread_global_index != end_index_received;
           ++tmp_thread_global_index) {
        tmp_ptr_array_previous_delta_weights[tmp_thread_global_index] =
            tmp_delta_weigth =
                lr *
                    tmp_ptr_array_partial_derivative[tmp_thread_global_index] +
                lr_mom * tmp_ptr_array_previous_delta_weights
                                            [tmp_thread_global_index];

        tmp_ptr_array_parameters[tmp_thread_global_index] -=
            tmp_delta_weigth;  // Gradient descent

        tmp_ptr_array_partial_derivative[tmp_thread_global_index] = 0_r;
      }
    }
  }
}
