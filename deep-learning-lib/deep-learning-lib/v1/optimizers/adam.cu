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
__global__ void kernel__Update_Parameter__Adam(
    T const beta1_received, T const beta2_received, T const epsilon_received,
    T const learning_rate_at_time_t_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_biased_first_moment_received,
    T *const ptr_array_previous_biased_second_moment_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_partial_derivative(
      ptr_array_partial_derivative_received[tmp_thread_global_index]),
      tmp_biased_first_moment, tmp_biased_second_moment;

  ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] =
      tmp_biased_first_moment =
          beta1_received * ptr_array_previous_biased_first_moment_received
                               [tmp_thread_global_index] +
          (T(1) - beta1_received) * tmp_partial_derivative;
  ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] =
      tmp_biased_second_moment =
          beta2_received * ptr_array_previous_biased_second_moment_received
                               [tmp_thread_global_index] +
          (T(1) - beta2_received) * tmp_partial_derivative *
              tmp_partial_derivative;

  ptr_array_parameters_received[tmp_thread_global_index] -=
      learning_rate_at_time_t_received * tmp_biased_first_moment /
      (sqrt(tmp_biased_second_moment) + epsilon_received);

  ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template <typename T>
__global__ void kernel__Update_Parameter__Adam(
    size_t const size_received, T const beta1_received, T const beta2_received,
    T const epsilon_received, T const learning_rate_at_time_t_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_biased_first_moment_received,
    T *const ptr_array_previous_biased_second_moment_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_partial_derivative, tmp_biased_first_moment, tmp_biased_second_moment;

  if (tmp_thread_global_index < size_received) {
    tmp_partial_derivative =
        ptr_array_partial_derivative_received[tmp_thread_global_index];

    ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] =
        tmp_biased_first_moment =
            beta1_received * ptr_array_previous_biased_first_moment_received
                                 [tmp_thread_global_index] +
            (T(1) - beta1_received) * tmp_partial_derivative;
    ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] =
        tmp_biased_second_moment =
            beta2_received * ptr_array_previous_biased_second_moment_received
                                 [tmp_thread_global_index] +
            (T(1) - beta2_received) * tmp_partial_derivative *
                tmp_partial_derivative;

    ptr_array_parameters_received[tmp_thread_global_index] -=
        learning_rate_at_time_t_received * tmp_biased_first_moment /
        (sqrt(tmp_biased_second_moment) + epsilon_received);

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
  }
}

template <typename T>
__global__ void kernel_while__Update_Parameter__Adam(
    size_t const size_received, T const beta1_received, T const beta2_received,
    T const epsilon_received, T const learning_rate_at_time_t_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_biased_first_moment_received,
    T *const ptr_array_previous_biased_second_moment_received) {
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_partial_derivative, tmp_biased_first_moment, tmp_biased_second_moment;

  do {
    tmp_partial_derivative =
        ptr_array_partial_derivative_received[tmp_thread_global_index];

    ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] =
        tmp_biased_first_moment =
            beta1_received * ptr_array_previous_biased_first_moment_received
                                 [tmp_thread_global_index] +
            (T(1) - beta1_received) * tmp_partial_derivative;
    ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] =
        tmp_biased_second_moment =
            beta2_received * ptr_array_previous_biased_second_moment_received
                                 [tmp_thread_global_index] +
            (T(1) - beta2_received) * tmp_partial_derivative *
                tmp_partial_derivative;

    ptr_array_parameters_received[tmp_thread_global_index] -=
        learning_rate_at_time_t_received * tmp_biased_first_moment /
        (sqrt(tmp_biased_second_moment) + epsilon_received);

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

    tmp_thread_global_index += gridDim.x * blockDim.x;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__global__ void kernel__Update_Parameter__Adam(
    T const weight_decay_received, T const beta1_received,
    T const beta2_received, T const epsilon_received,
    T const learning_rate_at_time_t_received,
    T const *const ptr_array_connections_mask_rergularization_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_biased_first_moment_received,
    T *const ptr_array_previous_biased_second_moment_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_partial_derivative(
      ptr_array_partial_derivative_received[tmp_thread_global_index]),
      tmp_biased_first_moment, tmp_biased_second_moment;

  ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] =
      tmp_biased_first_moment =
          beta1_received * ptr_array_previous_biased_first_moment_received
                               [tmp_thread_global_index] +
          (T(1) - beta1_received) * tmp_partial_derivative;
  ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] =
      tmp_biased_second_moment =
          beta2_received * ptr_array_previous_biased_second_moment_received
                               [tmp_thread_global_index] +
          (T(1) - beta2_received) * tmp_partial_derivative *
              tmp_partial_derivative;

  ptr_array_parameters_received[tmp_thread_global_index] -=
      learning_rate_at_time_t_received * tmp_biased_first_moment /
          (sqrt(tmp_biased_second_moment) + epsilon_received) +
      ptr_array_connections_mask_rergularization_received
              [tmp_thread_global_index] *
          weight_decay_received *
          ptr_array_parameters_received[tmp_thread_global_index];

  ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
}

template <typename T>
__global__ void kernel__Update_Parameter__Adam(
    size_t const size_received, T const weight_decay_received,
    T const beta1_received, T const beta2_received, T const epsilon_received,
    T const learning_rate_at_time_t_received,
    T const *const ptr_array_connections_mask_rergularization_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_biased_first_moment_received,
    T *const ptr_array_previous_biased_second_moment_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_partial_derivative, tmp_biased_first_moment, tmp_biased_second_moment;

  if (tmp_thread_global_index < size_received) {
    tmp_partial_derivative =
        ptr_array_partial_derivative_received[tmp_thread_global_index];

    ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] =
        tmp_biased_first_moment =
            beta1_received * ptr_array_previous_biased_first_moment_received
                                 [tmp_thread_global_index] +
            (T(1) - beta1_received) * tmp_partial_derivative;
    ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] =
        tmp_biased_second_moment =
            beta2_received * ptr_array_previous_biased_second_moment_received
                                 [tmp_thread_global_index] +
            (T(1) - beta2_received) * tmp_partial_derivative *
                tmp_partial_derivative;

    ptr_array_parameters_received[tmp_thread_global_index] -=
        learning_rate_at_time_t_received * tmp_biased_first_moment /
            (sqrt(tmp_biased_second_moment) + epsilon_received) +
        ptr_array_connections_mask_rergularization_received
                [tmp_thread_global_index] *
            weight_decay_received *
            ptr_array_parameters_received[tmp_thread_global_index];

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);
  }
}

template <typename T>
__global__ void kernel_while__Update_Parameter__Adam(
    size_t const size_received, T const weight_decay_received,
    T const beta1_received, T const beta2_received, T const epsilon_received,
    T const learning_rate_at_time_t_received,
    T const *const ptr_array_connections_mask_rergularization_received,
    T *const ptr_array_partial_derivative_received,
    T *const ptr_array_parameters_received,
    T *const ptr_array_previous_biased_first_moment_received,
    T *const ptr_array_previous_biased_second_moment_received) {
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_partial_derivative, tmp_biased_first_moment, tmp_biased_second_moment;

  do {
    tmp_partial_derivative =
        ptr_array_partial_derivative_received[tmp_thread_global_index];

    ptr_array_previous_biased_first_moment_received[tmp_thread_global_index] =
        tmp_biased_first_moment =
            beta1_received * ptr_array_previous_biased_first_moment_received
                                 [tmp_thread_global_index] +
            (T(1) - beta1_received) * tmp_partial_derivative;
    ptr_array_previous_biased_second_moment_received[tmp_thread_global_index] =
        tmp_biased_second_moment =
            beta2_received * ptr_array_previous_biased_second_moment_received
                                 [tmp_thread_global_index] +
            (T(1) - beta2_received) * tmp_partial_derivative *
                tmp_partial_derivative;

    ptr_array_parameters_received[tmp_thread_global_index] -=
        learning_rate_at_time_t_received * tmp_biased_first_moment /
            (sqrt(tmp_biased_second_moment) + epsilon_received) +
        ptr_array_connections_mask_rergularization_received
                [tmp_thread_global_index] *
            weight_decay_received *
            ptr_array_parameters_received[tmp_thread_global_index];

    ptr_array_partial_derivative_received[tmp_thread_global_index] = T(0);

    tmp_thread_global_index += gridDim.x * blockDim.x;
  } while (tmp_thread_global_index < size_received);
}

__device__ void cuModel::Update_Parameter__Adam(
    size_t const batch_size, size_t const training_size,
    size_t const start_index_received, size_t const end_index_received) {
  size_t i;

  var const lr_scale(this->use_warm_restarts
                                       ? this->warm_restarts_decay() /
                                             this->adam_learning_rate
                                       : 1_r);

  ++this->optimizer_time_step;

  var const *const mask(
      this->ptr_array_mask_regularized_parameters),
      lr(lr_scale * this->adam_learning_rate),
      wd(this->use_normalized_weight_decay
                           ? this->normalized_wd(
                                 batch_size, training_size)
                           : this->weight_decay),
      tmp_beta1(this->adam_beta1), tmp_beta2(this->adam_beta2),
      tmp_epsilon(this->adam_epsilon),
      tmp_adam_epochs(this->optimizer_time_step),
      tmp_learning_rate_at_time_t(lr *
                                  sqrt(1_r - pow(tmp_beta2, tmp_adam_epochs)) /
                                  (1_r - pow(tmp_beta1, tmp_adam_epochs)));
  var *const tmp_ptr_array_partial_derivative(
      this->ptr_array_derivatives_parameters),
      *const tmp_ptr_array_parameters(this->ptr_array_parameters),
          *const tmp_ptr_array_previous_biased_first_moment(
              this->ptr_array_previous_biased_first_moment),
              *const tmp_ptr_array_previous_biased_second_moment(
                  this->ptr_array_previous_biased_second_moment),
      tmp_partial_derivative, tmp_biased_first_moment, tmp_biased_second_moment;

  if (wd != 0_r) {
    if (USE_PARALLEL && end_index_received - start_index_received >= warpSize) {
      // KERNEL LAUNCH
      //    1: Launching do-while elements.
      if (this->ptr_array_dim3_grid[1].x * this->ptr_array_dim3_block[1].x <
          end_index_received - start_index_received) {
        kernel_while__Update_Parameter__Adam<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                end_index_received - start_index_received, wd,
                tmp_beta1, tmp_beta2, tmp_epsilon, tmp_learning_rate_at_time_t,
                mask +
                    start_index_received,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_biased_first_moment +
                    start_index_received,
                tmp_ptr_array_previous_biased_second_moment +
                    start_index_received);
      }
      //    2: Launching size condition.
      else if (this->ptr_array_dim3_grid[1].x *
                   this->ptr_array_dim3_block[1].x >
               end_index_received - start_index_received) {
        kernel__Update_Parameter__Adam<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                end_index_received - start_index_received, wd,
                tmp_beta1, tmp_beta2, tmp_epsilon, tmp_learning_rate_at_time_t,
                mask +
                    start_index_received,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_biased_first_moment +
                    start_index_received,
                tmp_ptr_array_previous_biased_second_moment +
                    start_index_received);
      }
      //    3: Standard.
      else {
        kernel__Update_Parameter__Adam<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                wd, tmp_beta1, tmp_beta2, tmp_epsilon,
                tmp_learning_rate_at_time_t,
                mask +
                    start_index_received,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_biased_first_moment +
                    start_index_received,
                tmp_ptr_array_previous_biased_second_moment +
                    start_index_received);
      }
      // |END| KERNEL LAUNCH |END|

      CUDA__Check_Error();
    } else {
      for (i = start_index_received; i != end_index_received; ++i) {
        tmp_partial_derivative = tmp_ptr_array_partial_derivative[i];

        tmp_ptr_array_previous_biased_first_moment[i] =
            tmp_biased_first_moment =
                tmp_beta1 * tmp_ptr_array_previous_biased_first_moment[i] +
                (1_r - tmp_beta1) * tmp_partial_derivative;
        tmp_ptr_array_previous_biased_second_moment[i] =
            tmp_biased_second_moment =
                tmp_beta2 * tmp_ptr_array_previous_biased_second_moment[i] +
                (1_r - tmp_beta2) * tmp_partial_derivative *
                    tmp_partial_derivative;

        tmp_ptr_array_parameters[i] -=
            tmp_learning_rate_at_time_t * tmp_biased_first_moment /
                (sqrt(tmp_biased_second_moment) + tmp_epsilon) +
            mask[i] *
                wd * tmp_ptr_array_parameters[i];

        tmp_ptr_array_partial_derivative[i] = 0_r;
      }
    }
  } else {
    if (USE_PARALLEL && end_index_received - start_index_received >= warpSize) {
      // KERNEL LAUNCH
      //    1: Launching do-while elements.
      if (this->ptr_array_dim3_grid[1].x * this->ptr_array_dim3_block[1].x <
          end_index_received - start_index_received) {
        kernel_while__Update_Parameter__Adam<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                end_index_received - start_index_received, tmp_beta1, tmp_beta2,
                tmp_epsilon, tmp_learning_rate_at_time_t,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_biased_first_moment +
                    start_index_received,
                tmp_ptr_array_previous_biased_second_moment +
                    start_index_received);
      }
      //    2: Launching size condition.
      else if (this->ptr_array_dim3_grid[1].x *
                   this->ptr_array_dim3_block[1].x >
               end_index_received - start_index_received) {
        kernel__Update_Parameter__Adam<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                end_index_received - start_index_received, tmp_beta1, tmp_beta2,
                tmp_epsilon, tmp_learning_rate_at_time_t,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_biased_first_moment +
                    start_index_received,
                tmp_ptr_array_previous_biased_second_moment +
                    start_index_received);
      }
      //    3: Standard.
      else {
        kernel__Update_Parameter__Adam<var>
            <<<this->ptr_array_dim3_grid[1], this->ptr_array_dim3_block[1]>>>(
                tmp_beta1, tmp_beta2, tmp_epsilon, tmp_learning_rate_at_time_t,
                tmp_ptr_array_partial_derivative + start_index_received,
                tmp_ptr_array_parameters + start_index_received,
                tmp_ptr_array_previous_biased_first_moment +
                    start_index_received,
                tmp_ptr_array_previous_biased_second_moment +
                    start_index_received);
      }
      // |END| KERNEL LAUNCH |END|

      CUDA__Check_Error();
    } else {
      for (i = start_index_received; i != end_index_received; ++i) {
        tmp_partial_derivative = tmp_ptr_array_partial_derivative[i];

        tmp_ptr_array_previous_biased_first_moment[i] =
            tmp_biased_first_moment =
                tmp_beta1 * tmp_ptr_array_previous_biased_first_moment[i] +
                (1_r - tmp_beta1) * tmp_partial_derivative;
        tmp_ptr_array_previous_biased_second_moment[i] =
            tmp_biased_second_moment =
                tmp_beta2 * tmp_ptr_array_previous_biased_second_moment[i] +
                (1_r - tmp_beta2) * tmp_partial_derivative *
                    tmp_partial_derivative;

        tmp_ptr_array_parameters[i] -=
            tmp_learning_rate_at_time_t * tmp_biased_first_moment /
            (sqrt(tmp_biased_second_moment) + tmp_epsilon);

        tmp_ptr_array_partial_derivative[i] = 0_r;
      }
    }
  }
}
