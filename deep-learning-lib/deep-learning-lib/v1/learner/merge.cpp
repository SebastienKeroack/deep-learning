/* Copyright 2016, 2022 Sébastien Kéroack. All Rights Reserved.

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

// PCH:
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/v1/learner/model.hpp"

// Deep learning lib:
#include "deep-learning-lib/v1/mem/reallocate.hpp"

namespace DL::v1 {
void Model::merge_mp_accu_loss(void) {
  if (this->use_mp && this->is_mp_initialized && this->number_threads > 1_UZ) {
    size_t thread_id;

    if (this->ptr_array_number_loss != nullptr) {
      for (thread_id = 1_UZ; thread_id != this->number_threads; ++thread_id) {
        this->ptr_array_number_loss[0] +=
            this->ptr_array_number_loss[thread_id];
      }
    }

    if (this->ptr_array_loss_values != nullptr) {
      for (thread_id = 1_UZ; thread_id != this->number_threads; ++thread_id) {
        this->ptr_array_loss_values[0] +=
            this->ptr_array_loss_values[thread_id];
      }
    }

    if (this->type_loss_function == LOSS_FN::BIT &&
        this->ptr_array_number_bit_fail != nullptr) {
      for (thread_id = 1_UZ; thread_id != this->number_threads; ++thread_id) {
        this->ptr_array_number_bit_fail[0] +=
            this->ptr_array_number_bit_fail[thread_id];
      }
    }

    if (this->ptr_array_accuracy_values != nullptr) {
      for (thread_id = 1_UZ; thread_id != this->number_threads; ++thread_id) {
        this->ptr_array_accuracy_values[0][0] +=
            this->ptr_array_accuracy_values[0][thread_id];
        this->ptr_array_accuracy_values[1][0] +=
            this->ptr_array_accuracy_values[1][thread_id];
        this->ptr_array_accuracy_values[2][0] +=
            this->ptr_array_accuracy_values[2][thread_id];
        this->ptr_array_accuracy_values[3][0] +=
            this->ptr_array_accuracy_values[3][thread_id];
        this->ptr_array_accuracy_values[4][0] +=
            this->ptr_array_accuracy_values[4][thread_id];
      }
    }
  }
}

void Model::Merge__Accuracy__R(void) {
  if (this->use_mp && this->is_mp_initialized && this->number_threads > 1_UZ) {
    size_t thread_id;

    for (thread_id = 1_UZ; thread_id != this->number_threads; ++thread_id) {
      this->ptr_array_accuracy_values[2][0] +=
          this->ptr_array_accuracy_values[2][thread_id];
      this->ptr_array_accuracy_values[3][0] +=
          this->ptr_array_accuracy_values[3][thread_id];
      this->ptr_array_accuracy_values[4][0] +=
          this->ptr_array_accuracy_values[4][thread_id];
    }
  }
}

void Model::merge_mp_derivatives(size_t const begin, size_t const end) {
  if (!this->use_mp || !this->is_mp_initialized || this->number_threads <= 1_UZ)
    return;

  int const end_(static_cast<int>(end));
  int i;

  size_t thread_id;

  real const *derivatives_wrt_thread;
  real *const derivatives(this->ptr_array_derivatives_parameters);

  for (thread_id = 1_UZ; thread_id != this->number_threads; ++thread_id) {
    derivatives_wrt_thread = this->ptr_array_derivatives_parameters +
                             thread_id * this->total_parameters_allocated;

#pragma omp parallel for schedule(static)
    for (i = static_cast<int>(begin); i < end_; ++i)
      derivatives[i] += derivatives_wrt_thread[i];
  }

  // Reset derivatives except from the main thread.
  memset(derivatives + this->total_parameters_allocated, 0,
         (this->number_threads - 1_UZ) * this->total_parameters_allocated *
             sizeof(real));
}
}  // namespace DL::v1