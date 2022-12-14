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

#include "pch.hpp"

#include "deep-learning/v1/learner/model.hpp"

#include <omp.h>

namespace DL::v1 {
void Model::Update_Parameter__Gradient_Descent(size_t const batch_size,
                                               size_t const training_size,
                                               size_t const str,
                                               size_t const end) {
  // TODO: Make Nesterov independent.
  if (this->use_mp && this->is_mp_initialized) {
    if (this->learning_momentum != 0_r) {
      if (this->use_nesterov) {
        this->Update_Parameter_Nesterov_Accelerated_Gradient__OpenMP(
            batch_size, training_size, str, end);
      } else {
        this->Update_Parameter__Gradient_Descent_Momentum__OpenMP(
            batch_size, training_size, str, end);
      }
    } else {
      this->Update_Parameter__Gradient_Descent__OpenMP(batch_size,
                                                       training_size, str, end);
    }
  } else {
    if (this->learning_momentum != 0_r) {
      if (this->use_nesterov) {
        this->Update_Parameter_Nesterov_Accelerated_Gradient__Loop(
            batch_size, training_size, str, end);
      } else {
        this->Update_Parameter__Gradient_Descent_Momentum__Loop(
            batch_size, training_size, str, end);
      }
    } else {
      this->Update_Parameter__Gradient_Descent__Loop(batch_size, training_size,
                                                     str, end);
    }
  }
}

void Model::Update_Parameter__Gradient_Descent__Loop(size_t const batch_size,
                                                     size_t const training_size,
                                                     size_t const str,
                                                     size_t const end) {
  size_t i;

  // var const lr_scale((this->use_warm_restarts ?
  // this->warm_restarts_decay() / this->learning_rate : 1_r) /
  // static_cast<var>(training_size));
  real const lr_scale(this->use_warm_restarts
                          ? this->warm_restarts_decay() / this->learning_rate
                          : 1_r);

  this->optimizer_time_step += 1_r;

  real const *const mask(this->ptr_array_mask_regularized_parameters),
      lr(lr_scale * this->learning_rate),
      wd(lr_scale * (this->use_normalized_weight_decay
                         ? this->normalized_wd(batch_size, training_size)
                         : this->weight_decay));
  real *const derivatives(this->ptr_array_derivatives_parameters);

  var *const parameters(this->ptr_array_parameters);

  if (wd != 0_r) {
    for (i = str; i != end; ++i) {
      parameters[i] -= lr * derivatives[i] + mask[i] * wd * parameters[i];

      derivatives[i] = 0_r;
    }
  } else {
    for (i = str; i != end; ++i) {
      parameters[i] -= lr * derivatives[i];

      derivatives[i] = 0_r;
    }
  }
}

void Model::Update_Parameter__Gradient_Descent__OpenMP(
    size_t const batch_size, size_t const training_size, size_t const str,
    size_t const end) {
  int const end_(static_cast<int>(end));

  real const lr_scale(this->use_warm_restarts
                          ? this->warm_restarts_decay() / this->learning_rate
                          : 1_r);

  this->optimizer_time_step += 1_r;

  real const *const mask(this->ptr_array_mask_regularized_parameters),
      lr(lr_scale * this->learning_rate),
      wd(lr_scale * (this->use_normalized_weight_decay
                         ? this->normalized_wd(batch_size, training_size)
                         : this->weight_decay));
  real *const derivatives(this->ptr_array_derivatives_parameters);

  var *const parameters(this->ptr_array_parameters);

  if (wd != 0_r) {
#pragma omp parallel for schedule(static)
    for (int i = static_cast<int>(str); i < end_; ++i) {
      parameters[i] -= lr * derivatives[i] + mask[i] * wd * parameters[i];

      derivatives[i] = 0_r;
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int i = static_cast<int>(str); i < end_; ++i) {
      parameters[i] -= lr * derivatives[i];

      derivatives[i] = 0_r;
    }
  }
}
}  // namespace DL
