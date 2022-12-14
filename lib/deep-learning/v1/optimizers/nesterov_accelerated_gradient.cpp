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
void Model::Update_Parameter_Nesterov_Accelerated_Gradient__Loop(
    size_t const batch_size, size_t const training_size, size_t const str,
    size_t const end) {
  size_t i;

  real const lr_scale(this->use_warm_restarts
                          ? this->warm_restarts_decay() / this->learning_rate
                          : 1_r);

  this->optimizer_time_step += 1_r;

  real const *const mask(this->ptr_array_mask_regularized_parameters),
      lr(lr_scale * this->learning_rate), lr_mom(this->learning_momentum),
      wd(lr_scale * (this->use_normalized_weight_decay
                         ? this->normalized_wd(batch_size, training_size)
                         : this->weight_decay));
  real *const derivatives(this->ptr_array_derivatives_parameters),
      *const delta_tm1(this->ptr_array_previous_delta_parameters), delta;

  var *const parameters(this->ptr_array_parameters);

  if (wd != 0_r) {
    for (i = str; i != end; ++i) {
      delta_tm1[i] = delta = lr * derivatives[i] + lr_mom * delta_tm1[i];

      parameters[i] -=
          lr_mom * delta - lr * derivatives[i] + mask[i] * wd * parameters[i];

      derivatives[i] = 0_r;
    }
  } else {
    for (i = str; i != end; ++i) {
      delta_tm1[i] = delta = lr * derivatives[i] + lr_mom * delta_tm1[i];

      parameters[i] -=
          lr_mom * delta - lr * derivatives[i];  // Gradient descent

      derivatives[i] = 0_r;
    }
  }
}

void Model::Update_Parameter_Nesterov_Accelerated_Gradient__OpenMP(
    size_t const batch_size, size_t const training_size, size_t const str,
    size_t const end) {
  int const end_(static_cast<int>(end));

  real const lr_scale(this->use_warm_restarts
                          ? this->warm_restarts_decay() / this->learning_rate
                          : 1_r);

  this->optimizer_time_step += 1_r;

  real const *const mask(this->ptr_array_mask_regularized_parameters),
      lr(lr_scale * this->learning_rate), lr_mom(this->learning_momentum),
      wd(lr_scale * (this->use_normalized_weight_decay
                         ? this->normalized_wd(batch_size, training_size)
                         : this->weight_decay));
  real *const derivatives(this->ptr_array_derivatives_parameters),
      *const delta_tm1(this->ptr_array_previous_delta_parameters), delta(0_r);

  var *const parameters(this->ptr_array_parameters);

  if (wd != 0_r) {
#pragma omp parallel for schedule(static) private(delta)
    for (int i = static_cast<int>(str); i < end_; ++i) {
      delta_tm1[i] = delta = lr * derivatives[i] + lr_mom * delta_tm1[i];

      parameters[i] -=
          lr_mom * delta - lr * derivatives[i] + mask[i] * wd * parameters[i];

      derivatives[i] = 0_r;
    }
  } else {
#pragma omp parallel for schedule(static) private(delta)
    for (int i = static_cast<int>(str); i < end_; ++i) {
      delta_tm1[i] = delta = lr * derivatives[i] + lr_mom * delta_tm1[i];

      parameters[i] -=
          lr_mom * delta - lr * derivatives[i];  // Gradient descent

      derivatives[i] = 0_r;
    }
  }
}
}  // namespace DL
