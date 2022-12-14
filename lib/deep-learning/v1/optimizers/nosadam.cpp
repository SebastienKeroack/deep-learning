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
void Model::Update_Parameters__NosAdam(size_t const batch_size,
                                       size_t const training_size,
                                       size_t const str, size_t const end) {
  if (this->use_mp && this->is_mp_initialized)
    this->Update_Parameters__NosAdam__OpenMP(batch_size, training_size, str,
                                             end);
  else
    this->Update_Parameters__NosAdam__Loop(batch_size, training_size, str, end);
}

void Model::Update_Parameters__NosAdam__Loop(size_t const batch_size,
                                             size_t const training_size,
                                             size_t const str,
                                             size_t const end) {
  size_t i;

  real const lr_scale(this->use_warm_restarts ? this->warm_restarts_decay() /
                                                    this->adam_learning_rate
                                              : 1_r);

  this->optimizer_time_step += 1_r;

  real const *const mask(this->ptr_array_mask_regularized_parameters),
      lr(lr_scale * this->adam_learning_rate),
      wd(this->use_normalized_weight_decay
             ? this->normalized_wd(batch_size, training_size)
             : this->weight_decay),
      beta1(this->adam_beta1), beta2(this->adam_beta2),
      beta2_at_power_gamma(std::pow(this->optimizer_time_step, -this->adam_gamma)),
      beta2_at_time_t(this->adam_previous_beta2 /
                      (this->adam_previous_beta2 + beta2_at_power_gamma)),
      epsilon(this->adam_epsilon),
      lr_t(this->use_adam_bias_correction
               ? lr * std::sqrt(1_r - std::pow(beta2, this->optimizer_time_step)) /
                     (1_r - std::pow(beta1, this->optimizer_time_step))
               : lr);
  real *const derivatives(this->ptr_array_derivatives_parameters),
      *const biased_first_moments_tm1(
          this->ptr_array_previous_biased_first_moment),
          *const biased_secod_moments_tm1(
              this->ptr_array_previous_biased_second_moment),
      derivative, biased_first_moment, biased_secod_moment;

  var *const parameters(this->ptr_array_parameters);

  this->adam_previous_beta2 += beta2_at_power_gamma;

  if (wd != 0_r) {
    for (i = str; i != end; ++i) {
      derivative = derivatives[i];

      biased_first_moments_tm1[i] = biased_first_moment =
          beta1 * biased_first_moments_tm1[i] + (1_r - beta1) * derivative;
      biased_secod_moments_tm1[i] = biased_secod_moment =
          beta2_at_time_t * biased_secod_moments_tm1[i] +
          (1_r - beta2_at_time_t) * derivative * derivative;

      parameters[i] -=
          lr_t * biased_first_moment / (std::sqrt(biased_secod_moment) + epsilon) +
          mask[i] * wd * parameters[i];

      derivatives[i] = 0_r;
    }
  } else {
    for (i = str; i != end; ++i) {
      derivative = derivatives[i];

      biased_first_moments_tm1[i] = biased_first_moment =
          beta1 * biased_first_moments_tm1[i] + (1_r - beta1) * derivative;
      biased_secod_moments_tm1[i] = biased_secod_moment =
          beta2_at_time_t * biased_secod_moments_tm1[i] +
          (1_r - beta2_at_time_t) * derivative * derivative;

      parameters[i] -=
          lr_t * biased_first_moment / (std::sqrt(biased_secod_moment) + epsilon);

      derivatives[i] = 0_r;
    }
  }
}

void Model::Update_Parameters__NosAdam__OpenMP(size_t const batch_size,
                                               size_t const training_size,
                                               size_t const str,
                                               size_t const end) {
  int const end_(static_cast<int>(end));

  real const lr_scale(this->use_warm_restarts ? this->warm_restarts_decay() /
                                                    this->adam_learning_rate
                                              : 1_r);

  this->optimizer_time_step += 1_r;

  real const *const mask(this->ptr_array_mask_regularized_parameters),
      lr(lr_scale * this->adam_learning_rate),
      wd(this->use_normalized_weight_decay
             ? this->normalized_wd(batch_size, training_size)
             : this->weight_decay),
      beta1(this->adam_beta1), beta2(this->adam_beta2),
      beta2_at_power_gamma(std::pow(this->optimizer_time_step, -this->adam_gamma)),
      beta2_at_time_t(this->adam_previous_beta2 /
                      (this->adam_previous_beta2 + beta2_at_power_gamma)),
      epsilon(this->adam_epsilon),
      lr_t(this->use_adam_bias_correction
               ? lr * std::sqrt(1_r - std::pow(beta2, this->optimizer_time_step)) /
                     (1_r - std::pow(beta1, this->optimizer_time_step))
               : lr);
  real *const derivatives(this->ptr_array_derivatives_parameters),
      *const biased_first_moments_tm1(
          this->ptr_array_previous_biased_first_moment),
          *const biased_secod_moments_tm1(
              this->ptr_array_previous_biased_second_moment),
      derivative(0_r), biased_first_moment(0_r), biased_secod_moment(0_r);

  var *const parameters(this->ptr_array_parameters);

  this->adam_previous_beta2 += beta2_at_power_gamma;

  if (wd != 0_r) {
#pragma omp parallel for schedule(static) private( \
    derivative, biased_first_moment, biased_secod_moment)
    for (int i = static_cast<int>(str); i < end_; ++i) {
      derivative = derivatives[i];

      biased_first_moments_tm1[i] = biased_first_moment =
          beta1 * biased_first_moments_tm1[i] + (1_r - beta1) * derivative;
      biased_secod_moments_tm1[i] = biased_secod_moment =
          beta2_at_time_t * biased_secod_moments_tm1[i] +
          (1_r - beta2_at_time_t) * derivative * derivative;

      parameters[i] -=
          lr_t * biased_first_moment / (std::sqrt(biased_secod_moment) + epsilon) +
          mask[i] * wd * parameters[i];

      derivatives[i] = 0_r;
    }
  } else {
#pragma omp parallel for schedule(static) private( \
    derivative, biased_first_moment, biased_secod_moment)
    for (int i = static_cast<int>(str); i < end_; ++i) {
      derivative = derivatives[i];

      biased_first_moments_tm1[i] = biased_first_moment =
          beta1 * biased_first_moments_tm1[i] + (1_r - beta1) * derivative;
      biased_secod_moments_tm1[i] = biased_secod_moment =
          beta2_at_time_t * biased_secod_moments_tm1[i] +
          (1_r - beta2_at_time_t) * derivative * derivative;

      parameters[i] -=
          lr_t * biased_first_moment / (std::sqrt(biased_secod_moment) + epsilon);

      derivatives[i] = 0_r;
    }
  }
}
}  // namespace DL
