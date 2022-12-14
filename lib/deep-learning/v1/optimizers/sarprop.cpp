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

#include "deep-learning/ops/distributions/real.hpp"
#include "deep-learning/v1/learner/model.hpp"
#include "deep-learning/v1/optimizers/sarprop.hpp"

namespace DL::v1 {
void update_model_sarprop(Model *model, size_t const epoch, size_t const str,
                          size_t const end) {
  var *parameters(model->ptr_array_parameters);

  real const incr_factor(model->rprop_increase_factor),
      decr_factor(model->rprop_decrease_factor),
      delta_min(model->rprop_delta_min), delta_max(model->rprop_delta_max),
      weight_decay_shift(model->sarprop_weight_decay_shift),
      step_error_threshold_factor(model->sarprop_step_error_threshold_factor),
      step_error_shift(model->sarprop_step_error_shift),
      temperature(model->sarprop_temperature), epoch_(static_cast<real>(epoch)),
      mse(static_cast<real>(model->get_loss(ENV::NONE))), rmse(std::sqrt(mse));

  real *prev_steps(model->ptr_array_previous_steps),
      *derivatives(model->ptr_array_derivatives_parameters),
      *derivatives_tm1(model->ptr_array_previous_derivatives_parameters);

  Dist::Real gen(0_r, 1_r);

  for (size_t i(str); i != end; ++i) {
    real slope(-derivatives[i] -
               cast(parameters[i]) *
                   std::exp2(-temperature * epoch_ + weight_decay_shift)),
        next_step(0_r);
    real const prev_step(std::max(prev_steps[i], 0.000001_r)),
        same_sign(derivatives_tm1[i] * slope);

    if (same_sign > 0_r) {
      next_step = std::min(prev_step * incr_factor, delta_max);

      if (slope < 0_r) {
        parameters[i] += next_step;
      } else {
        parameters[i] -= next_step;
      }
    } else if (same_sign < 0_r) {
      if (prev_step < step_error_threshold_factor * mse) {
        next_step =
            prev_step * decr_factor +
            gen() * rmse * std::exp2(-temperature * epoch_ + step_error_shift);
      } else {
        next_step = std::max(prev_step * decr_factor, delta_min);
      }

      slope = 0_r;
    } else {
      if (slope < 0_r) {
        parameters[i] += prev_step;
      } else {
        parameters[i] -= prev_step;
      }
    }

    prev_steps[i] = next_step;
    derivatives_tm1[i] = slope;
    derivatives[i] = 0_r;
  }

  ++model->sarprop_epoch;
}
}  // namespace DL