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
#include "deep-learning/ops/math.hpp"

#include <omp.h>

using namespace DL::Math;

namespace DL::v1 {
void Model::Update_Parameter__iRPROP_plus(size_t const str, size_t const end) {
  if (this->use_mp && this->is_mp_initialized) {
    this->Update_Parameter__iRPROP_plus__OpenMP(str, end);
  } else {
    this->Update_Parameter__iRPROP_plus__Loop(str, end);
  }
}

void Model::Update_Parameter__iRPROP_plus__Loop(size_t const str,
                                                size_t const end) {
  bool const error_is_worst(this->loss_rprop > this->loss_rprop_tm1);

  real const incr_factor(this->rprop_increase_factor),
      decr_factor(this->rprop_decrease_factor),
      delta_min(this->rprop_delta_min), delta_max(this->rprop_delta_max);
  real *const derivatives(this->ptr_array_derivatives_parameters),
      *const derivatives_tm1(this->ptr_array_previous_derivatives_parameters),
          *const delta_w_tm1(this->ptr_array_previous_delta_parameters),
              *const prev_steps(this->ptr_array_previous_steps), derivative,
      delta_w, delta_t;

  var *const parameters(this->ptr_array_parameters);

  for (size_t i(str); i != end; ++i) {
    // Gradient ascent
    // derivative = -derivatives[i];
    // Gradient descent
    derivative = derivatives[i];

    if (derivatives_tm1[i] * derivative > 0_r) {
      delta_t = prev_steps[i] * incr_factor;
      prev_steps[i] = delta_t = std::min(delta_t, delta_max);

      delta_w_tm1[i] = delta_w = -sign(derivative) * delta_t;

      parameters[i] += delta_w;

      derivatives_tm1[i] = derivative;
    } else if (derivatives_tm1[i] * derivative < 0_r) {
      delta_t = prev_steps[i] * decr_factor;
      prev_steps[i] = std::max(delta_t, delta_min);

      if (error_is_worst) parameters[i] -= delta_w_tm1[i];

      derivatives_tm1[i] = 0_r;
    } else  // if(derivatives_tm1[i] *
            // derivative == 0_r)
    {
      delta_w_tm1[i] = delta_w = -sign(derivative) * prev_steps[i];

      parameters[i] += delta_w;

      derivatives_tm1[i] = derivative;
    }

    derivatives[i] = 0_r;
  }
}

void Model::Update_Parameter__iRPROP_plus__OpenMP(size_t const str,
                                                  size_t const end) {
  bool const error_is_worst(this->loss_rprop > this->loss_rprop_tm1);

  int const end_(static_cast<int>(end));

  real const incr_factor(this->rprop_increase_factor),
      decr_factor(this->rprop_decrease_factor),
      delta_min(this->rprop_delta_min), delta_max(this->rprop_delta_max);
  real *const derivatives(this->ptr_array_derivatives_parameters),
      *const derivatives_tm1(this->ptr_array_previous_derivatives_parameters),
          *const delta_w_tm1(this->ptr_array_previous_delta_parameters),
              *const prev_steps(this->ptr_array_previous_steps),
      derivative(0_r), delta_w(0_r), delta_t(0_r);

  var *const parameters(this->ptr_array_parameters);

#pragma omp parallel for schedule(static) private(derivative, delta_w, delta_t)
  for (int i = static_cast<int>(str); i < end_; ++i) {
    // Gradient ascent
    // derivative = -derivatives[i];
    // Gradient descent
    derivative = derivatives[i];

    if (derivatives_tm1[i] * derivative > 0_r) {
      delta_t = prev_steps[i] * incr_factor;
      prev_steps[i] = delta_t = std::min(delta_t, delta_max);

      delta_w_tm1[i] = delta_w = -sign(derivative) * delta_t;

      parameters[i] += delta_w;

      derivatives_tm1[i] = derivative;
    } else if (derivatives_tm1[i] * derivative < 0_r) {
      delta_t = prev_steps[i] * decr_factor;
      prev_steps[i] = std::max(delta_t, delta_min);

      if (error_is_worst) parameters[i] -= delta_w_tm1[i];

      derivatives_tm1[i] = 0_r;
    } else  // if(derivatives_tm1[i] *
            // derivative == 0_r)
    {
      delta_w_tm1[i] = delta_w = -sign(derivative) * prev_steps[i];

      parameters[i] += delta_w;

      derivatives_tm1[i] = derivative;
    }

    derivatives[i] = 0_r;
  }
}
}  // namespace DL
