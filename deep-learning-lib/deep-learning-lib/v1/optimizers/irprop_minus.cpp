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

#include "deep-learning-lib/pch.hpp"

#include "deep-learning-lib/v1/learner/model.hpp"
#include "deep-learning-lib/ops/math.hpp"

using namespace DL::Math;

namespace DL::v1 {
void Model::Update_Parameter__iRPROP_minus__Loop(size_t const str,
                                                 size_t const end) {
  real const incr_factor(this->rprop_increase_factor),
      decr_factor(this->rprop_decrease_factor),
      delta_min(this->rprop_delta_min), delta_max(this->rprop_delta_max);

  real *const derivatives(this->ptr_array_derivatives_parameters),
      *const derivatives_tm1(this->ptr_array_previous_derivatives_parameters),
          *const prev_steps(this->ptr_array_previous_steps), derivative, delta;

  var *const parameters(this->ptr_array_parameters);

  for (size_t i(str); i != end; ++i) {
    // Gradient ascent
    // derivative = -derivatives[i];
    // Gradient descent
    derivative = derivatives[i];

    if (derivatives_tm1[i] * derivative > 0_r) {
      delta = prev_steps[i] * incr_factor;
      prev_steps[i] = delta = std::min(delta, delta_max);
    } else if (derivatives_tm1[i] * derivative < 0_r) {
      delta = prev_steps[i] * decr_factor;
      prev_steps[i] = delta = std::max(delta, delta_min);

      derivative = 0_r;
    } else {
      delta = prev_steps[i];
    }

    parameters[i] += -sign(derivative) * delta;

    derivatives_tm1[i] = derivative;
    derivatives[i] = 0_r;
  }
}
}  // namespace DL
