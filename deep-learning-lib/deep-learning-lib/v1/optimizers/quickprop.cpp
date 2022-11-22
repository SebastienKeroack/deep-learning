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
#include "deep-learning-lib/v1/optimizers/quickprop.hpp"

using namespace DL::Math;

namespace DL::v1 {
void update_model_quickprop(Model *model, size_t const batch_size,
                            size_t const str, size_t const end) {
  var *parameters(model->ptr_array_parameters);

  real const epsilon(model->learning_rate / static_cast<real>(batch_size)),
      decay(model->quickprop_decay), mu(model->quickprop_mu),
      shrink_factor(mu / (1_r + mu));

  real *derivatives(model->ptr_array_derivatives_parameters),
      *derivatives_tm1(model->ptr_array_previous_derivatives_parameters),
      *prev_steps(model->ptr_array_previous_steps);

  for (size_t i(str); i != end; ++i) {
    real const prev_step(prev_steps[i]), slope_tm1(derivatives_tm1[i]);
    real next_step(0_r);
    real const slope(derivatives[i] + decay * cast(parameters[i]));

    if (prev_step > 0.001_r) {
      if (slope > 0_r) next_step += epsilon * slope;

      if (slope > (shrink_factor * slope_tm1))
        next_step += mu * prev_step;
      else
        next_step += prev_step * slope / (slope_tm1 - slope);
    } else if (prev_step < -0.001_r) {
      if (slope < 0_r) next_step += epsilon * slope;

      if (slope < (shrink_factor * slope_tm1))
        next_step += mu * prev_step;
      else
        next_step += prev_step * slope / (slope_tm1 - slope);
    } else {
      next_step += epsilon * slope;
    }

    prev_steps[i] = next_step;
    parameters[i] = clip(cast(parameters[i]) + next_step, -1500_r, 1500_r);
    derivatives_tm1[i] = slope;
    derivatives[i] = 0.0;
  }
}
}  // namespace DL