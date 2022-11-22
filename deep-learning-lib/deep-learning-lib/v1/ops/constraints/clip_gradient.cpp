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
#include "deep-learning-lib/data/string.hpp"
#include "deep-learning-lib/data/time.hpp"
#include "deep-learning-lib/io/logger.hpp"

#include <omp.h>

namespace DL::v1 {
void Model::set_clip_gradient(bool const use_clip_gradient) {
  this->use_clip_gradient = use_clip_gradient;
}

bool Model::set_clip_gradient(real const clip_gradient) {
  if (this->clip_gradient == clip_gradient) return true;

  bool *layers_use_default(new bool[this->total_layers - 2_UZ]);
  memset(layers_use_default, 0, (this->total_layers - 2_UZ) * sizeof(bool));

  size_t const n_layers(this->total_layers - 2_UZ);
  size_t k;

  Layer *layer_it;

  for (k = 0_UZ; k != n_layers; ++k) {
    layer_it = this->ptr_array_layers + k + 1;

    // Regularization on recurrent connection(s) (Independently RNN).
    layers_use_default[k] =
        layer_it->type_layer ==
            LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT &&
        this->Check__Use__Regularization__Constraint_Recurrent_Weight__Default(
            layer_it);
  }

  this->clip_gradient = clip_gradient;

  for (k = 0_UZ; k != n_layers; ++k) {
    layer_it = this->ptr_array_layers + k + 1;

    // Regularization on recurrent connection(s) (Independently RNN).
    if (layers_use_default[k] &&
        this->Set__Regularization__Constraint_Recurrent_Weight__Default(
            layer_it) == false) {
      ERR(L"An error has been triggered from the "
          "\"Set__Regularization__Constraint_Recurrent_Weight__Default(ptr)\" "
          "function.", );
      return false;
    }
  }

  return true;
}

void Model::Clip_Gradient__Loop(size_t const str, size_t const end) {
  euclidean_norm_st(str, end, this->clip_gradient,
                    this->ptr_array_derivatives_parameters);
}

void Model::Clip_Gradient__OpenMP(size_t const str, size_t const end) {
  euclidean_norm_mp(str, end, this->clip_gradient,
                    this->ptr_array_derivatives_parameters);
}
}  // namespace DL