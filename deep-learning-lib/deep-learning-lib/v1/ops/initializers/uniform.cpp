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

#include <array>

namespace DL::v1 {
void Model::Initialization__Uniform(real const bias, real const lower_bound,
                                    real const upper_bound) {
  Layer const *const last_layer(this->ptr_last_layer);
  Layer *layer_it(this->ptr_array_layers + 1);

  // Loop though each layer.
  for (; layer_it != last_layer; ++layer_it) {
    // If the current layer is a pooling/residual layer, continue.
    if (layer_it->type_layer == LAYER::AVERAGE_POOLING ||
        layer_it->type_layer == LAYER::MAX_POOLING ||
        layer_it->type_layer == LAYER::RESIDUAL)
      continue;

    switch (layer_it->type_layer) {
      case LAYER::FULLY_CONNECTED:
        this->weights_initialize_uniform(
            this->ptr_array_parameters + *layer_it->ptr_first_connection_index,
            this->ptr_array_parameters + *layer_it->ptr_last_connection_index,
            lower_bound, upper_bound);

        this->layer_initialize_const_bias(bias, layer_it);
        break;
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        this->weights_initialize_uniform(
            this->ptr_array_parameters + *layer_it->ptr_first_connection_index,
            this->ptr_array_parameters + *layer_it->ptr_last_connection_index,
            lower_bound, upper_bound);

        this->indrec_initialize_uniform(layer_it);

        this->layer_initialize_const_bias(bias, layer_it);
        break;
      case LAYER::LSTM:
        this->lstm_initialize_uniform(
            std::array<real, 5_UZ>{lower_bound, lower_bound, lower_bound,
                                   lower_bound, lower_bound}
                .data(),
            std::array<real, 5_UZ>{upper_bound, upper_bound, upper_bound,
                                   upper_bound, upper_bound}
                .data(),
            layer_it);

        this->lstm_initialize_const_bias(bias, layer_it);
        break;
      default:
        ERR(L"Can not initialize weights in the layer %zu with (%d | %ls) as "
            L"the type layer.",
            static_cast<size_t>(layer_it - this->ptr_array_layers),
            layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
        break;
    }
  }

  // Independently recurrent neural network.
  if (this->seq_w > 1_UZ && this->n_time_delay + 1_UZ == this->seq_w)
    this->indrec_initialize_uniform_ltm();

  if (this->ptr_array_derivatives_parameters != nullptr)
    this->clear_training_arrays();

  if (this->Use__Normalization())
    this->Clear__Parameter__Normalized_Unit();

  this->_initialized__weight = true;
  this->_type_weights_initializer = INITIALIZER::UNIFORM;
}
}  // namespace DL