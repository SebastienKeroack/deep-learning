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

#include "deep-learning/io/logger.hpp"
#include "deep-learning/v1/learner/model.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"

#include <array>

namespace DL::v1 {
void Model::Initialization__Identity(real const bias)
{
    Layer const *const last_layer(this->ptr_last_layer);
    Layer *layer_it(this->ptr_array_layers + 1);

    // Loop though each layer.
    for(; layer_it != last_layer; ++layer_it)
    {
        // If the current layer is a pooling/residual layer, continue.
        if(layer_it->type_layer == LAYER::AVERAGE_POOLING
          ||
          layer_it->type_layer == LAYER::MAX_POOLING
          ||
          layer_it->type_layer == LAYER::RESIDUAL) { continue; }
        
        switch(layer_it->type_layer)
        {
            case LAYER::FULLY_CONNECTED:
                this->weights_initialize_identity(*layer_it->ptr_array_neuron_units->ptr_number_connections,
                                                   static_cast<size_t>(layer_it->ptr_last_neuron_unit - layer_it->ptr_array_neuron_units),
                                                   this->ptr_array_parameters + *layer_it->ptr_first_connection_index);

                this->layer_initialize_const_bias(bias, layer_it);
                    break;
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->weights_initialize_identity(*layer_it->ptr_array_neuron_units->ptr_number_connections,
                                                   static_cast<size_t>(layer_it->ptr_last_neuron_unit - layer_it->ptr_array_neuron_units),
                                                   this->ptr_array_parameters + *layer_it->ptr_first_connection_index);

                this->indrec_initialize_uniform(layer_it);

                this->layer_initialize_const_bias(bias, layer_it);
                    break;
            case LAYER::LSTM:
                this->lstm_initialize_identity(layer_it);

                this->lstm_initialize_const_bias(bias, layer_it);
                    break;
            default:
                ERR(L"Can not initialize weights in the layer %zu with (%d | %ls) as the type layer.",
                                         static_cast<size_t>(layer_it - this->ptr_array_layers),
                                         layer_it->type_layer,
                                         LAYER_NAME[layer_it->type_layer].c_str());
                    break;
        }
    }

    // Independently recurrent neural network.
    if(this->seq_w > 1_UZ
      &&
      this->n_time_delay + 1_UZ == this->seq_w)
      this->indrec_initialize_uniform_ltm();

    if(this->ptr_array_derivatives_parameters != nullptr) { this->clear_training_arrays(); }

    if(this->Use__Normalization()) { this->Clear__Parameter__Normalized_Unit(); }

    this->_initialized__weight = true;
    this->_type_weights_initializer = INITIALIZER::IDENTITY;
}

[[deprecated("Not properly implemented.")]]
void Model::lstm_initialize_identity(Layer const *const layer_it) {
  // NotImplementedError.
  // ...
  // ...
}

void Model::weights_initialize_identity(size_t const rows,
                                                          size_t const cols,
                                                          var *const parameters)
{
    VARZERO(parameters, rows * cols * sizeof(var));

    size_t const smallest_length(std::min(rows, cols));
    size_t i;

    for(i = 0_UZ; i != smallest_length; ++i) { parameters[i * cols + i] = 1_r; }
}
}
