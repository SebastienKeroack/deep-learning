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
#include "deep-learning-lib/v1/mem/reallocate.hpp"

#include <omp.h>

namespace DL::v1 {
void Model::Tied__Transpose__Weight(Layer *const ptr_layer_received)
{
    Layer *const tmp_ptr_mirror_layer_it(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1),
                      *next_layer_it(const_cast<Layer *>(tmp_ptr_mirror_layer_it->next_connected_layers[0]));
    Layer const *const next_layer_end(tmp_ptr_mirror_layer_it->next_connected_layers[0] + tmp_ptr_mirror_layer_it->next_connected_layers.size());

    // Recurrent tied weights.
    if(ptr_layer_received != tmp_ptr_mirror_layer_it)
    {
        switch(ptr_layer_received->type_layer)
        {
            case LAYER::FULLY_CONNECTED: break;
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Tied__Transpose__Weight__FC_Ind_RNN(ptr_layer_received, tmp_ptr_mirror_layer_it); break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         ptr_layer_received->type_layer,
                                         LAYER_NAME[ptr_layer_received->type_layer].c_str());
                    break;
        }
    }

    // Forward tied weights.
    switch(ptr_layer_received->type_layer)
    {
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case LAYER::FULLY_CONNECTED_RECURRENT:
            for(; next_layer_it != next_layer_end; ++next_layer_it)
            {
                switch(next_layer_it->type_layer)
                {
                    case LAYER::FULLY_CONNECTED:
                    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case LAYER::FULLY_CONNECTED_RECURRENT: this->Tied__Transpose__Weight__FC(ptr_layer_received, next_layer_it); break;
                    default:
                        ERR(L"Layer type (%d | %ls) is not managed in",
                                                 next_layer_it->type_layer,
                                                 LAYER_NAME[next_layer_it->type_layer].c_str());
                            return;
                }
            }
        default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                                     ptr_layer_received->type_layer,
                                     LAYER_NAME[ptr_layer_received->type_layer].c_str());
                return;
    }
}

void Transpose(size_t const source_stride_received,
                      size_t const destination_stride_received,
                      var const *ptr_array_source_values_received,
                      var const *const ptr_array_source_value_end_received,
                      var *ptr_array_destination_values_received)
{
    size_t tmp_index;

    for(; ptr_array_source_values_received != ptr_array_source_value_end_received; ptr_array_source_values_received += source_stride_received,
                                                                                                                        ++ptr_array_destination_values_received)
    {
        for(tmp_index = 0_UZ; tmp_index != source_stride_received; ++tmp_index) { ptr_array_destination_values_received[tmp_index * destination_stride_received] = ptr_array_source_values_received[tmp_index]; }
    }
}

void Model::Tied__Transpose__Weight__FC(Layer const *const ptr_coded_layer_it_received, Layer const *const ptr_mirror_layer_it_received)
{
    Neuron_unit const *const tmp_ptr_coded_layer_it_first_neuron(ptr_coded_layer_it_received->ptr_array_neuron_units),
                                         *const tmp_ptr_mirror_layer_it_first_neuron(ptr_mirror_layer_it_received->ptr_array_neuron_units);
    
    Transpose(*tmp_ptr_coded_layer_it_first_neuron->ptr_number_connections,
                    *tmp_ptr_mirror_layer_it_first_neuron->ptr_number_connections,
                    this->ptr_array_parameters + *tmp_ptr_coded_layer_it_first_neuron->ptr_first_connection_index,
                    this->ptr_array_parameters + *tmp_ptr_coded_layer_it_first_neuron->ptr_last_connection_index,
                    this->ptr_array_parameters + *tmp_ptr_mirror_layer_it_first_neuron->ptr_first_connection_index);
}

void Model::Tied__Transpose__Weight__FC_Ind_RNN(Layer const *const ptr_encoded_layer_it_received, Layer const *const ptr_mirror_layer_it_received)
{
    AF_Ind_recurrent_unit const *const tmp_ptr_encoded_layer_it_first_AF_ind(ptr_encoded_layer_it_received->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_number_units(static_cast<size_t>(ptr_encoded_layer_it_received->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_encoded_layer_it_first_AF_ind));
    
    VARCOPY(this->ptr_array_parameters + *ptr_mirror_layer_it_received->ptr_array_AF_Ind_recurrent_units->ptr_recurrent_connection_index,
                   this->ptr_array_parameters + *tmp_ptr_encoded_layer_it_first_AF_ind->ptr_recurrent_connection_index,
                   tmp_number_units * sizeof(var));
}
}
