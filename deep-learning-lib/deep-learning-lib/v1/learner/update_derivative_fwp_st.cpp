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
#include "deep-learning-lib/ops/math.hpp"

namespace DL::v1 {
void Model::update_derivatives(size_t const batch_size,
                                                                      Layer *const layer_it,
                                                                      Layer const *const last_layer)
{
    switch(this->type)
    {
        case MODEL::AUTOENCODER:
            if(this->pre_training_level != 0_UZ)
            {
                this->Update_Derivative_Weight__Pre_Training(batch_size);

                break;
            }
        default:
            if(this->seq_w > 1_UZ)
            {
                if(this->use_mp && this->is_mp_initialized)
                {
                    this->RNN__Update_Derivative_Weight_Batch__OpenMP(batch_size,
                                                                                                        layer_it,
                                                                                                        last_layer);
                }
                else
                {
                    this->RNN__Update_Derivative_Weight_Batch__Loop(batch_size,
                                                                                                  layer_it,
                                                                                                  last_layer);
                }
            }
            else
            {
                if(this->use_mp && this->is_mp_initialized)
                {
                    this->FF__Update_Derivative_Weight_Batch__OpenMP(batch_size,
                                                                                                     layer_it,
                                                                                                     last_layer);
                }
                else
                {
                    this->FF__Update_Derivative_Weight_Batch__Loop(batch_size,
                                                                                               layer_it,
                                                                                               last_layer);
                }
            }
                break;
    }
}

void Model::Update_Derivative_Weight__Pre_Training(size_t const batch_size)
{
    if(this->pre_training_level == 0_UZ)
    {
        ERR(L"The neural network use the pre-training function without the mode pre-training activate.",);

        return;
    }

    if(this->seq_w > 1_UZ)
    {
        if(this->use_mp && this->is_mp_initialized)
        { this->RNN__Update_Derivative_Weight_Batch__Pre_Training__OpenMP(batch_size); }
        else
        { this->RNN__Update_Derivative_Weight_Batch__Pre_Training__Loop(batch_size); }
    }
    else
    {
        if(this->use_mp && this->is_mp_initialized)
        { this->FF__Update_Derivative_Weight_Batch__Pre_Training__OpenMP(batch_size); }
        else
        { this->FF__Update_Derivative_Weight_Batch__Pre_Training__Loop(batch_size); }
    }
}

void Model::FF__Update_Derivative_Weight_Batch__Loop(size_t const batch_size,
                                                                                                  Layer *layer_it,
                                                                                                  Layer const *const last_layer)
{
    Layer const *prev_conn_layer;

    for(; layer_it != last_layer; ++layer_it)
    {
        // If the current layer is a pooling/residual layer, continue.
        if(layer_it->type_layer == LAYER::AVERAGE_POOLING
          ||
          layer_it->type_layer == LAYER::MAX_POOLING
          ||
          layer_it->type_layer == LAYER::RESIDUAL) { continue; }
        
        prev_conn_layer = layer_it->previous_connected_layers[0];

        switch(layer_it->type_layer)
        {
            case LAYER::FULLY_CONNECTED:
                this->Update_Derivative_Weight__FC__Loop(0_UZ,
                                                                                 batch_size,
                                                                                 *prev_conn_layer->ptr_number_outputs,
                                                                                 prev_conn_layer->ptr_array_outputs,
                                                                                 layer_it);
                    break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         layer_it->type_layer,
                                         LAYER_NAME[layer_it->type_layer].c_str());
                    return;
        }
    }
}

void Model::FF__Update_Derivative_Weight_Batch__Pre_Training__Loop(size_t const batch_size)
{
    Layer const *prev_conn_layer;
    Layer *layer_it;
    
    // Coded level part.
    layer_it = this->ptr_array_layers + this->pre_training_level;
    prev_conn_layer = layer_it->previous_connected_layers[0];
    
    switch(layer_it->type_layer)
    {
        case LAYER::FULLY_CONNECTED:
            this->Update_Derivative_Weight__FC__Loop(0_UZ,
                                                                             batch_size,
                                                                             *prev_conn_layer->ptr_number_outputs,
                                                                             prev_conn_layer->ptr_array_outputs,
                                                                             layer_it);
                break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                                     layer_it->type_layer,
                                     LAYER_NAME[layer_it->type_layer].c_str());
                return;
    }
    // |END| Coded level part. |END|

    // Decode level part.
    prev_conn_layer = layer_it;
    layer_it = this->ptr_last_layer - static_cast<size_t>(layer_it - this->ptr_array_layers);
    
    switch(layer_it->type_layer)
    {
        case LAYER::FULLY_CONNECTED:
            this->Update_Derivative_Weight__FC__Loop(0_UZ,
                                                                             batch_size,
                                                                             *prev_conn_layer->ptr_number_outputs,
                                                                             prev_conn_layer->ptr_array_outputs,
                                                                             layer_it);
                break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                                     layer_it->type_layer,
                                     LAYER_NAME[layer_it->type_layer].c_str());
                return;
    }
    // |END| Decode level part. |END|
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Update_Derivative_Weight__FC__Loop(size_t const time_step_index_received,
                                                                                        size_t const batch_size,
                                                                                        size_t const input_size_received,
                                                                                        var const *const ptr_array_inputs_received,
                                                                                        Layer *const layer_it)
{
    Neuron_unit *const tmp_ptr_layer_first_neuron_unit(layer_it->ptr_array_neuron_units);

    AF_unit *const tmp_ptr_layer_first_AF_unit(layer_it->ptr_array_AF_units);
    AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(layer_it->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_AF_size(static_cast<size_t>(layer_it->ptr_last_AF_unit - tmp_ptr_layer_first_AF_unit) + static_cast<size_t>(layer_it->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_first_AF_Ind_recurrent_unit)),
                       tmp_output_size(static_cast<size_t>(layer_it->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));

    // Weights.
    this->Update_Derivative_Weight__FC__Loop(time_step_index_received,
                                                                     batch_size,
                                                                     input_size_received,
                                                                     tmp_output_size,
                                                                     layer_it->ptr_array_pre_summations,
                                                                     tmp_ptr_layer_first_neuron_unit->ptr_array_errors,
                                                                     this->ptr_array_derivatives_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index);

    // Bias.
    if(layer_it->Use__Bias())
    {
        this->Update_Derivative_Weight__Bias__Loop(time_step_index_received,
                                                                           batch_size,
                                                                           tmp_output_size,
                                                                           tmp_ptr_layer_first_neuron_unit->ptr_array_errors,
                                                                           this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index);
    }

    // Recurrent connection(s).
    if(time_step_index_received != 0_UZ
      &&
      layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT
      &&
      tmp_AF_size != 0_UZ)
    {
        this->Update_Derivative_Weight__FC_Ind_RNN__Loop(time_step_index_received,
                                                                                        batch_size,
                                                                                        tmp_AF_size,
                                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_dAFs,
                                                                                        this->ptr_array_derivatives_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Update_Derivative_Weight__FC__Loop(size_t const time_step_index_received,
                                                                                        size_t const batch_size,
                                                                                        size_t const input_size_received,
                                                                                        size_t const derivative_size_received,
                                                                                        var const *const ptr_array_inputs_received,
                                                                                        real const *const ptr_array_derivative_inputs_received,
                                                                                        real *const ptr_array_derivatives_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_derivative_index;
    
    var const *tmp_ptr_array_inputs;
    real const *tmp_ptr_array_derivative_inputs;
    real *tmp_ptr_array_derivatives_parameters,
         tmp_error;

    for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * derivative_size_received + tmp_derivative_timed_batched_index;

        tmp_ptr_array_derivatives_parameters = ptr_array_derivatives_received;
        
        for(tmp_derivative_index = 0_UZ; tmp_derivative_index != derivative_size_received; ++tmp_derivative_index,
                                                                                                                              tmp_ptr_array_derivatives_parameters += input_size_received)
        {
            tmp_error = tmp_ptr_array_derivative_inputs[tmp_derivative_index];
            
            for(tmp_connection_index = 0_UZ; tmp_connection_index != input_size_received; ++tmp_connection_index) { tmp_ptr_array_derivatives_parameters[tmp_connection_index] += tmp_error * cast(tmp_ptr_array_inputs[tmp_connection_index]); }
        }
    }
}

void Model::Update_Derivative_Weight__Bias__Loop(size_t const time_step_index_received,
                                                                                          size_t const batch_size,
                                                                                          size_t const derivative_size_received,
                                                                                          real const *const ptr_array_derivative_inputs_received,
                                                                                          real *const ptr_array_derivatives_received)
{
    size_t const tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_example_index,
              tmp_unit_index;
    
    real const *tmp_ptr_array_derivative_inputs;

    for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
    {
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + tmp_example_index * derivative_size_received + tmp_derivative_timed_batched_index;

        for(tmp_unit_index = 0_UZ; tmp_unit_index != derivative_size_received; ++tmp_unit_index) { ptr_array_derivatives_received[tmp_unit_index] += tmp_ptr_array_derivative_inputs[tmp_unit_index]; }
    }
}
}
