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
#include "deep-learning-lib/v1/ops/activations/functions.hpp"
#include "deep-learning-lib/v1/mem/reallocate.hpp"

namespace DL::v1 {
void Model::RNN__Forward_Pass_Batch__Loop(size_t const batch_size, real const *const *const ptr_array_inputs_received,
                                                                                    Layer *const ptr_first_layer_received,
                                                                                    Layer const *const last_layer)
{
    Layer const *prev_conn_layer;
    Layer *layer_it(ptr_first_layer_received + 1);
    
    // Training mode.
    if(this->type_state_propagation >= PROPAGATION::TRAINING)
    {
        // If the network use normalization.
        if(this->Use__Normalization())
        {
            VARZERO(this->ptr_array_normalized_batch_units_means,
                          this->number_threads * this->total_normalized_units_allocated * this->seq_w * sizeof(var));
            VARZERO(this->ptr_array_normalized_batch_units_variances,
                          this->number_threads * this->total_normalized_units_allocated * this->seq_w * sizeof(var));
        }
    }

    // Input layer.
    this->assign_inputs_rec_st(batch_size, ptr_array_inputs_received);
    // |END| Input layer. |END|
    
    // Loop through each layer and do a forward propagation.
    for(; layer_it != last_layer; ++layer_it)
    {
        prev_conn_layer = layer_it->previous_connected_layers[0];

        switch(layer_it->type_layer)
        {
            case LAYER::AVERAGE_POOLING:
                this->Recurrent__Forward_Pass__Average_Pooling__Loop(batch_size,
                                                                                                      *prev_conn_layer->ptr_number_outputs,
                                                                                                      prev_conn_layer->ptr_array_outputs,
                                                                                                      layer_it);
                    break;
            case LAYER::FULLY_CONNECTED:
            case LAYER::FULLY_CONNECTED_RECURRENT:
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Recurrent__Forward_Pass__FC__Loop(batch_size,
                                                                                  *prev_conn_layer->ptr_number_outputs,
                                                                                  prev_conn_layer->ptr_array_outputs,
                                                                                  layer_it);
                    break;
            case LAYER::LSTM:
                if(layer_it->Use__Bidirectional())
                {
                    this->Recurrent__Forward_Pass__LSTM__Loop(true,
                                                                                          batch_size,
                                                                                          *prev_conn_layer->ptr_number_outputs,
                                                                                          prev_conn_layer->ptr_array_outputs,
                                                                                          &layer_it->ptr_Bidirectional_Layer->forward_layer);
                    this->Recurrent__Forward_Pass__LSTM__Loop(false,
                                                                                          batch_size,
                                                                                          *prev_conn_layer->ptr_number_outputs,
                                                                                          prev_conn_layer->ptr_array_outputs,
                                                                                          &layer_it->ptr_Bidirectional_Layer->backward_layer);
                }
                else
                {
                    this->Recurrent__Forward_Pass__LSTM__Loop(true,
                                                                                          batch_size,
                                                                                          *prev_conn_layer->ptr_number_outputs,
                                                                                          prev_conn_layer->ptr_array_outputs,
                                                                                          layer_it);
                }
                    break;
            case LAYER::MAX_POOLING:
                this->Recurrent__Forward_Pass__Max_Pooling__Loop(batch_size,
                                                                                                *prev_conn_layer->ptr_number_outputs,
                                                                                                prev_conn_layer->ptr_array_outputs,
                                                                                                layer_it);
                    break;
            case LAYER::RESIDUAL: this->Recurrent__Forward_Pass__Residual__Loop(batch_size, layer_it); break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         layer_it->type_layer,
                                         LAYER_NAME[layer_it->type_layer].c_str());
                    return;
        }
    }
}

void Model::RNN__Forward_Pass_Batch__Pre_Training__Loop(
    size_t const batch_size,
    real const *const *const ptr_array_inputs_received) {
    Layer const *const last_layer(this->ptr_array_layers + this->pre_training_level),
                               *prev_conn_layer;
    Layer *layer_it(this->ptr_array_layers + 1);
    
    // Training mode.
    if(this->type_state_propagation >= PROPAGATION::TRAINING)
    {
        // If the network use normalization.
        if(this->Use__Normalization())
        {
            VARZERO(this->ptr_array_normalized_batch_units_means,
                        this->number_threads * this->total_normalized_units_allocated * this->seq_w * sizeof(var));
            VARZERO(this->ptr_array_normalized_batch_units_variances,
                        this->number_threads * this->total_normalized_units_allocated * this->seq_w * sizeof(var));
        }
    }

    // Input layer.
    this->assign_inputs_rec_st(batch_size, ptr_array_inputs_received);
    // |END| Input layer. |END|
    
    // Loop through each encoded layer and do a forward propagation.
    for(; layer_it != last_layer; ++layer_it)
    {
        prev_conn_layer = layer_it->previous_connected_layers[0];

        switch(layer_it->type_layer)
        {
            case LAYER::FULLY_CONNECTED:
            case LAYER::FULLY_CONNECTED_RECURRENT:
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Recurrent__Forward_Pass__Encode__FC__Loop(batch_size,
                                                                                                     *prev_conn_layer->ptr_number_outputs,
                                                                                                     prev_conn_layer->ptr_array_outputs,
                                                                                                     layer_it);
                    break;
            case LAYER::LSTM:
                if(layer_it->Use__Bidirectional())
                {
                    this->Recurrent__Forward_Pass__Encode__LSTM__Loop(true,
                                                                                                         batch_size,
                                                                                                         *prev_conn_layer->ptr_number_outputs,
                                                                                                         prev_conn_layer->ptr_array_outputs,
                                                                                                         &layer_it->ptr_Bidirectional_Layer->forward_layer);
                    this->Recurrent__Forward_Pass__Encode__LSTM__Loop(false,
                                                                                                         batch_size,
                                                                                                         *prev_conn_layer->ptr_number_outputs,
                                                                                                         prev_conn_layer->ptr_array_outputs,
                                                                                                         &layer_it->ptr_Bidirectional_Layer->backward_layer);
                }
                else
                {
                    this->Recurrent__Forward_Pass__Encode__LSTM__Loop(true,
                                                                                                        batch_size,
                                                                                                        *prev_conn_layer->ptr_number_outputs,
                                                                                                        prev_conn_layer->ptr_array_outputs,
                                                                                                        layer_it);
                }
                    break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         layer_it->type_layer,
                                         LAYER_NAME[layer_it->type_layer].c_str());
                    return;
        }
    }
    
    // Coded level part.
    prev_conn_layer = layer_it->previous_connected_layers[0];

    switch(layer_it->type_layer)
    {
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Forward_Pass__Code__FC__Loop(batch_size,
                                                                                         *prev_conn_layer->ptr_number_outputs,
                                                                                         prev_conn_layer->ptr_array_outputs,
                                                                                         layer_it);
                break;
        case LAYER::LSTM:
            if(layer_it->Use__Bidirectional())
            {
                this->Recurrent__Forward_Pass__Code__LSTM__Loop(true,
                                                                                                 batch_size,
                                                                                                 *prev_conn_layer->ptr_number_outputs,
                                                                                                 prev_conn_layer->ptr_array_outputs,
                                                                                                 &layer_it->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Forward_Pass__Code__LSTM__Loop(false,
                                                                                                 batch_size,
                                                                                                 *prev_conn_layer->ptr_number_outputs,
                                                                                                 prev_conn_layer->ptr_array_outputs,
                                                                                                 &layer_it->ptr_Bidirectional_Layer->backward_layer);
            }
            else
            {
                this->Recurrent__Forward_Pass__Code__LSTM__Loop(true,
                                                                                                 batch_size,
                                                                                                 *prev_conn_layer->ptr_number_outputs,
                                                                                                 prev_conn_layer->ptr_array_outputs,
                                                                                                 layer_it);
            }
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
            this->Recurrent__Forward_Pass__Decode__FC__Loop(batch_size,
                                                                                                 *prev_conn_layer->ptr_number_outputs,
                                                                                                 prev_conn_layer->ptr_array_outputs,
                                                                                                 layer_it);
                break;
        case LAYER::LSTM:
            if(layer_it->Use__Bidirectional())
            {
                this->Recurrent__Forward_Pass__Decode__LSTM__Loop(true,
                                                                                                     batch_size,
                                                                                                     *prev_conn_layer->ptr_number_outputs,
                                                                                                     prev_conn_layer->ptr_array_outputs,
                                                                                                     &layer_it->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Forward_Pass__Decode__LSTM__Loop(false,
                                                                                                     batch_size,
                                                                                                     *prev_conn_layer->ptr_number_outputs,
                                                                                                     prev_conn_layer->ptr_array_outputs,
                                                                                                     &layer_it->ptr_Bidirectional_Layer->backward_layer);
            }
            else
            {
                this->Recurrent__Forward_Pass__Decode__LSTM__Loop(true,
                                                                                                    batch_size,
                                                                                                    *prev_conn_layer->ptr_number_outputs,
                                                                                                    prev_conn_layer->ptr_array_outputs,
                                                                                                    layer_it);
            }
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

void Model::Recurrent__Forward_Pass__Average_Pooling__Loop(size_t const batch_size,
                                                                                                             size_t const input_unit_size_received,
                                                                                                             var const *const ptr_array_inputs_received,
                                                                                                             Layer *const layer_it)
{
    for(size_t tmp_time_step_index(0_UZ); tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Forward_Pass__Average_Pooling__Loop(tmp_time_step_index,
                                                                            batch_size,
                                                                            input_unit_size_received,
                                                                            ptr_array_inputs_received,
                                                                            layer_it);
    }
}

void Model::Recurrent__Forward_Pass__Dropout__Bernoulli__Inference__Loop(size_t const batch_size,
                                                                                                                                 size_t const input_unit_size_received,
                                                                                                                                 real const retention_probability_received,
                                                                                                                                 var *const ptr_array_inputs_received)
{
    for(size_t tmp_time_step_index(0_UZ); tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(tmp_time_step_index,
                                                                                                batch_size,
                                                                                                input_unit_size_received,
                                                                                                retention_probability_received,
                                                                                                ptr_array_inputs_received);
    }
}

void Model::Recurrent__Forward_Pass__Dropout__ShakeDrop__Loop(size_t const batch_size,
                                                                                                                   size_t const input_unit_size_received,
                                                                                                                   bool *const ptr_array_mask_dopout_shakedrop_received,
                                                                                                                   real const lower_bound,
                                                                                                                   real const upper_bound,
                                                                                                                   real const dropout_probability_received,
                                                                                                                   var *const ptr_array_inputs_received)
{
    for(size_t tmp_time_step_index(0_UZ); tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Forward_Pass__Dropout__ShakeDrop__Loop(tmp_time_step_index,
                                                                                   batch_size,
                                                                                   input_unit_size_received,
                                                                                   ptr_array_mask_dopout_shakedrop_received,
                                                                                   lower_bound,
                                                                                   upper_bound,
                                                                                   dropout_probability_received,
                                                                                   ptr_array_inputs_received);
    }
}

void Model::Recurrent__Forward_Pass__FC__Loop(size_t const batch_size,
                                                                                         size_t const input_unit_size_received,
                                                                                         var const *const ptr_array_inputs_received,
                                                                                         Layer *const layer_it)
{
    for(size_t tmp_time_step_index(0_UZ); tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Forward_Pass__FC__Loop(tmp_time_step_index,
                                                        batch_size,
                                                        input_unit_size_received,
                                                        ptr_array_inputs_received,
                                                        layer_it);
    }
}

void Model::Recurrent__Forward_Pass__Encode__FC__Loop(size_t const batch_size,
                                                                                                            size_t const input_unit_size_received,
                                                                                                            var const *const ptr_array_inputs_received,
                                                                                                            Layer *const layer_it)
{
    for(size_t tmp_time_step_index(0_UZ); tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Forward_Pass__Encode__FC__Loop(tmp_time_step_index,
                                                                       batch_size,
                                                                       input_unit_size_received,
                                                                       ptr_array_inputs_received,
                                                                       layer_it);
    }
}

void Model::Recurrent__Forward_Pass__Code__FC__Loop(size_t const batch_size,
                                                                                                         size_t const input_unit_size_received,
                                                                                                         var const *const ptr_array_inputs_received,
                                                                                                         Layer *const layer_it)
{
    for(size_t tmp_time_step_index(0_UZ); tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Forward_Pass__Code__FC__Loop(tmp_time_step_index,
                                                                   batch_size,
                                                                   input_unit_size_received,
                                                                   ptr_array_inputs_received,
                                                                   layer_it);
    }
}

void Model::Recurrent__Forward_Pass__Decode__FC__Loop(size_t const batch_size,
                                                                                                            size_t const input_unit_size_received,
                                                                                                            var const *const ptr_array_inputs_received,
                                                                                                            Layer *const layer_it)
{
    for(size_t tmp_time_step_index(0_UZ); tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Forward_Pass__Decode__FC__Loop(tmp_time_step_index,
                                                                       batch_size,
                                                                       input_unit_size_received,
                                                                       ptr_array_inputs_received,
                                                                       layer_it);
    }
}

void Model::Recurrent__Forward_Pass__LSTM__Loop(bool const forward_layer_received,
                                                                                             size_t const batch_size,
                                                                                             size_t const input_unit_size_received,
                                                                                             var const *const ptr_array_inputs_received,
                                                                                             Layer *const layer_it)
{
    long long int tmp_time_step_index,
                       tmp_time_step_reverse_direction,
                       tmp_time_step_start(forward_layer_received ? 0ll : static_cast<long long int>(this->seq_w - 1_UZ)),
                       tmp_time_step_end(forward_layer_received ? static_cast<long long int>(this->seq_w) : -1ll);
    
    for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? ++tmp_time_step_index : --tmp_time_step_index)
    {
        tmp_time_step_reverse_direction = forward_layer_received ? (tmp_time_step_index - 1ll) : (tmp_time_step_index + 1ll);

        this->Forward_Pass__LSTM__Loop(tmp_time_step_index,
                                                             tmp_time_step_reverse_direction,
                                                             tmp_time_step_start,
                                                             batch_size,
                                                             input_unit_size_received,
                                                             ptr_array_inputs_received,
                                                             layer_it);
    }
}

void Model::Recurrent__Forward_Pass__Encode__LSTM__Loop(bool const forward_layer_received,
                                                                                                            size_t const batch_size,
                                                                                                            size_t const input_unit_size_received,
                                                                                                            var const *const ptr_array_inputs_received,
                                                                                                            Layer *const layer_it)
{
    long long int tmp_time_step_index,
                       tmp_time_step_reverse_direction,
                       tmp_time_step_start(forward_layer_received ? 0ll : static_cast<long long int>(this->seq_w - 1_UZ)),
                       tmp_time_step_end(forward_layer_received ? static_cast<long long int>(this->seq_w) : -1ll);
    
    for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? ++tmp_time_step_index : --tmp_time_step_index)
    {
        tmp_time_step_reverse_direction = forward_layer_received ? (tmp_time_step_index - 1ll) : (tmp_time_step_index + 1ll);

        this->Forward_Pass__Encode__LSTM__Loop(tmp_time_step_index,
                                                                           tmp_time_step_reverse_direction,
                                                                           tmp_time_step_start,
                                                                           batch_size,
                                                                           input_unit_size_received,
                                                                           ptr_array_inputs_received,
                                                                           layer_it);
    }
}

void Model::Recurrent__Forward_Pass__Code__LSTM__Loop(bool const forward_layer_received,
                                                                                                        size_t const batch_size,
                                                                                                        size_t const input_unit_size_received,
                                                                                                        var const *const ptr_array_inputs_received,
                                                                                                        Layer *const layer_it)
{
    long long int tmp_time_step_index,
                       tmp_time_step_reverse_direction,
                       tmp_time_step_start(forward_layer_received ? 0ll : static_cast<long long int>(this->seq_w - 1_UZ)),
                       tmp_time_step_end(forward_layer_received ? static_cast<long long int>(this->seq_w) : -1ll);
    
    for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? ++tmp_time_step_index : --tmp_time_step_index)
    {
        tmp_time_step_reverse_direction = forward_layer_received ? (tmp_time_step_index - 1ll) : (tmp_time_step_index + 1ll);

        this->Forward_Pass__Code__LSTM__Loop(tmp_time_step_index,
                                                                       tmp_time_step_reverse_direction,
                                                                       tmp_time_step_start,
                                                                       batch_size,
                                                                       input_unit_size_received,
                                                                       ptr_array_inputs_received,
                                                                       layer_it);
    }
}

void Model::Recurrent__Forward_Pass__Decode__LSTM__Loop(bool const forward_layer_received,
                                                                                                            size_t const batch_size,
                                                                                                            size_t const input_unit_size_received,
                                                                                                            var const *const ptr_array_inputs_received,
                                                                                                            Layer *const layer_it)
{
    long long int tmp_time_step_index,
                       tmp_time_step_reverse_direction,
                       tmp_time_step_start(forward_layer_received ? 0ll : static_cast<long long int>(this->seq_w - 1_UZ)),
                       tmp_time_step_end(forward_layer_received ? static_cast<long long int>(this->seq_w) : -1ll);
    
    for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? ++tmp_time_step_index : --tmp_time_step_index)
    {
        tmp_time_step_reverse_direction = forward_layer_received ? (tmp_time_step_index - 1ll) : (tmp_time_step_index + 1ll);

        this->Forward_Pass__Decode__LSTM__Loop(tmp_time_step_index,
                                                                           tmp_time_step_reverse_direction,
                                                                           tmp_time_step_start,
                                                                           batch_size,
                                                                           input_unit_size_received,
                                                                           ptr_array_inputs_received,
                                                                           layer_it);
    }
}

void Model::Recurrent__Forward_Pass__Max_Pooling__Loop(size_t const batch_size,
                                                                                                       size_t const input_unit_size_received,
                                                                                                       var const *const ptr_array_inputs_received,
                                                                                                       Layer *const layer_it)
{
    for(size_t tmp_time_step_index(0_UZ); tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Forward_Pass__Max_Pooling__Loop(tmp_time_step_index,
                                                                       batch_size,
                                                                       input_unit_size_received,
                                                                       ptr_array_inputs_received,
                                                                       layer_it);
    }
}

void Model::Recurrent__Forward_Pass__Residual__Loop(size_t const batch_size, Layer *&layer_it)
{
    size_t tmp_time_step_index;

    var *tmp_ptr_array_inputs;
    
    Layer const *const tmp_ptr_end_block_layer(layer_it + layer_it->block_depth + 1),
                               *prev_conn_layer;
    Layer *const tmp_ptr_residual_layer(layer_it);
    
    union Normalized_unit *const tmp_ptr_residual_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // First block layer.
    this->Recurrent__Forward_Pass__Residual__Layer__Loop(true,
                                                                                          batch_size,
                                                                                          ++layer_it);
    // |END| First block layer. |END|

    // Remaining layer(s).
    for(++layer_it; layer_it != tmp_ptr_end_block_layer; ++layer_it)
    {
        this->Recurrent__Forward_Pass__Residual__Layer__Loop(false,
                                                                                              batch_size,
                                                                                              layer_it);
    }
    // |END| Remaining layer(s). |END|
    
    // Assign layer iterator to the last layer inside the block.
    --layer_it;

    // Shortcut.
    //  Assign previous layer iterator to the previously connected layer from the residual layer.
    prev_conn_layer = tmp_ptr_residual_layer->previous_connected_layers[0];
    
    //  Store the input(s) (block, last layer output(s)).
    tmp_ptr_array_inputs = layer_it->ptr_array_outputs;
    
    // Normalization.
    if(tmp_ptr_residual_layer->Use__Normalization())
    {
        // Training mode.
        if(this->type_state_propagation >= PROPAGATION::TRAINING)
        {
            switch(tmp_ptr_residual_layer->type_normalization)
            {
                case LAYER_NORM::BATCH_NORMALIZATION:
                    for(tmp_time_step_index = 0_UZ; tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
                    {
                        this->Forward_Pass__Batch_Normalization__Training__Loop(tmp_time_step_index,
                                                                                                                 batch_size,
                                                                                                                 *layer_it->ptr_number_outputs,
                                                                                                                 tmp_ptr_array_inputs,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                 tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                    }
                        break;
                case LAYER_NORM::BATCH_RENORMALIZATION:
                    for(tmp_time_step_index = 0_UZ; tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
                    {
                        this->Forward_Pass__Batch_Renormalization__Training__Loop(tmp_time_step_index,
                                                                                                                    batch_size,
                                                                                                                    *layer_it->ptr_number_outputs,
                                                                                                                    tmp_ptr_array_inputs,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                    }
                        break;
                default:
                    ERR(L"Layer normalization (%u | %ls) is not managed in",
                                             layer_it->type_normalization,
                                             LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                        break;
            }
        }
        // Inference mode.
        else
        {
            for(tmp_time_step_index = 0_UZ; tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
            {
                this->Forward_Pass__Batch_Normalization__Inference__Loop(tmp_time_step_index,
                                                                                                          batch_size,
                                                                                                          *layer_it->ptr_number_outputs,
                                                                                                          tmp_ptr_array_inputs,
                                                                                                          tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                          tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                          tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                          tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                          tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            }
        }

        // Store the new inputs (value normalize).
        tmp_ptr_array_inputs = tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
    }
    
    // Dropout.
    if(tmp_ptr_residual_layer->type_dropout == LAYER_DROPOUT::SHAKEDROP)
    {
        // If the state of propagation is strictly at training.
        if(this->type_state_propagation == PROPAGATION::TRAINING)
        {
            this->Recurrent__Forward_Pass__Dropout__ShakeDrop__Loop(batch_size,
                                                                                                        *layer_it->ptr_number_outputs,
                                                                                                        tmp_ptr_residual_layer->ptr_array__mask__dropout__shakedrop,
                                                                                                        -1_r,
                                                                                                        1_r,
                                                                                                        tmp_ptr_residual_layer->dropout_values[0],
                                                                                                        tmp_ptr_array_inputs);
        }
        // Inference mode.
        else
        {
            this->Recurrent__Forward_Pass__Dropout__Bernoulli__Inference__Loop(batch_size,
                                                                                                                      *layer_it->ptr_number_outputs,
                                                                                                                      1_r - tmp_ptr_residual_layer->dropout_values[0],
                                                                                                                      tmp_ptr_array_inputs);
        }
    }

    //  Zero-padded identity-mapping shortcut.
    this->Recurrent__Forward_Pass__Zero_Padded_Identity__Loop(batch_size,
                                                                                                 *prev_conn_layer->ptr_number_outputs, // Shortcut.
                                                                                                 *layer_it->ptr_number_outputs, // Block, last layer.
                                                                                                 prev_conn_layer->ptr_array_outputs, // Shortcut.
                                                                                                 tmp_ptr_array_inputs, // Block, last layer.
                                                                                                 tmp_ptr_residual_layer);
    // |END| Shortcut. |END|
}

void Model::Recurrent__Forward_Pass__Residual__Layer__Loop(bool const is_block_input_layer_received,
                                                                                                             size_t const batch_size,
                                                                                                             Layer *&layer_it)
{
    Layer const *const prev_conn_layer(layer_it->previous_connected_layers[0]);

    switch(layer_it->type_layer)
    {
        case LAYER::AVERAGE_POOLING:
            this->Recurrent__Forward_Pass__Average_Pooling__Loop(batch_size,
                                                                                                  *prev_conn_layer->ptr_number_outputs,
                                                                                                  prev_conn_layer->ptr_array_outputs,
                                                                                                  layer_it);
                break;
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Forward_Pass__Residual__FC__Loop(is_block_input_layer_received,
                                                                                              batch_size,
                                                                                              *prev_conn_layer->ptr_number_outputs,
                                                                                              prev_conn_layer->ptr_array_outputs,
                                                                                              layer_it);
                break;
        case LAYER::LSTM:
            if(layer_it->Use__Bidirectional())
            {
                this->Recurrent__Forward_Pass__LSTM__Loop(true,
                                                                                      batch_size,
                                                                                      *prev_conn_layer->ptr_number_outputs,
                                                                                      prev_conn_layer->ptr_array_outputs,
                                                                                      &layer_it->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Forward_Pass__LSTM__Loop(false,
                                                                                      batch_size,
                                                                                      *prev_conn_layer->ptr_number_outputs,
                                                                                      prev_conn_layer->ptr_array_outputs,
                                                                                      &layer_it->ptr_Bidirectional_Layer->backward_layer);
            }
            else
            {
                this->Recurrent__Forward_Pass__LSTM__Loop(true,
                                                                                      batch_size,
                                                                                      *prev_conn_layer->ptr_number_outputs,
                                                                                      prev_conn_layer->ptr_array_outputs,
                                                                                      layer_it);
            }
                break;
        case LAYER::MAX_POOLING:
            this->Recurrent__Forward_Pass__Max_Pooling__Loop(batch_size,
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

void Model::Recurrent__Forward_Pass__Residual__FC__Loop(bool const is_block_input_layer_received,
                                                                                                         size_t const batch_size,
                                                                                                         size_t const input_unit_size_received,
                                                                                                         var const *const ptr_array_inputs_received,
                                                                                                         Layer *const layer_it)
{
    for(size_t tmp_time_step_index(0_UZ); tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Forward_Pass__Residual__FC__Loop(is_block_input_layer_received,
                                                                        tmp_time_step_index,
                                                                        batch_size,
                                                                        input_unit_size_received,
                                                                        ptr_array_inputs_received,
                                                                        layer_it);
    }
}

void Model::Recurrent__Forward_Pass__Zero_Padded_Identity__Loop(size_t const batch_size,
                                                                                                                    size_t const size_A_received,
                                                                                                                    size_t const size_B_received,
                                                                                                                    var const *const ptr_array_A_received,
                                                                                                                    var const *const ptr_array_B_received,
                                                                                                                    Layer *const layer_it)
{
    size_t const tmp_padding(layer_it->pooling_values[2]);

    var *const tmp_ptr_array_outputs(layer_it->ptr_array_basic_units->ptr_array_values);

    for(size_t tmp_time_step_index(0_UZ); tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Forward_Pass__Zero_Padded_Identity__Loop(tmp_time_step_index,
                                                                                   batch_size,
                                                                                   size_A_received,
                                                                                   size_B_received,
                                                                                   tmp_padding,
                                                                                   ptr_array_A_received,
                                                                                   ptr_array_B_received,
                                                                                   tmp_ptr_array_outputs);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Forward_Pass__FC_Ind_RNN__Loop(size_t const time_step_index_received,
                                                                                      size_t const batch_size,
                                                                                      size_t const input_size_received,
                                                                                      var const *const ptr_array_parameters_received,
                                                                                      var const *const ptr_array_AFs_received,
                                                                                      var const *const ptr_array_inputs_received,
                                                                                      var *const ptr_array_outputs_received)
{
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_input_previous_timed_batched_index(this->batch_size * input_size_received * (time_step_index_received - 1_UZ));
    size_t tmp_example_index,
              tmp_input_index;
    
    var const *tmp_ptr_array_inverse_timed_AFs,
                 *tmp_ptr_array_inputs;
    var *tmp_ptr_array_outputs;

    if(time_step_index_received != 0_UZ)
    {
        for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
        {
            tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

            tmp_ptr_array_inverse_timed_AFs = ptr_array_AFs_received + tmp_example_index * input_size_received + tmp_input_previous_timed_batched_index;
            tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

            for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_outputs[tmp_input_index] = tmp_ptr_array_inputs[tmp_input_index] + ptr_array_parameters_received[tmp_input_index] * tmp_ptr_array_inverse_timed_AFs[tmp_input_index]; }
        }
    }
    else
    {
#ifdef COMPILE_ADEPT
        // Identity.
        for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
        {
            tmp_ptr_array_inputs = ptr_array_inputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

            tmp_ptr_array_outputs = ptr_array_outputs_received + tmp_example_index * input_size_received + tmp_input_timed_batched_index;

            for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_outputs[tmp_input_index] = tmp_ptr_array_inputs[tmp_input_index]; }
        }
#else
        VARCOPY(ptr_array_outputs_received + tmp_input_timed_batched_index,
                       ptr_array_inputs_received + tmp_input_timed_batched_index,
                       this->batch_size * input_size_received * sizeof(var));
#endif
    }
}

void Model::Forward_Pass__LSTM__Gates_CIFO__Loop(long long int const time_step_index_received,
                                                                                                 long long int const time_step_reverse_direction_received,
                                                                                                 long long int const time_step_prediction_start_received,
                                                                                                 size_t const batch_size,
                                                                                                 size_t const layer_block_unit_size_received,
                                                                                                 size_t const layer_cell_unit_size_received,
                                                                                                 size_t const input_unit_size_received,
                                                                                                 var const *const ptr_array_inputs_received,
                                                                                                 Layer *const layer_it)
{
    size_t const tmp_number_inputs_connections(layer_it->ptr_array_block_units->last_index_feedforward_connection_input_gate - layer_it->ptr_array_block_units->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrents_connection(layer_it->ptr_array_block_units->last_index_recurrent_connection_input_gate - layer_it->ptr_array_block_units->first_index_recurrent_connection_input_gate);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_data_timed_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_reverse_direction_timed_index;
    
    var const *tmp_ptr_array_previous_layer_outputs,
                  *tmp_ptr_array_layer_reverse_direction_timed_outputs,
                  *tmp_ptr_array_cell_input_parameters,
                  *tmp_ptr_array_input_gate_parameters,
                  *tmp_ptr_array_forget_gate_parameters,
                  *tmp_ptr_array_output_gate_parameters;
    var tmp_cell_input_summation,
        tmp_input_gate_summation,
        tmp_forget_gate_summation,
        tmp_output_gate_summation;

    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it;

    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;
    
    if(time_step_index_received != time_step_prediction_start_received)
    {
        for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_cell_data_reverse_direction_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);

            tmp_ptr_array_previous_layer_outputs = ptr_array_inputs_received + tmp_example_index * input_unit_size_received + this->batch_size * input_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_ptr_array_layer_reverse_direction_timed_outputs = layer_it->ptr_array_cell_units->ptr_cell_output + tmp_cell_data_reverse_direction_timed_index;
            
            for(tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    // Cell-Input.
                    tmp_cell_input_summation = 0_r;

                    tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

                    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                    {
                        tmp_cell_input_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_cell_input_parameters[tmp_connection_index];
                    }

                    tmp_ptr_cell_unit_it->ptr_summation_input_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    // |END| Cell-Input. |END|

                    // Cell-Recurrent.
                    tmp_cell_input_summation = 0_r;

                    tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;

                    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
                    {
                        tmp_cell_input_summation += tmp_ptr_array_layer_reverse_direction_timed_outputs[tmp_connection_index] * tmp_ptr_array_cell_input_parameters[tmp_connection_index];
                    }

                    tmp_ptr_cell_unit_it->ptr_summation_recurrent_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    // |END| Cell-Recurrent. |END|
                }

                // Gates-Input.
                tmp_input_gate_summation = 0_r;
                tmp_forget_gate_summation = 0_r;
                tmp_output_gate_summation = 0_r;

                tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
                tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
                tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;
            
                for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                {
                    tmp_input_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_input_gate_parameters[tmp_connection_index];
                    tmp_forget_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_forget_gate_parameters[tmp_connection_index];
                    tmp_output_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_output_gate_parameters[tmp_connection_index];
                }

                tmp_ptr_block_unit_it->ptr_summation_input_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_input_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_input_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;
                // |END| Gates-Input. |END|

                // Gates-Recurrent.
                tmp_input_gate_summation = 0_r;
                tmp_forget_gate_summation = 0_r;
                tmp_output_gate_summation = 0_r;
                
                tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;
                tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;
                tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;
                
                for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
                {
                    tmp_input_gate_summation += tmp_ptr_array_layer_reverse_direction_timed_outputs[tmp_connection_index] * tmp_ptr_array_input_gate_parameters[tmp_connection_index];
                    tmp_forget_gate_summation += tmp_ptr_array_layer_reverse_direction_timed_outputs[tmp_connection_index] * tmp_ptr_array_forget_gate_parameters[tmp_connection_index];
                    tmp_output_gate_summation += tmp_ptr_array_layer_reverse_direction_timed_outputs[tmp_connection_index] * tmp_ptr_array_output_gate_parameters[tmp_connection_index];
                }

                tmp_ptr_block_unit_it->ptr_summation_recurrent_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_recurrent_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_recurrent_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;
                // |END| Gates-Recurrent. |END|
            }
        }
    }
    else
    {
        for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_ptr_array_previous_layer_outputs = ptr_array_inputs_received + tmp_example_index * input_unit_size_received + this->batch_size * input_unit_size_received * static_cast<size_t>(time_step_index_received);

            for(tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    // Cell-Input.
                    tmp_cell_input_summation = 0_r;

                    tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

                    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                    {
                        tmp_cell_input_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_cell_input_parameters[tmp_connection_index];
                    }

                    tmp_ptr_cell_unit_it->ptr_summation_input_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    // |END| Cell-Input. |END|
                }

                // Gates-Input.
                tmp_input_gate_summation = 0_r;
                tmp_forget_gate_summation = 0_r;
                tmp_output_gate_summation = 0_r;

                tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
                tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
                tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;
            
                for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                {
                    tmp_input_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_input_gate_parameters[tmp_connection_index];
                    tmp_forget_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_forget_gate_parameters[tmp_connection_index];
                    tmp_output_gate_summation += tmp_ptr_array_previous_layer_outputs[tmp_connection_index] * tmp_ptr_array_output_gate_parameters[tmp_connection_index];
                }

                tmp_ptr_block_unit_it->ptr_summation_input_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_input_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_input_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;
                // |END| Gates-Input. |END|
            }
        }
    }
}

void Model::Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(long long int const time_step_index_received,
                                                                                                              long long int const time_step_reverse_direction_received,
                                                                                                              long long int const time_step_prediction_start_received,
                                                                                                              size_t const batch_size,
                                                                                                              size_t const layer_block_unit_size_received,
                                                                                                              size_t const layer_cell_unit_size_received,
                                                                                                              var const *const ptr_array_summation_input_block_inputs_received,
                                                                                                              var const *const ptr_array_summation_recurrent_block_inputs_received,
                                                                                                              var const *const ptr_array_summation_input_inputs_gates_received,
                                                                                                              var const *const ptr_array_summation_recurrent_inputs_gates_received,
                                                                                                              var const *const ptr_array_summation_input_forgets_gates_received,
                                                                                                              var const *const ptr_array_summation_recurrent_forgets_gates_received,
                                                                                                              Layer *const layer_it)
{
    size_t const tmp_number_peepholes_connections(layer_it->ptr_array_block_units->last_index_peephole_input_gate - layer_it->ptr_array_block_units->first_index_peephole_input_gate);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_reverse_direction_timed_index;
    
    var const *const tmp_ptr_array_cell_input_bias(this->ptr_array_parameters + layer_it->first_bias_connection_index),
                  *const tmp_ptr_array_input_gate_bias(this->ptr_array_parameters + layer_it->first_bias_connection_index + layer_cell_unit_size_received),
                  *const tmp_ptr_array_forget_gate_bias(this->ptr_array_parameters + layer_it->first_bias_connection_index + layer_cell_unit_size_received + layer_block_unit_size_received),
                  *tmp_ptr_array_peephole_input_gate_parameters,
                  *tmp_ptr_array_peephole_forget_gate_parameters;
    var tmp_cell_input_summation,
        tmp_input_gate_summation,
        tmp_forget_gate_summation;

    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it;

    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;
    
    if(time_step_index_received != time_step_prediction_start_received)
    {
        for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_reverse_direction_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);

            tmp_ptr_block_unit_it = layer_it->ptr_array_block_units;

            for(tmp_cell_index = 0_UZ,
                tmp_block_index = 0_UZ; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Gates.        
                tmp_input_gate_summation = ptr_array_summation_input_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                            ptr_array_summation_recurrent_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                            tmp_ptr_array_input_gate_bias[tmp_block_index];
                
                tmp_forget_gate_summation = ptr_array_summation_input_forgets_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                             ptr_array_summation_recurrent_forgets_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                             tmp_ptr_array_forget_gate_bias[tmp_block_index];

            #ifndef NO_PEEPHOLE
                tmp_ptr_array_peephole_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate;
                tmp_ptr_array_peephole_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;

                for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                {
                    tmp_input_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_reverse_direction_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_input_gate_parameters[tmp_connection_index];
                    tmp_forget_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_reverse_direction_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_forget_gate_parameters[tmp_connection_index];
                }
            #endif
                
                tmp_ptr_block_unit_it->ptr_summation_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_summation;

                tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_input_gate_summation);
                tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_forget_gate_summation);
                // [0] |END| Gates. |END|
                
                // [0] Cell input/state.        
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                       ++tmp_cell_index)
                {
                    tmp_cell_input_summation = ptr_array_summation_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Input.
                                                               ptr_array_summation_recurrent_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Recurrent.
                                                               tmp_ptr_array_cell_input_bias[tmp_cell_index];

                    tmp_ptr_cell_unit_it->ptr_summation_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    
                    AF_FIRE(tmp_ptr_block_unit_it->activation_function_io,
                                  tmp_cell_input_summation,
                                  tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index]);
                    
                    tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] + tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index] * tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_reverse_direction_timed_index];
                }
                // [0] |END| Cell input/state. |END|
            }
        }
    }
    else
    {
        for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_ptr_block_unit_it = layer_it->ptr_array_block_units;

            for(tmp_cell_index = 0_UZ,
                tmp_block_index = 0_UZ; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Input gate.        
                tmp_input_gate_summation = ptr_array_summation_input_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                            tmp_ptr_array_input_gate_bias[tmp_block_index];
                
                tmp_ptr_block_unit_it->ptr_summation_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;

                tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_input_gate_summation);
                // [0] |END| Input gate. |END|
                
                // [0] Cell input/state.        
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                       ++tmp_cell_index)
                {
                    tmp_cell_input_summation = ptr_array_summation_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Input.
                                                               tmp_ptr_array_cell_input_bias[tmp_cell_index];

                    tmp_ptr_cell_unit_it->ptr_summation_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    
                    AF_FIRE(tmp_ptr_block_unit_it->activation_function_io,
                                  tmp_cell_input_summation,
                                  tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index]);

                    tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index];
                }
                // [0] |END| Cell input/state. |END|
            }
        }
    }
}

void Model::Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__Loop(long long int const time_step_index_received,
                                                                                                                             long long int const time_step_reverse_direction_received,
                                                                                                                             long long int const time_step_prediction_start_received,
                                                                                                                             size_t const batch_size,
                                                                                                                             size_t const layer_block_unit_size_received,
                                                                                                                             size_t const layer_cell_unit_size_received,
                                                                                                                             var const *const ptr_array_summation_input_block_inputs_received,
                                                                                                                             var const *const ptr_array_summation_recurrent_block_inputs_received,
                                                                                                                             var const *const ptr_array_summation_input_inputs_gates_received,
                                                                                                                             var const *const ptr_array_summation_recurrent_inputs_gates_received,
                                                                                                                             var const *const ptr_array_summation_input_forgets_gates_received,
                                                                                                                             var const *const ptr_array_summation_recurrent_forgets_gates_received,
                                                                                                                             Layer *const layer_it)
{
    size_t const tmp_number_peepholes_connections(layer_it->ptr_array_block_units->last_index_peephole_input_gate - layer_it->ptr_array_block_units->first_index_peephole_input_gate),
                       tmp_zoneout_mask_index(static_cast<size_t>(time_step_index_received) * layer_cell_unit_size_received);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_reverse_direction_timed_index;
    
    var const *const tmp_ptr_array_cell_input_bias(this->ptr_array_parameters + layer_it->first_bias_connection_index),
                  *const tmp_ptr_array_input_gate_bias(this->ptr_array_parameters + layer_it->first_bias_connection_index + layer_cell_unit_size_received),
                  *const tmp_ptr_array_forget_gate_bias(this->ptr_array_parameters + layer_it->first_bias_connection_index + layer_cell_unit_size_received + layer_block_unit_size_received),
                  *tmp_ptr_array_peephole_input_gate_parameters,
                  *tmp_ptr_array_peephole_forget_gate_parameters;
    var tmp_cell_input_summation,
        tmp_input_gate_summation,
        tmp_forget_gate_summation;

    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it;

    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;
    
    if(time_step_index_received != time_step_prediction_start_received)
    {
        for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_reverse_direction_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);

            tmp_ptr_block_unit_it = layer_it->ptr_array_block_units;

            for(tmp_cell_index = 0_UZ,
                tmp_block_index = 0_UZ; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Gates.        
                tmp_input_gate_summation = ptr_array_summation_input_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                            ptr_array_summation_recurrent_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                            tmp_ptr_array_input_gate_bias[tmp_block_index];
                
                tmp_forget_gate_summation = ptr_array_summation_input_forgets_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                             ptr_array_summation_recurrent_forgets_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                             tmp_ptr_array_forget_gate_bias[tmp_block_index];

            #ifndef NO_PEEPHOLE
                tmp_ptr_array_peephole_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate;
                tmp_ptr_array_peephole_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;

                for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                {
                    tmp_input_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_reverse_direction_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_input_gate_parameters[tmp_connection_index];
                    tmp_forget_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_reverse_direction_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_forget_gate_parameters[tmp_connection_index];
                }
            #endif
                
                tmp_ptr_block_unit_it->ptr_summation_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;
                tmp_ptr_block_unit_it->ptr_summation_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_summation;

                tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_input_gate_summation);
                tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_forget_gate_summation);
                // [0] |END| Gates. |END|
                
                // [0] Cell input/state.        
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                       ++tmp_cell_index)
                {
                    tmp_cell_input_summation = ptr_array_summation_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Input.
                                                               ptr_array_summation_recurrent_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Recurrent.
                                                               tmp_ptr_array_cell_input_bias[tmp_cell_index];

                    tmp_ptr_cell_unit_it->ptr_summation_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    
                    AF_FIRE(tmp_ptr_block_unit_it->activation_function_io,
                                  tmp_cell_input_summation,
                                  tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index]);
                    
                    if(tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state[tmp_zoneout_mask_index])
                    { tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] + tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index] * tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_reverse_direction_timed_index]; }
                    else
                    { tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_reverse_direction_timed_index]; }
                }
                // [0] |END| Cell input/state. |END|
            }
        }
    }
    else
    {
        for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_ptr_block_unit_it = layer_it->ptr_array_block_units;

            for(tmp_cell_index = 0_UZ,
                tmp_block_index = 0_UZ; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Input gate.        
                tmp_input_gate_summation = ptr_array_summation_input_inputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                            tmp_ptr_array_input_gate_bias[tmp_block_index];
                
                tmp_ptr_block_unit_it->ptr_summation_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_summation;

                tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_input_gate_summation);
                // [0] |END| Input gate. |END|
                
                // [0] Cell input/state.        
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                       ++tmp_cell_index)
                {
                    tmp_cell_input_summation = ptr_array_summation_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index] + // Input.
                                                               tmp_ptr_array_cell_input_bias[tmp_cell_index];

                    tmp_ptr_cell_unit_it->ptr_summation_cell_input[tmp_cell_data_timed_index] = tmp_cell_input_summation;
                    
                    AF_FIRE(tmp_ptr_block_unit_it->activation_function_io,
                                  tmp_cell_input_summation,
                                  tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index]);
                    
                    if(tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state[tmp_zoneout_mask_index])
                    { tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_input[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index]; }
                    else
                    { tmp_ptr_cell_unit_it->ptr_cell_state[tmp_cell_data_timed_index] = 0_r; }
                }
                // [0] |END| Cell input/state. |END|
            }
        }
    }
}

void Model::Forward_Pass__LSTM__Output__Loop(long long int const time_step_index_received,
                                                                                         size_t const batch_size,
                                                                                         size_t const layer_block_unit_size_received,
                                                                                         size_t const layer_cell_unit_size_received,
                                                                                         var const *const ptr_array_summation_input_outputs_gates_received,
                                                                                         var const *const ptr_array_summation_recurrent_outputs_gates_received,
                                                                                         Layer *const layer_it)
{
    size_t const tmp_number_peepholes_connections(layer_it->ptr_array_block_units->last_index_peephole_output_gate - layer_it->ptr_array_block_units->first_index_peephole_output_gate);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_cell_data_timed_index;
    
    var const *const tmp_ptr_array_output_gate_bias(this->ptr_array_parameters + layer_it->first_bias_connection_index + layer_cell_unit_size_received + 2_UZ * layer_block_unit_size_received),
                  *tmp_ptr_array_peephole_output_gate_parameters;
    var tmp_output_gate_summation;

    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it;
    
    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;
    
    for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
    {
        tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
        
        tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
        
        tmp_ptr_block_unit_it = layer_it->ptr_array_block_units;
        
        for(tmp_block_index = 0_UZ; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                            ++tmp_block_index)
        {
            // [0] Output gate.
            tmp_output_gate_summation = ptr_array_summation_input_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                          ptr_array_summation_recurrent_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                          tmp_ptr_array_output_gate_bias[tmp_block_index];
            
        #ifndef NO_PEEPHOLE
            tmp_ptr_array_peephole_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;

            for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
            {
                tmp_output_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_output_gate_parameters[tmp_connection_index];
            }
        #endif

            tmp_ptr_block_unit_it->ptr_summation_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;

            tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_output_gate_summation);
            // [0] |END| Output gate. |END|

            // [0] Cell output.
            for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
            {
                tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index];
            }
            // [0] |END| Cell output. |END|
        }
    }
}

void Model::Forward_Pass__LSTM__Output__Zoneout__Loop(long long int const time_step_index_received,
                                                                                                        long long int const time_step_reverse_direction_received,
                                                                                                        long long int const time_step_prediction_start_received,
                                                                                                        size_t const batch_size,
                                                                                                        size_t const layer_block_unit_size_received,
                                                                                                        size_t const layer_cell_unit_size_received,
                                                                                                        var const *const ptr_array_summation_input_outputs_gates_received,
                                                                                                        var const *const ptr_array_summation_recurrent_outputs_gates_received,
                                                                                                        Layer *const layer_it)
{
    size_t const tmp_number_peepholes_connections(layer_it->ptr_array_block_units->last_index_peephole_output_gate - layer_it->ptr_array_block_units->first_index_peephole_output_gate),
                       tmp_zoneout_mask_index(static_cast<size_t>(time_step_index_received) * layer_cell_unit_size_received);
    size_t tmp_example_index,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_reverse_direction_timed_index;
    
    var const *const tmp_ptr_array_output_gate_bias(this->ptr_array_parameters + layer_it->first_bias_connection_index + layer_cell_unit_size_received + 2_UZ * layer_block_unit_size_received),
                  *tmp_ptr_array_peephole_output_gate_parameters;
    var tmp_output_gate_summation;

    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it;
    
    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;
    
    if(time_step_index_received != time_step_prediction_start_received)
    {
        for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_reverse_direction_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);
            
            tmp_ptr_block_unit_it = layer_it->ptr_array_block_units;
            
            for(tmp_block_index = 0_UZ; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Output gate.
                tmp_output_gate_summation = ptr_array_summation_input_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                              ptr_array_summation_recurrent_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                              tmp_ptr_array_output_gate_bias[tmp_block_index];
                
            #ifndef NO_PEEPHOLE
                tmp_ptr_array_peephole_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;

                for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                {
                    tmp_output_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_output_gate_parameters[tmp_connection_index];
                }
            #endif

                tmp_ptr_block_unit_it->ptr_summation_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;

                tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_output_gate_summation);
                // [0] |END| Output gate. |END|

                // [0] Cell output.
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    if(tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output[tmp_zoneout_mask_index])
                    { tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index]; }
                    else
                    { tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_reverse_direction_timed_index]; }
                }
                // [0] |END| Cell output. |END|
            }
        }
    }
    else
    {
        for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
        {
            tmp_block_data_timed_index = tmp_example_index * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_ptr_block_unit_it = layer_it->ptr_array_block_units;
            
            for(tmp_block_index = 0_UZ; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                              ++tmp_block_index)
            {
                // [0] Output gate.
                tmp_output_gate_summation = ptr_array_summation_input_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Input.
                                                              ptr_array_summation_recurrent_outputs_gates_received[tmp_block_data_timed_index + tmp_block_index] + // Recurrent.
                                                              tmp_ptr_array_output_gate_bias[tmp_block_index];
                
            #ifndef NO_PEEPHOLE
                tmp_ptr_array_peephole_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;

                for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                {
                    tmp_output_gate_summation += tmp_ptr_block_unit_it->ptr_array_cells_states[tmp_cell_data_timed_index + tmp_connection_index] * tmp_ptr_array_peephole_output_gate_parameters[tmp_connection_index];
                }
            #endif

                tmp_ptr_block_unit_it->ptr_summation_outputs_gates[tmp_block_data_timed_index] = tmp_output_gate_summation;

                tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_real(tmp_output_gate_summation);
                // [0] |END| Output gate. |END|

                // [0] Cell output.
                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    if(tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output[tmp_zoneout_mask_index])
                    { tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index] * tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index]; }
                    else
                    { tmp_ptr_cell_unit_it->ptr_cell_output[tmp_cell_data_timed_index] = 0_r; }
                }
                // [0] |END| Cell output. |END|
            }
        }
    }
}

void Model::Forward_Pass__LSTM__States_AF__Loop(long long int const time_step_index_received,
                                                                                              size_t const batch_size,
                                                                                              size_t const layer_block_unit_size_received,
                                                                                              size_t const layer_cell_unit_size_received,
                                                                                              var const *const ptr_array_summation_cell_states_received,
                                                                                              Layer *const layer_it)
{
    size_t tmp_example_index,
              tmp_cell_index,
              tmp_cell_data_timed_index;
    
    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;
    
    ACTIVATION::TYPE const tmp_type_activation_function_io(layer_it->ptr_array_block_units->activation_function_io);

    for(tmp_example_index = 0_UZ; tmp_example_index != batch_size; ++tmp_example_index)
    {
        tmp_cell_data_timed_index = tmp_example_index * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(time_step_index_received);

        // [0] Cells states.        
        for(tmp_cell_index = 0_UZ,
            tmp_ptr_last_cell_unit = layer_it->ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = layer_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                              ++tmp_cell_index)
        {
            AF_FIRE(tmp_type_activation_function_io,
                          ptr_array_summation_cell_states_received[tmp_cell_data_timed_index + tmp_cell_index],
                          tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index]);
        }
        // [0] |END| Cells states. |END|
    }
}
}