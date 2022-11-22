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

#include <omp.h>

namespace DL::v1 {
void Model::RNN__Update_Derivative_Weight_Batch__OpenMP(size_t const batch_size,
                                                                                                           Layer *layer_it,
                                                                                                           Layer const *const last_layer)
{
    size_t tmp_number_units[5];

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
            case LAYER::FULLY_CONNECTED_RECURRENT:
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->Recurrent__Update_Derivative_Weight__FC__OpenMP(batch_size,
                                                                                                        *prev_conn_layer->ptr_number_outputs,
                                                                                                        prev_conn_layer->ptr_array_outputs,
                                                                                                        layer_it);
                    break;
            case LAYER::LSTM:
                tmp_number_units[0] = static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units);
                tmp_number_units[1] = static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units);

                if(layer_it->Use__Bidirectional())
                {
                    tmp_number_units[2] = tmp_number_units[0] >> 1_UZ;
                    tmp_number_units[3] = tmp_number_units[1] >> 1_UZ;

                    this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                                batch_size,
                                                                                                                tmp_number_units[2],
                                                                                                                tmp_number_units[3],
                                                                                                                *prev_conn_layer->ptr_number_outputs,
                                                                                                                prev_conn_layer->ptr_array_outputs,
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                                &layer_it->ptr_Bidirectional_Layer->forward_layer);
                    this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size,
                                                                                                                          tmp_number_units[2],
                                                                                                                          tmp_number_units[3],
                                                                                                                          layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                          layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                          layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                          layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index,
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3],
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3] + tmp_number_units[2],
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3] + 2_UZ * tmp_number_units[2]);
                    this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(false,
                                                                                                                batch_size,
                                                                                                                tmp_number_units[2],
                                                                                                                tmp_number_units[3],
                                                                                                                *prev_conn_layer->ptr_number_outputs,
                                                                                                                prev_conn_layer->ptr_array_outputs,
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                                layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                                &layer_it->ptr_Bidirectional_Layer->backward_layer);
                    this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size,
                                                                                                                          tmp_number_units[2],
                                                                                                                          tmp_number_units[3],
                                                                                                                          layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                          layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                          layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                          layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index,
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3],
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3] + tmp_number_units[2],
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3] + 2_UZ * tmp_number_units[2]);
                }
                else
                {
                    this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                                batch_size,
                                                                                                                tmp_number_units[0],
                                                                                                                tmp_number_units[1],
                                                                                                                *prev_conn_layer->ptr_number_outputs,
                                                                                                                prev_conn_layer->ptr_array_outputs,
                                                                                                                layer_it->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                layer_it->Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                                layer_it->Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                                layer_it->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                layer_it->Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                                layer_it->Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                                layer_it->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                layer_it->Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                                layer_it->Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                                layer_it->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                layer_it->Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                                layer_it->Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                                layer_it);
                    this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size,
                                                                                                                          tmp_number_units[0],
                                                                                                                          tmp_number_units[1],
                                                                                                                          layer_it->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                          layer_it->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                          layer_it->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                          layer_it->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index,
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index + tmp_number_units[1],
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index + tmp_number_units[1] + tmp_number_units[0],
                                                                                                                          this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index + tmp_number_units[1] + 2_UZ * tmp_number_units[0]);
                }
                    break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         layer_it->type_layer,
                                         LAYER_NAME[layer_it->type_layer].c_str());
                    return;
        }
    }
}

void Model::RNN__Update_Derivative_Weight_Batch__Pre_Training__OpenMP(size_t const batch_size)
{
    size_t tmp_number_units[4];

    Layer const *prev_conn_layer;
    Layer *layer_it;
    
    // Coded level part.
    layer_it = this->ptr_array_layers + this->pre_training_level;
    prev_conn_layer = layer_it->previous_connected_layers[0];
    
    switch(layer_it->type_layer)
    {
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Update_Derivative_Weight__FC__OpenMP(batch_size,
                                                                                                    *prev_conn_layer->ptr_number_outputs,
                                                                                                    prev_conn_layer->ptr_array_outputs,
                                                                                                    layer_it);
                break;
        case LAYER::LSTM:
            tmp_number_units[0] = static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units);
            tmp_number_units[1] = static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units);

            if(layer_it->Use__Bidirectional())
            {
                tmp_number_units[2] = tmp_number_units[0] >> 1_UZ;
                tmp_number_units[3] = tmp_number_units[1] >> 1_UZ;

                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                            batch_size,
                                                                                                            tmp_number_units[2],
                                                                                                            tmp_number_units[3],
                                                                                                            *prev_conn_layer->ptr_number_outputs,
                                                                                                            prev_conn_layer->ptr_array_outputs,
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            &layer_it->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size,
                                                                                                                       tmp_number_units[2],
                                                                                                                       tmp_number_units[3],
                                                                                                                       layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                       layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                       layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                       layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                       this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index,
                                                                                                                       this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3],
                                                                                                                       this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3] + tmp_number_units[2],
                                                                                                                       this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3] + 2_UZ * tmp_number_units[2]);
                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(false,
                                                                                                            batch_size,
                                                                                                            tmp_number_units[2],
                                                                                                            tmp_number_units[3],
                                                                                                            *prev_conn_layer->ptr_number_outputs,
                                                                                                            prev_conn_layer->ptr_array_outputs,
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            &layer_it->ptr_Bidirectional_Layer->backward_layer);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size,
                                                                                                                        tmp_number_units[2],
                                                                                                                        tmp_number_units[3],
                                                                                                                        layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                        layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                        layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                        layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index,
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3],
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3] + tmp_number_units[2],
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3] + 2_UZ * tmp_number_units[2]);
            }
            else
            {
                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                            batch_size,
                                                                                                            tmp_number_units[0],
                                                                                                            tmp_number_units[1],
                                                                                                            *prev_conn_layer->ptr_number_outputs,
                                                                                                            prev_conn_layer->ptr_array_outputs,
                                                                                                            layer_it->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            layer_it->Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            layer_it->Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            layer_it->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            layer_it->Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            layer_it->Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            layer_it->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            layer_it->Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            layer_it->Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            layer_it->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            layer_it->Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            layer_it->Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            layer_it);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size,
                                                                                                                        tmp_number_units[0],
                                                                                                                        tmp_number_units[1],
                                                                                                                        layer_it->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                        layer_it->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                        layer_it->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                        layer_it->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index,
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index + tmp_number_units[1],
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index + tmp_number_units[1] + tmp_number_units[0],
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index + tmp_number_units[1] + 2_UZ * tmp_number_units[0]);
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
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Update_Derivative_Weight__FC__OpenMP(batch_size,
                                                                                                    *prev_conn_layer->ptr_number_outputs,
                                                                                                    prev_conn_layer->ptr_array_outputs,
                                                                                                    layer_it);
                break;
        case LAYER::LSTM:
            tmp_number_units[0] = static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units);
            tmp_number_units[1] = static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units);

            if(layer_it->Use__Bidirectional())
            {
                tmp_number_units[2] = tmp_number_units[0] >> 1_UZ;
                tmp_number_units[3] = tmp_number_units[1] >> 1_UZ;

                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                            batch_size,
                                                                                                            tmp_number_units[2],
                                                                                                            tmp_number_units[3],
                                                                                                            *prev_conn_layer->ptr_number_outputs,
                                                                                                            prev_conn_layer->ptr_array_outputs,
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->forward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            &layer_it->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size,
                                                                                                                       tmp_number_units[2],
                                                                                                                       tmp_number_units[3],
                                                                                                                       layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                       layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                       layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                       layer_it->ptr_Bidirectional_Layer->forward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                       this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index,
                                                                                                                       this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3],
                                                                                                                       this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3] + tmp_number_units[2],
                                                                                                                       this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->forward_layer.first_bias_connection_index + tmp_number_units[3] + 2_UZ * tmp_number_units[2]);
                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(false,
                                                                                                            batch_size,
                                                                                                            tmp_number_units[2],
                                                                                                            tmp_number_units[3],
                                                                                                            *prev_conn_layer->ptr_number_outputs,
                                                                                                            prev_conn_layer->ptr_array_outputs,
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            layer_it->ptr_Bidirectional_Layer->backward_layer.Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            &layer_it->ptr_Bidirectional_Layer->backward_layer);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size,
                                                                                                                        tmp_number_units[2],
                                                                                                                        tmp_number_units[3],
                                                                                                                        layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                        layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                        layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                        layer_it->ptr_Bidirectional_Layer->backward_layer.ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index,
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3],
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3] + tmp_number_units[2],
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->ptr_Bidirectional_Layer->backward_layer.first_bias_connection_index + tmp_number_units[3] + 2_UZ * tmp_number_units[2]);
            }
            else
            {
                this->Recurrent__Update_Derivative_Weight__LSTM__OpenMP(true,
                                                                                                            batch_size,
                                                                                                            tmp_number_units[0],
                                                                                                            tmp_number_units[1],
                                                                                                            *prev_conn_layer->ptr_number_outputs,
                                                                                                            prev_conn_layer->ptr_array_outputs,
                                                                                                            layer_it->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                            layer_it->Get__Array_Deltas__Cell__Block_Input__Input(),
                                                                                                            layer_it->Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                                            layer_it->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                            layer_it->Get__Array_Deltas__Block__Input_Gate__Input(),
                                                                                                            layer_it->Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                                            layer_it->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                            layer_it->Get__Array_Deltas__Block__Forget_Gate__Input(),
                                                                                                            layer_it->Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                                            layer_it->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                            layer_it->Get__Array_Deltas__Block__Output_Gate__Input(),
                                                                                                            layer_it->Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                                            layer_it);
                this->Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(batch_size,
                                                                                                                        tmp_number_units[0],
                                                                                                                        tmp_number_units[1],
                                                                                                                        layer_it->ptr_array_cell_units->ptr_delta_cell_input,
                                                                                                                        layer_it->ptr_array_block_units->ptr_delta_inputs_gates,
                                                                                                                        layer_it->ptr_array_block_units->ptr_delta_forgets_gates,
                                                                                                                        layer_it->ptr_array_block_units->ptr_delta_outputs_gates,
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index,
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index + tmp_number_units[1],
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index + tmp_number_units[1] + tmp_number_units[0],
                                                                                                                        this->ptr_array_derivatives_parameters + layer_it->first_bias_connection_index + tmp_number_units[1] + 2_UZ * tmp_number_units[0]);
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

void Model::Recurrent__Update_Derivative_Weight__FC__OpenMP(size_t const batch_size,
                                                                                                               size_t const input_unit_size_received,
                                                                                                               var const *const ptr_array_inputs_received,
                                                                                                               Layer *const layer_it)
{
    for(size_t tmp_time_step_index(0_UZ); tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Update_Derivative_Weight__FC__OpenMP(tmp_time_step_index,
                                                                               batch_size,
                                                                               input_unit_size_received,
                                                                               ptr_array_inputs_received,
                                                                               layer_it);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Update_Derivative_Weight__FC_Ind_RNN__OpenMP(size_t const time_step_index_received,
                                                                                                            size_t const batch_size,
                                                                                                            size_t const derivative_size_received,
                                                                                                            var const *const ptr_array_inputs_received,
                                                                                                            real const *const ptr_array_derivative_inputs_received,
                                                                                                            real *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_derivative_previous_timed_batched_index(this->batch_size * derivative_size_received * (time_step_index_received - 1_UZ)),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_derivative_index;
    
    var const *tmp_ptr_array_previous_timed_inputs;
    real const *tmp_ptr_array_derivative_inputs;
    real *tmp_ptr_array_derivatives;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_previous_timed_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_previous_timed_batched_index;
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(omp_get_thread_num()) * this->total_parameters_allocated;

        for(tmp_derivative_index = 0_UZ; tmp_derivative_index != derivative_size_received; ++tmp_derivative_index) { tmp_ptr_array_derivatives[tmp_derivative_index] += cast(tmp_ptr_array_previous_timed_inputs[tmp_derivative_index]) * tmp_ptr_array_derivative_inputs[tmp_derivative_index]; }
    }
}

void Model::Recurrent__Update_Derivative_Weight__LSTM__OpenMP(bool const forward_layer_received,
                                                                                                                    size_t const batch_size,
                                                                                                                    size_t const block_unit_size_received,
                                                                                                                    size_t const cell_unit_size_received,
                                                                                                                    size_t const input_unit_size_received,
                                                                                                                    var const *const ptr_array_inputs_received,
                                                                                                                    real const *const ptr_array_delta_block_inputs_received,
                                                                                                                    real const *const ptr_array_delta_input_block_inputs_received,
                                                                                                                    real const *const ptr_array_delta_recurrent_block_inputs_received,
                                                                                                                    real const *const ptr_array_delta_input_gates_received,
                                                                                                                    real const *const ptr_array_delta_input_input_gates_received,
                                                                                                                    real const *const ptr_array_delta_recurrent_input_gates_received,
                                                                                                                    real const *const ptr_array_delta_forget_gates_received,
                                                                                                                    real const *const ptr_array_delta_input_forget_gates_received,
                                                                                                                    real const *const ptr_array_delta_recurrent_forget_gates_received,
                                                                                                                    real const *const ptr_array_delta_output_gates_received,
                                                                                                                    real const *const ptr_array_delta_input_output_gates_received,
                                                                                                                    real const *const ptr_array_delta_recurrent_output_gates_received,
                                                                                                                    Layer *const layer_it)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_number_inputs_connections(layer_it->ptr_array_block_units->last_index_feedforward_connection_input_gate - layer_it->ptr_array_block_units->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrents_connection(layer_it->ptr_array_block_units->last_index_recurrent_connection_input_gate - layer_it->ptr_array_block_units->first_index_recurrent_connection_input_gate),
                       tmp_number_peepholes_connections(layer_it->ptr_array_block_units->last_index_peephole_input_gate - layer_it->ptr_array_block_units->first_index_peephole_input_gate);
    size_t tmp_thread_index,
              tmp_time_step_direction_direction,
              tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_block_data_direction_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_direction_timed_index;
    
    var const *tmp_ptr_array_previous_layer_outputs,
                  *tmp_ptr_array_layer_timed_outputs,
                  *tmp_ptr_array_cells_states;
    real *tmp_ptr_array_cell_input_derivatives_parameters,
         *tmp_ptr_array_input_gate_derivatives_parameters,
         *tmp_ptr_array_forget_gate_derivatives_parameters,
         *tmp_ptr_array_output_gate_derivatives_parameters,
         tmp_cell_state,
         tmp_cell_input_error,
         tmp_input_gate_error,
         tmp_forget_gate_error,
         tmp_output_gate_error;

    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it;

    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;
    
    long long int tmp_time_step_index,
                       tmp_time_step_start(forward_layer_received ? 0ll : static_cast<long long int>(this->seq_w - 1_UZ)),
                       tmp_time_step_end(forward_layer_received ? static_cast<long long int>(this->seq_w) : -1ll),
                       tmp_time_prediction_end(forward_layer_received ? static_cast<long long int>(this->seq_w - 1_UZ) : 0ll);
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index = static_cast<size_t>(omp_get_thread_num());

        for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? ++tmp_time_step_index : --tmp_time_step_index)
        {
            if(tmp_time_step_index != tmp_time_prediction_end)
            {
                tmp_time_step_direction_direction = forward_layer_received ? static_cast<size_t>(tmp_time_step_index + 1ll) : static_cast<size_t>(tmp_time_step_index - 1ll);
                
                tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                tmp_block_data_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * tmp_time_step_direction_direction;

                tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                tmp_cell_data_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * tmp_time_step_direction_direction;
                
                tmp_ptr_array_previous_layer_outputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_unit_size_received + this->batch_size * input_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                tmp_ptr_array_layer_timed_outputs = layer_it->ptr_array_cell_units->ptr_cell_output + tmp_cell_data_timed_index;
                
                for(tmp_cell_index = 0_UZ,
                    tmp_block_index = 0_UZ,
                    tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                                                                                                 ++tmp_block_index)
                {
                    // [0] Cells inputs.
                    for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                        tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                           ++tmp_cell_index)
                    {
                        // Cell inputs.
                        tmp_cell_input_error = ptr_array_delta_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index];
                        
                        tmp_ptr_array_cell_input_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input + tmp_thread_index * this->total_parameters_allocated;

                        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                        {
                            tmp_ptr_array_cell_input_derivatives_parameters[tmp_connection_index] += tmp_cell_input_error * cast(tmp_ptr_array_previous_layer_outputs[tmp_connection_index]);
                        }
                        // |END| Cell inputs. |END|

                        // Cell recurrents.
                        tmp_cell_input_error = ptr_array_delta_recurrent_block_inputs_received[tmp_cell_data_direction_timed_index + tmp_cell_index];
                        
                        tmp_ptr_array_cell_input_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input + tmp_thread_index * this->total_parameters_allocated;

                        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
                        {
                            tmp_ptr_array_cell_input_derivatives_parameters[tmp_connection_index] += tmp_cell_input_error * cast(tmp_ptr_array_layer_timed_outputs[tmp_connection_index]);
                        }
                        // |END| Cell recurrents. |END|
                    }
                    // [0] |END| Cells inputs. |END|
                    
                    // [0] Gates-inputs.
                    tmp_input_gate_error = ptr_array_delta_input_input_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    tmp_forget_gate_error = ptr_array_delta_input_forget_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    tmp_output_gate_error = ptr_array_delta_input_output_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    
                    tmp_ptr_array_input_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_forget_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_output_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate + tmp_thread_index * this->total_parameters_allocated;

                    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                    {
                        tmp_ptr_array_input_gate_derivatives_parameters[tmp_connection_index] += tmp_input_gate_error * cast(tmp_ptr_array_previous_layer_outputs[tmp_connection_index]);
                        tmp_ptr_array_forget_gate_derivatives_parameters[tmp_connection_index] += tmp_forget_gate_error * cast(tmp_ptr_array_previous_layer_outputs[tmp_connection_index]);
                        tmp_ptr_array_output_gate_derivatives_parameters[tmp_connection_index] += tmp_output_gate_error * cast(tmp_ptr_array_previous_layer_outputs[tmp_connection_index]);
                    }

                    // [0] Output gate, peepholes.
                    tmp_input_gate_error = ptr_array_delta_recurrent_input_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];
                    tmp_forget_gate_error = ptr_array_delta_recurrent_forget_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];

                #ifndef NO_PEEPHOLE
                    tmp_ptr_array_cells_states = tmp_ptr_block_unit_it->ptr_array_cells_states + tmp_cell_data_timed_index;
                    
                    tmp_ptr_array_input_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_forget_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_output_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate + tmp_thread_index * this->total_parameters_allocated;

                    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                    {
                        tmp_cell_state = cast(tmp_ptr_array_cells_states[tmp_connection_index]);

                        tmp_ptr_array_input_gate_derivatives_parameters[tmp_connection_index] += tmp_input_gate_error * tmp_cell_state;
                        tmp_ptr_array_forget_gate_derivatives_parameters[tmp_connection_index] += tmp_forget_gate_error * tmp_cell_state;
                        tmp_ptr_array_output_gate_derivatives_parameters[tmp_connection_index] += tmp_output_gate_error * tmp_cell_state;
                    }
                #endif
                    // [0] |END| Output gate, peepholes. |END|
                    // [0] |END| Gates-inputs. |END|

                    // [0] Gates-recurrents.
                    tmp_output_gate_error = ptr_array_delta_recurrent_output_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];
                    
                    tmp_ptr_array_input_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_forget_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_output_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate + tmp_thread_index * this->total_parameters_allocated;

                    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
                    {
                        tmp_ptr_array_input_gate_derivatives_parameters[tmp_connection_index] += tmp_input_gate_error * cast(tmp_ptr_array_layer_timed_outputs[tmp_connection_index]);
                        tmp_ptr_array_forget_gate_derivatives_parameters[tmp_connection_index] += tmp_forget_gate_error * cast(tmp_ptr_array_layer_timed_outputs[tmp_connection_index]);
                        tmp_ptr_array_output_gate_derivatives_parameters[tmp_connection_index] += tmp_output_gate_error * cast(tmp_ptr_array_layer_timed_outputs[tmp_connection_index]);
                    }
                    // [0] |END| Gates-recurrents. |END|
                }
            }
            else
            {
                tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                tmp_ptr_array_previous_layer_outputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_unit_size_received + this->batch_size * input_unit_size_received * static_cast<size_t>(tmp_time_step_index);

                for(tmp_cell_index = 0_UZ,
                    tmp_block_index = 0_UZ,
                    tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                                                                                                 ++tmp_block_index)
                {
                    // [0] Cells inputs.
                    for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                        tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                           ++tmp_cell_index)
                    {
                        // Cell inputs.
                        tmp_cell_input_error = ptr_array_delta_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index];
                        
                        tmp_ptr_array_cell_input_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input + tmp_thread_index * this->total_parameters_allocated;

                        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                        {
                            tmp_ptr_array_cell_input_derivatives_parameters[tmp_connection_index] += tmp_cell_input_error * cast(tmp_ptr_array_previous_layer_outputs[tmp_connection_index]);
                        }
                        // |END| Cell inputs. |END|
                    }
                    // [0] |END| Cells inputs. |END|
                    
                    // [0] Gates-inputs.
                    tmp_input_gate_error = ptr_array_delta_input_input_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    tmp_forget_gate_error = ptr_array_delta_input_forget_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    tmp_output_gate_error = ptr_array_delta_input_output_gates_received[tmp_block_data_timed_index + tmp_block_index];
                    
                    tmp_ptr_array_input_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_forget_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate + tmp_thread_index * this->total_parameters_allocated;
                    tmp_ptr_array_output_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate + tmp_thread_index * this->total_parameters_allocated;

                    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
                    {
                        tmp_ptr_array_input_gate_derivatives_parameters[tmp_connection_index] += tmp_input_gate_error * cast(tmp_ptr_array_previous_layer_outputs[tmp_connection_index]);
                        tmp_ptr_array_forget_gate_derivatives_parameters[tmp_connection_index] += tmp_forget_gate_error * cast(tmp_ptr_array_previous_layer_outputs[tmp_connection_index]);
                        tmp_ptr_array_output_gate_derivatives_parameters[tmp_connection_index] += tmp_output_gate_error * cast(tmp_ptr_array_previous_layer_outputs[tmp_connection_index]);
                    }

                    // [0] Output gate, peepholes.
                #ifndef NO_PEEPHOLE
                    tmp_ptr_array_cells_states = tmp_ptr_block_unit_it->ptr_array_cells_states + tmp_cell_data_timed_index;
                    
                    tmp_ptr_array_output_gate_derivatives_parameters = this->ptr_array_derivatives_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate + tmp_thread_index * this->total_parameters_allocated;

                    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_peepholes_connections; ++tmp_connection_index)
                    {
                        tmp_ptr_array_output_gate_derivatives_parameters[tmp_connection_index] += tmp_output_gate_error * cast(tmp_ptr_array_cells_states[tmp_connection_index]);
                    }
                #endif
                    // [0] |END| Output gate, peepholes. |END|
                    // [0] |END| Gates-inputs. |END|
                }
            }
        }
    }
}

void Model::Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(size_t const batch_size,
                                                                                                                              size_t const layer_block_unit_size_received,
                                                                                                                              size_t const layer_cell_unit_size_received,
                                                                                                                              real const *const ptr_array_delta_block_inputs_received,
                                                                                                                              real const *const ptr_array_delta_input_gates_received,
                                                                                                                              real const *const ptr_array_delta_forget_gates_received,
                                                                                                                              real const *const ptr_array_delta_output_gates_received,
                                                                                                                              real *const ptr_array_cell_input_derivatives_bias_received,
                                                                                                                              real *const ptr_array_input_gate_derivatives_bias_received,
                                                                                                                              real *const ptr_array_forget_gate_derivatives_bias_received,
                                                                                                                              real *const ptr_array_output_gate_derivatives_bias_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t tmp_thread_index,
              tmp_time_step_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_block_index,
              tmp_block_data_timed_index;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index = static_cast<size_t>(omp_get_thread_num());

        for(tmp_time_step_index = 0_UZ; tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
        {
            tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * layer_cell_unit_size_received + this->batch_size * layer_cell_unit_size_received * static_cast<size_t>(tmp_time_step_index);

            tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * layer_block_unit_size_received + this->batch_size * layer_block_unit_size_received * static_cast<size_t>(tmp_time_step_index);

            for(tmp_cell_index = 0_UZ; tmp_cell_index != layer_cell_unit_size_received; ++tmp_cell_index)
            {
                ptr_array_cell_input_derivatives_bias_received[tmp_thread_index * this->total_parameters_allocated + tmp_cell_index] += ptr_array_delta_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index];
            }

            for(tmp_block_index = 0_UZ; tmp_block_index != layer_block_unit_size_received; ++tmp_block_index)
            {
                ptr_array_input_gate_derivatives_bias_received[tmp_thread_index * this->total_parameters_allocated + tmp_block_index] += ptr_array_delta_input_gates_received[tmp_block_data_timed_index + tmp_block_index];
                ptr_array_forget_gate_derivatives_bias_received[tmp_thread_index * this->total_parameters_allocated + tmp_block_index] += ptr_array_delta_forget_gates_received[tmp_block_data_timed_index + tmp_block_index];
                ptr_array_output_gate_derivatives_bias_received[tmp_thread_index * this->total_parameters_allocated + tmp_block_index] += ptr_array_delta_output_gates_received[tmp_block_data_timed_index + tmp_block_index];
            }
        }
    }
}
}
