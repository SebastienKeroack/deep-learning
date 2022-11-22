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

namespace DL::v1 {
bool Model::Set__Pre_Training_Level(size_t const pre_training_level_received)
{
    if(this->pre_training_level == pre_training_level_received) { return true; }
    else if(pre_training_level_received > (this->total_layers - 3_UZ) / 2_UZ + 1_UZ)
    {
        ERR(L"Pre training level (%zu) overflow (%zu).",
                                 pre_training_level_received,
                                 (this->total_layers - 3_UZ) / 2_UZ + 1_UZ);

        return false;
    }

    size_t const tmp_past_pre_training_level(this->pre_training_level);

    this->pre_training_level = pre_training_level_received;

    if((tmp_past_pre_training_level == 0_UZ && pre_training_level_received != 0_UZ)
      ||
      (tmp_past_pre_training_level != 0_UZ && pre_training_level_received == 0_UZ))
    { this->Order__Layers__Output(); }

    if(this->Use__Regularization_Parameter())
    {
        if(this->pre_training_level != 0_UZ) { this->Indexing_Regularization_Parameters__Pre_training(); }
        else { this->Indexing_Regularization_Parameters(); }
    }

    return true;
}

void Model::Indexing_Regularization_Parameters(void)
{
    Layer const *const last_layer(this->ptr_last_layer),
                               *layer_it(this->ptr_array_layers + 1);
    
    for(; layer_it != last_layer; ++layer_it)
    {
        switch(layer_it->type_layer)
        {
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(1_r, layer_it);
            case LAYER::FULLY_CONNECTED:
            case LAYER::FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(1_r, layer_it); break;
            case LAYER::LSTM: this->Indexing_Regularization__Weights__LSTM(1_r, layer_it); break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         layer_it->type_layer,
                                         LAYER_NAME[layer_it->type_layer].c_str());
                    break;
        }

        this->Indexing_Regularization__Bias(0_r, layer_it);
    }
}

void Model::Indexing_Regularization_Parameters__Pre_training(void)
{
    if(this->pre_training_level == 0_UZ)
    {
        ERR(L"The neural network use the pre-training function without the mode pre-training activate.",);

        return;
    }

    Layer const *const last_layer(this->ptr_last_layer),
                               *const tmp_ptr_input_layer(this->ptr_array_layers + this->pre_training_level),
                               *const tmp_ptr_output_layer(this->Get__Output_Layer()),
                               *layer_it(this->ptr_array_layers + 1);
    
    // First layer to coded layer, Mask zero.
    for(; layer_it < tmp_ptr_input_layer; ++layer_it)
    {
        switch(layer_it->type_layer)
        {
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(0_r, layer_it);
            case LAYER::FULLY_CONNECTED:
            case LAYER::FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(0_r, layer_it); break;
            case LAYER::LSTM: this->Indexing_Regularization__Weights__LSTM(0_r, layer_it); break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         layer_it->type_layer,
                                         LAYER_NAME[layer_it->type_layer].c_str());
                    break;
        }

        this->Indexing_Regularization__Bias(0_r, layer_it);
    }
    
    // Coded layer, Mask one.
    switch(layer_it->type_layer)
    {
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(1_r, layer_it);
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(1_r, layer_it); break;
        case LAYER::LSTM: this->Indexing_Regularization__Weights__LSTM(1_r, layer_it); break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                                        layer_it->type_layer,
                                        LAYER_NAME[layer_it->type_layer].c_str());
                break;
    }

    this->Indexing_Regularization__Bias(0_r, layer_it);
    // |END| Coded layer, Mask one. |END|
    
    // Coded layer to output layer, Mask zero.
    for(++layer_it; layer_it < tmp_ptr_output_layer; ++layer_it)
    {
        switch(layer_it->type_layer)
        {
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(0_r, layer_it);
            case LAYER::FULLY_CONNECTED:
            case LAYER::FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(0_r, layer_it); break;
            case LAYER::LSTM: this->Indexing_Regularization__Weights__LSTM(0_r, layer_it); break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         layer_it->type_layer,
                                         LAYER_NAME[layer_it->type_layer].c_str());
                    break;
        }

        this->Indexing_Regularization__Bias(0_r, layer_it);
    }
    
    // Output layer, Mask one.
    switch(layer_it->type_layer)
    {
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(1_r, layer_it);
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(1_r, layer_it); break;
        case LAYER::LSTM: this->Indexing_Regularization__Weights__LSTM(1_r, layer_it); break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                                        layer_it->type_layer,
                                        LAYER_NAME[layer_it->type_layer].c_str());
                break;
    }

    this->Indexing_Regularization__Bias(0_r, layer_it);
    // |END| Output layer, Mask one. |END|
    
    // Output layer to last layer, Mask zero.
    for(++layer_it; layer_it < last_layer; ++layer_it)
    {
        switch(layer_it->type_layer)
        {
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Indexing_Regularization__Weights__AF_Ind_Recurrent(0_r, layer_it);
            case LAYER::FULLY_CONNECTED:
            case LAYER::FULLY_CONNECTED_RECURRENT: this->Indexing_Regularization__Weights__FC__Forward(0_r, layer_it); break;
            case LAYER::LSTM: this->Indexing_Regularization__Weights__LSTM(0_r, layer_it); break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         layer_it->type_layer,
                                         LAYER_NAME[layer_it->type_layer].c_str());
                    break;
        }

        this->Indexing_Regularization__Bias(0_r, layer_it);
    }
}

void Model::Indexing_Regularization__Weights__FC__Forward(real const mask_received, Layer const *const layer_it)
{
    Neuron_unit const *const tmp_ptr_last_neuron_unit(layer_it->ptr_last_neuron_unit),
                                         *tmp_ptr_neuron_unit_it(layer_it->ptr_array_neuron_units);

    real const *tmp_ptr_last_mask_regularization;
    real *tmp_ptr_mask_regularization_it;
    
    for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
    {
        tmp_ptr_mask_regularization_it = this->ptr_array_mask_regularized_parameters + *tmp_ptr_neuron_unit_it->ptr_first_connection_index;
        tmp_ptr_last_mask_regularization = this->ptr_array_mask_regularized_parameters + *tmp_ptr_neuron_unit_it->ptr_last_connection_index;

        for(; tmp_ptr_mask_regularization_it != tmp_ptr_last_mask_regularization; ++tmp_ptr_mask_regularization_it) { *tmp_ptr_mask_regularization_it = mask_received; }
    }
}

void Model::Indexing_Regularization__Weights__AF_Ind_Recurrent(real const mask_received, Layer const *const layer_it)
{
    AF_Ind_recurrent_unit const *const tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit(layer_it->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_number_units(static_cast<size_t>(layer_it->ptr_last_AF_Ind_recurrent_unit - tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit));

    real *tmp_ptr_array_mask_regularization_it(this->ptr_array_mask_regularized_parameters + *tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index);
    real const *const tmp_ptr_array_mask_regularization_end(tmp_ptr_array_mask_regularization_it + tmp_number_units);
    
    for(; tmp_ptr_array_mask_regularization_it != tmp_ptr_array_mask_regularization_end; ++tmp_ptr_array_mask_regularization_it) { *tmp_ptr_array_mask_regularization_it = mask_received; }
}

void Model::Indexing_Regularization__Weights__LSTM(real const mask_received, Layer const *const layer_it)
{
    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit),
                                      *tmp_ptr_block_unit_it(layer_it->ptr_array_block_units);
    
    size_t const tmp_number_peephole_connections(tmp_ptr_block_unit_it->last_index_peephole_input_gate - tmp_ptr_block_unit_it->first_index_peephole_input_gate),
                       tmp_number_feedforward_connections(tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrent_connections(tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate - tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate);
    size_t tmp_connection_index;
    
    CellUnit const *tmp_ptr_block_ptr_last_cell_unit,
                                    *tmp_ptr_block_ptr_cell_unit_it;

    real *tmp_ptr_array_cell_input_regularized_connections,
         *tmp_ptr_array_input_gate_regularized_connections,
         *tmp_ptr_array_forget_gate_regularized_connections,
         *tmp_ptr_array_output_gate_regularized_connections;

    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        // [0] Cell input.
        for(tmp_ptr_block_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
            tmp_ptr_block_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
        {
            //    [1] Input, cell input.
            tmp_ptr_array_cell_input_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;
            
            for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index) { tmp_ptr_array_cell_input_regularized_connections[tmp_connection_index] = mask_received; }
            //    [1] |END| Input, cell input. |END|
            
            //    [1] Recurrent, cell input.
            tmp_ptr_array_cell_input_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;
            
            for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index) { tmp_ptr_array_cell_input_regularized_connections[tmp_connection_index] = mask_received; }
            //    [1] |END| Recurrent, cell input. |END|

        }
        // [0] |END| Cell input. |END|
        
        // Input, gates.
        tmp_ptr_array_input_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
        tmp_ptr_array_forget_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
        tmp_ptr_array_output_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;
        
        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_feedforward_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_forget_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_output_gate_regularized_connections[tmp_connection_index] = mask_received;
        }
        // |END| Input, gates. |END|
        
        // Recurrent, gates.
        tmp_ptr_array_input_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;
        tmp_ptr_array_forget_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;
        tmp_ptr_array_output_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;
        
        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrent_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_forget_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_output_gate_regularized_connections[tmp_connection_index] = mask_received;
        }
        // |END| Recurrent, gates. |END|
        
    #ifndef NO_PEEPHOLE
        // [0] Peepholes.
        tmp_ptr_array_input_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_peephole_input_gate;
        tmp_ptr_array_forget_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_peephole_forget_gate;
        tmp_ptr_array_output_gate_regularized_connections = this->ptr_array_mask_regularized_parameters + tmp_ptr_block_unit_it->first_index_peephole_output_gate;
        
        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_peephole_connections; ++tmp_connection_index)
        {
            tmp_ptr_array_input_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_forget_gate_regularized_connections[tmp_connection_index] = mask_received;
            tmp_ptr_array_output_gate_regularized_connections[tmp_connection_index] = mask_received;
        }
        // [0] |END| Peepholes. |END|
    #endif
    }
}

void Model::Indexing_Regularization__Bias(real const mask_received, Layer const *const layer_it)
{
    real const *const tmp_ptr_array_mask_regularization_end(this->ptr_array_mask_regularized_parameters + layer_it->last_bias_connection_index);
    real *tmp_ptr_array_mask_regularization_it(this->ptr_array_mask_regularized_parameters + layer_it->first_bias_connection_index);
    
    for(; tmp_ptr_array_mask_regularization_it != tmp_ptr_array_mask_regularization_end; ++tmp_ptr_array_mask_regularization_it) { *tmp_ptr_array_mask_regularization_it = mask_received; }
}
}
