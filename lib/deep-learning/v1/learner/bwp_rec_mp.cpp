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

#include "deep-learning/v1/learner/model.hpp"
#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/ops/math.hpp"
#include "deep-learning/v1/ops/activations/functions.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"

#include <omp.h>

namespace DL::v1 {
void Model::RNN__Backward_Pass_Batch__OpenMP(size_t const batch_size)
{
    size_t tmp_layer_number_outputs;
    
    real *tmp_ptr_array_layer_gradients;
    
    Layer const *const tmp_ptr_second_layer(this->ptr_array_layers + 1);

    Layer const *next_layer_end,
                               *next_layer_it;
    Layer *tmp_ptr_gradient_layer_it(this->ptr_last_layer - 1),
                      *layer_it;

    // If the network use normalization.
    #pragma omp single
    if(this->Use__Normalization())
    {
        memset(this->ptr_array_normalized_batch_units_derivatives_means,
                    0,
             this->number_threads * this->total_normalized_units_allocated *
                 this->seq_w * sizeof(real));
        memset(this->ptr_array_normalized_batch_units_derivatives_variances,
                    0,
                    this->number_threads * this->total_normalized_units_allocated * this->seq_w * sizeof(real));
    }

    // Loop through each layer and do a backward propagation.
    for(; tmp_ptr_gradient_layer_it != tmp_ptr_second_layer; --tmp_ptr_gradient_layer_it)
    {
        layer_it = this->ptr_array_layers + static_cast<size_t>(tmp_ptr_gradient_layer_it->previous_connected_layers[0] - this->ptr_array_layers);
        
        // clear past error(s).
        tmp_layer_number_outputs = *layer_it->ptr_number_outputs;

        tmp_ptr_array_layer_gradients = layer_it->ptr_array_derivative_outputs;

        #pragma omp single
        memset(tmp_ptr_array_layer_gradients,
                     0,
               this->batch_size * tmp_layer_number_outputs * this->seq_w *
                   sizeof(real));
        // |END| clear past error(s). |END|
        
        // Propagate the error(s) to the layer.
        for(next_layer_it = layer_it->next_connected_layers[0],
            next_layer_end = next_layer_it + layer_it->next_connected_layers.size(); next_layer_it != next_layer_end; ++next_layer_it)
        {
            switch(next_layer_it->type_layer)
            {
                case LAYER::AVERAGE_POOLING:
                    this->Recurrent__Backward_Pass__Average_Pooling__OpenMP(batch_size,
                                                                                                                  tmp_layer_number_outputs,
                                                                                                                  tmp_ptr_array_layer_gradients,
                                                                                                                  next_layer_it);
                        break;
                case LAYER::FULLY_CONNECTED:
                case LAYER::FULLY_CONNECTED_RECURRENT:
                case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    this->Recurrent__Backward_Pass__FC__OpenMP(batch_size,
                                                                                              tmp_layer_number_outputs,
                                                                                              tmp_ptr_array_layer_gradients,
                                                                                              next_layer_it);
                        break;
                case LAYER::LSTM:
                    this->Recurrent__Backward_Pass__LSTM__OpenMP(batch_size,
                                                                                                  tmp_layer_number_outputs,
                                                                                                  tmp_ptr_array_layer_gradients,
                                                                                                  next_layer_it);
                        break;
                case LAYER::MAX_POOLING:
                    this->Recurrent__Backward_Pass__Max_Pooling__OpenMP(batch_size,
                                                                                                             tmp_layer_number_outputs,
                                                                                                             tmp_ptr_array_layer_gradients,
                                                                                                             next_layer_it);
                        break;
                case LAYER::RESIDUAL:
                    this->Recurrent__Backward_Pass__Residual__OpenMP(batch_size,
                                                                                                 tmp_layer_number_outputs,
                                                                                                 tmp_ptr_array_layer_gradients,
                                                                                                 next_layer_it);
                        break;
                default:
                    ERR(L"Layer type (%d | %ls) is not managed in",
                                             next_layer_it->type_layer,
                                             LAYER_NAME[next_layer_it->type_layer].c_str());
                        return;
            }
        }
        // |END| Propagate the error(s) to the layer. |END|

        // Compute the gradients.
        switch(layer_it->type_layer)
        {
            case LAYER::AVERAGE_POOLING:
            case LAYER::MAX_POOLING: break;
            case LAYER::FULLY_CONNECTED:
            case LAYER::FULLY_CONNECTED_RECURRENT:
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Recurrent__Backward_Pass__Gradient__FC__OpenMP(batch_size, layer_it); break;
            case LAYER::LSTM:
                if(layer_it->Use__Bidirectional())
                {
                    this->Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(true,
                                                                                                                  batch_size,
                                                                                                                  tmp_layer_number_outputs,
                                                                                                                  tmp_ptr_array_layer_gradients,
                                                                                                                  &layer_it->ptr_Bidirectional_Layer->forward_layer);
                    this->Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(false,
                                                                                                                  batch_size,
                                                                                                                  tmp_layer_number_outputs,
                                                                                                                  tmp_ptr_array_layer_gradients,
                                                                                                                  &layer_it->ptr_Bidirectional_Layer->backward_layer);
                }
                else
                {
                    this->Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(true,
                                                                                                                  batch_size,
                                                                                                                  tmp_layer_number_outputs,
                                                                                                                  tmp_ptr_array_layer_gradients,
                                                                                                                  layer_it);
                }
                    break;
            case LAYER::RESIDUAL:
                this->Recurrent__Backward_Pass__Gradient__Residual__OpenMP(batch_size, layer_it);

                tmp_ptr_gradient_layer_it = layer_it + 1;
                    break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         layer_it->type_layer,
                                         LAYER_NAME[layer_it->type_layer].c_str());
                    return;
        }
        // |END| Compute the gradients. |END|
    }
}

void Model::RNN__Backward_Pass_Batch__Pre_Training__OpenMP(size_t const batch_size)
{
    size_t tmp_layer_number_outputs;
    
    real *tmp_ptr_array_layer_gradients;
    
    Layer *const tmp_ptr_coded_layer(this->ptr_array_layers + this->pre_training_level);
    Layer const *const tmp_ptr_decoded_layer(this->ptr_last_layer - static_cast<size_t>(tmp_ptr_coded_layer - this->ptr_array_layers));
    
    // If the network use normalization.
    #pragma omp single
    if(this->Use__Normalization())
    {
        memset(this->ptr_array_normalized_batch_units_derivatives_means,
                    0,
                    this->number_threads * this->total_normalized_units_allocated * this->seq_w * sizeof(real));
        memset(this->ptr_array_normalized_batch_units_derivatives_variances,
                    0,
                    this->number_threads * this->total_normalized_units_allocated * this->seq_w * sizeof(real));
    }

    // clear past error(s).
    tmp_layer_number_outputs = *tmp_ptr_coded_layer->ptr_number_outputs;

    tmp_ptr_array_layer_gradients = tmp_ptr_coded_layer->ptr_array_derivative_outputs;

    #pragma omp single
    memset(tmp_ptr_array_layer_gradients,
                   0,
                   this->batch_size * tmp_layer_number_outputs * this->seq_w * sizeof(real));
    // |END| clear past error(s). |END|
    
    // Propagate the error(s) to the layer.
    switch(tmp_ptr_decoded_layer->type_layer)
    {
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case LAYER::FULLY_CONNECTED_RECURRENT:
            this->Recurrent__Backward_Pass__FC__OpenMP(batch_size,
                                                                                      tmp_layer_number_outputs,
                                                                                      tmp_ptr_array_layer_gradients,
                                                                                      tmp_ptr_decoded_layer);
                break;
        case LAYER::LSTM:
            this->Recurrent__Backward_Pass__LSTM__OpenMP(batch_size,
                                                                                          tmp_layer_number_outputs,
                                                                                          tmp_ptr_array_layer_gradients,
                                                                                          tmp_ptr_decoded_layer);
                break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                                     tmp_ptr_decoded_layer->type_layer,
                                     LAYER_NAME[tmp_ptr_decoded_layer->type_layer].c_str());
                return;
    }
    // |END| Propagate the error(s) to the layer. |END|

    // Compute the gradients.
    switch(tmp_ptr_coded_layer->type_layer)
    {
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: this->Recurrent__Backward_Pass__Gradient__FC__OpenMP(batch_size, tmp_ptr_coded_layer); break;
        case LAYER::LSTM:
            if(tmp_ptr_coded_layer->Use__Bidirectional())
            {
                this->Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(true,
                                                                                                      batch_size,
                                                                                                      tmp_layer_number_outputs,
                                                                                                      tmp_ptr_array_layer_gradients,
                                                                                                      &tmp_ptr_coded_layer->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(false,
                                                                                                      batch_size,
                                                                                                      tmp_layer_number_outputs,
                                                                                                      tmp_ptr_array_layer_gradients,
                                                                                                      &tmp_ptr_coded_layer->ptr_Bidirectional_Layer->backward_layer);
            }
            else
            {
                this->Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(true,
                                                                                                      batch_size,
                                                                                                      tmp_layer_number_outputs,
                                                                                                      tmp_ptr_array_layer_gradients,
                                                                                                      tmp_ptr_coded_layer);
            }
                break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                                     tmp_ptr_coded_layer->type_layer,
                                     LAYER_NAME[tmp_ptr_coded_layer->type_layer].c_str());
                return;
    }
    // |END| Compute the gradients. |END|
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Recurrent__Backward_Pass__Average_Pooling__OpenMP(size_t const batch_size,
                                                                                                                     size_t const derivative_size_received,
                                                                                                                     real *const ptr_array_derivatives_received,
                                                                                                                     Layer const *const layer_it)
{
    for(size_t tmp_time_step_index(this->seq_w); tmp_time_step_index--;)
    {
        this->Backward_Pass__Average_Pooling__OpenMP(tmp_time_step_index,
                                                                                     batch_size,
                                                                                     derivative_size_received,
                                                                                     ptr_array_derivatives_received,
                                                                                     layer_it);
    }
}

void Model::Recurrent__Backward_Pass__FC__OpenMP(size_t const batch_size,
                                                                                                 size_t const derivative_size_received,
                                                                                                 real *const ptr_array_derivatives_received,
                                                                                                 Layer const *const layer_it)
{
    if(layer_it->type_group == GROUP::RESIDUAL)
    {
        for(size_t tmp_time_step_index(this->seq_w); tmp_time_step_index--;)
        {
            this->Backward_Pass__Residual__FC__OpenMP(tmp_time_step_index,
                                                                                     batch_size,
                                                                                     derivative_size_received,
                                                                                     ptr_array_derivatives_received,
                                                                                     layer_it);
        }
    }
    else
    {
        for(size_t tmp_time_step_index(this->seq_w); tmp_time_step_index--;)
        {
            this->Backward_Pass__FC__OpenMP(tmp_time_step_index,
                                                                     batch_size,
                                                                     derivative_size_received,
                                                                     ptr_array_derivatives_received,
                                                                     layer_it);
        }
    }
}

void Model::Recurrent__Backward_Pass__LSTM__OpenMP(size_t const batch_size,
                                                                                                      size_t const derivative_size_received,
                                                                                                      real *const ptr_array_derivatives_received,
                                                                                                      Layer const *const layer_it)
{
    size_t tmp_time_step_index;
    
    real const *const tmp_ptr_array_delta_input_block_inputs(layer_it->Get__Array_Deltas__Cell__Block_Input__Input()),
                  *const tmp_ptr_array_delta_input_input_gates(layer_it->Get__Array_Deltas__Block__Input_Gate__Input()),
                  *const tmp_ptr_array_delta_input_forget_gates(layer_it->Get__Array_Deltas__Block__Forget_Gate__Input()),
                  *const tmp_ptr_array_delta_input_output_gates(layer_it->Get__Array_Deltas__Block__Output_Gate__Input());

    for(tmp_time_step_index = 0_UZ; tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
    {
        this->Backward_Pass__LSTM__OpenMP(tmp_time_step_index,
                                                                     batch_size,
                                                                     derivative_size_received,
                                                                     tmp_ptr_array_delta_input_block_inputs,
                                                                     tmp_ptr_array_delta_input_input_gates,
                                                                     tmp_ptr_array_delta_input_forget_gates,
                                                                     tmp_ptr_array_delta_input_output_gates,
                                                                     ptr_array_derivatives_received,
                                                                     layer_it);
    }
}

void Model::Recurrent__Backward_Pass__Max_Pooling__OpenMP(size_t const batch_size,
                                                                                                                size_t const derivative_size_received,
                                                                                                                real *const ptr_array_derivatives_received,
                                                                                                                Layer const *const layer_it)
{
    for(size_t tmp_time_step_index(this->seq_w); tmp_time_step_index--;)
    {
        this->Backward_Pass__Max_Pooling__OpenMP(tmp_time_step_index,
                                                                               batch_size,
                                                                               derivative_size_received,
                                                                               ptr_array_derivatives_received,
                                                                               layer_it);
    }
}

void Model::Recurrent__Backward_Pass__Residual__OpenMP(size_t const batch_size,
                                                                                                          size_t const derivative_size_received,
                                                                                                          real *const ptr_array_derivatives_received,
                                                                                                          Layer const *const layer_it)
{
    for(size_t tmp_time_step_index(this->seq_w); tmp_time_step_index--;)
    {
        this->Backward_Pass__Residual__OpenMP(tmp_time_step_index,
                                                                               batch_size,
                                                                               derivative_size_received,
                                                                               ptr_array_derivatives_received,
                                                                               layer_it);
    }
}

void Model::Recurrent__Backward_Pass__Residual__Block__OpenMP(size_t const batch_size,
                                                                                                                     size_t const derivative_size_received,
                                                                                                                     real *const ptr_array_derivatives_received,
                                                                                                                     Layer const *const layer_it)
{
    for(size_t tmp_time_step_index(this->seq_w); tmp_time_step_index--;)
    {
        this->Backward_Pass__Residual__Block__OpenMP(tmp_time_step_index,
                                                                                     batch_size,
                                                                                     derivative_size_received,
                                                                                     ptr_array_derivatives_received,
                                                                                     layer_it);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Recurrent__Backward_Pass__Gradient__FC__OpenMP(size_t const batch_size, Layer const *const layer_it)
{
    for(size_t tmp_time_step_index(this->seq_w); tmp_time_step_index--;)
    {
        this->Backward_Pass__Gradient__FC__OpenMP(tmp_time_step_index,
                                                                        batch_size,
                                                                        layer_it);
    }
}

void Model::Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(bool const forward_layer_received,
                                                                                                                     size_t const batch_size,
                                                                                                                     size_t const derivative_input_size_received,
                                                                                                                     real *const ptr_array_derivative_inputs_received,
                                                                                                                     Layer *const layer_it)
{
    BlockUnit *const tmp_ptr_layer_first_block_unit(layer_it->ptr_array_block_units);
    
    CellUnit *const tmp_ptr_layer_first_cell_unit(layer_it->ptr_array_cell_units);
    
    size_t const tmp_number_block_units(static_cast<size_t>(layer_it->ptr_last_block_unit - tmp_ptr_layer_first_block_unit)),
                       tmp_number_cell_units(static_cast<size_t>(layer_it->ptr_last_cell_unit - tmp_ptr_layer_first_cell_unit));

    long long int tmp_time_step_index,
                       tmp_time_step_start(forward_layer_received ? static_cast<long long int>(this->seq_w - 1_UZ) : 0ll),
                       tmp_time_step_end(forward_layer_received ? -1ll : static_cast<long long int>(this->seq_w)),
                       tmp_time_prediction_direction_end(forward_layer_received ? 0ll : static_cast<long long int>(this->seq_w - 1_UZ));
    
    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    for(tmp_time_step_index = tmp_time_step_start; tmp_time_step_index != tmp_time_step_end; forward_layer_received ? --tmp_time_step_index : ++tmp_time_step_index)
    {
        // Output.
        this->Backward_Pass__LSTM_Derivative__Output__OpenMP(tmp_time_step_index,
                                                                                            forward_layer_received ? (tmp_time_step_index + 1ll) : (tmp_time_step_index - 1ll),
                                                                                            tmp_time_step_start,
                                                                                            batch_size,
                                                                                            tmp_number_block_units,
                                                                                            tmp_number_cell_units,
                                                                                            layer_it->Get__Array_Deltas__Cell__Block_Input__Recurrent(),
                                                                                            layer_it->Get__Array_Deltas__Block__Input_Gate__Recurrent(),
                                                                                            layer_it->Get__Array_Deltas__Block__Forget_Gate__Recurrent(),
                                                                                            layer_it->Get__Array_Deltas__Block__Output_Gate__Recurrent(),
                                                                                            layer_it);
        
        // Output gate normalization.
        if(layer_it->Use__Normalization())
        {
            // Output gate, memcpy.
            #pragma omp single
            {
                memcpy(tmp_ptr_layer_first_block_unit->ptr_delta_input_outputs_gates + this->batch_size * tmp_number_block_units * static_cast<size_t>(tmp_time_step_index),
                             tmp_ptr_layer_first_block_unit->ptr_delta_outputs_gates + this->batch_size * tmp_number_block_units * static_cast<size_t>(tmp_time_step_index),
                             batch_size * tmp_number_block_units * sizeof(real));
                
                if(tmp_time_step_index != tmp_time_prediction_direction_end)
                {
                  memcpy(tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_outputs_gates + this->batch_size * tmp_number_block_units * static_cast<size_t>(tmp_time_step_index),
                                   tmp_ptr_layer_first_block_unit->ptr_delta_outputs_gates + this->batch_size * tmp_number_block_units * static_cast<size_t>(tmp_time_step_index),
                                   batch_size * tmp_number_block_units * sizeof(real));
                }
            }
            // |END| Output gate, memcpy. |END|
            
            // Normalization.
            switch(layer_it->type_normalization)
            {
                case LAYER_NORM::BATCH_NORMALIZATION:
                    this->Backward_Pass__Batch_Normalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                batch_size,
                                                                                                tmp_number_block_units,
                                                                                                tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_delta_input_outputs_gates,
                                                                                                tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_delta_input_outputs_gates);

                    if(tmp_time_step_index != tmp_time_prediction_direction_end)
                    {
                        this->Backward_Pass__Batch_Normalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_outputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_outputs_gates);
                    }
                        break;
                case LAYER_NORM::BATCH_RENORMALIZATION:
                    this->Backward_Pass__Batch_Renormalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_r_correction,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_input_outputs_gates);
                    
                    if(tmp_time_step_index != tmp_time_prediction_direction_end)
                    {
                        this->Backward_Pass__Batch_Renormalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_r_correction,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_outputs_gates);
                    }
                        break;
                default: break;
            }
        }
        
        // Cell activation state.
        this->Backward_Pass__LSTM_Derivative__Cell_State_AF__OpenMP(tmp_time_step_index,
                                                                                                       forward_layer_received ? (tmp_time_step_index + 1ll) : (tmp_time_step_index - 1ll),
                                                                                                       tmp_time_prediction_direction_end,
                                                                                                       tmp_time_step_start,
                                                                                                       batch_size,
                                                                                                       tmp_number_block_units,
                                                                                                       tmp_number_cell_units,
                                                                                                       layer_it->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                                       layer_it);
        
        // Cell state normalization.
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                this->Backward_Pass__Batch_Normalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                            batch_size,
                                                                                            tmp_number_cell_units,
                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_means,
                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                            tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_hats,
                                                                                            tmp_ptr_layer_first_cell_unit->ptr_delta_cell_state,
                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_derivatives_means,
                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                            tmp_ptr_layer_first_cell_unit->ptr_delta_cell_state);
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                this->Backward_Pass__Batch_Renormalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                batch_size,
                                                                                                tmp_number_cell_units,
                                                                                                tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_r_correction,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_delta_cell_state,
                                                                                                tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_delta_cell_state);
                    break;
            default: break;
        }

        // CIF gate, activation, state.
        this->Backward_Pass__LSTM_Derivative__Gates_CIF_AF_State__OpenMP(tmp_time_step_index,
                                                                                                                 forward_layer_received ? (tmp_time_step_index + 1ll) : (tmp_time_step_index - 1ll),
                                                                                                                 forward_layer_received ? (tmp_time_step_index - 1ll) : (tmp_time_step_index + 1ll),
                                                                                                                 tmp_time_prediction_direction_end,
                                                                                                                 tmp_time_step_start,
                                                                                                                 batch_size,
                                                                                                                 tmp_number_block_units,
                                                                                                                 tmp_number_cell_units,
                                                                                                                 layer_it);

        // CIF gate normalization.
        if(layer_it->Use__Normalization())
        {
            // memcpy.
            #pragma omp single
            {
        memcpy(
            tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input_input +
                               this->batch_size * tmp_number_cell_units *
                                   static_cast<size_t>(tmp_time_step_index),
                           tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input +
                               this->batch_size * tmp_number_cell_units *
                                   static_cast<size_t>(tmp_time_step_index),
                             batch_size * tmp_number_cell_units * sizeof(real));

                memcpy(
            tmp_ptr_layer_first_block_unit->ptr_delta_input_inputs_gates +
                   this->batch_size * tmp_number_block_units *
                       static_cast<size_t>(tmp_time_step_index),
            tmp_ptr_layer_first_block_unit->ptr_delta_inputs_gates +
                this->batch_size * tmp_number_block_units *
                    static_cast<size_t>(tmp_time_step_index),
                             batch_size * tmp_number_block_units * sizeof(real));

                memcpy(tmp_ptr_layer_first_block_unit
                               ->ptr_delta_input_forgets_gates +
                           this->batch_size * tmp_number_block_units *
                               static_cast<size_t>(tmp_time_step_index),
                       tmp_ptr_layer_first_block_unit->ptr_delta_forgets_gates +
                           this->batch_size * tmp_number_block_units *
                               static_cast<size_t>(tmp_time_step_index),
                             batch_size * tmp_number_block_units * sizeof(real));
                
                if(tmp_time_step_index != tmp_time_prediction_direction_end)
                {
                  memcpy(tmp_ptr_layer_first_cell_unit
                                 ->ptr_delta_cell_recurrent_input +
                             this->batch_size * tmp_number_cell_units *
                                    static_cast<size_t>(tmp_time_step_index),
                            tmp_ptr_layer_first_cell_unit
                                    ->ptr_delta_cell_input +
                                this->batch_size * tmp_number_cell_units *
                                    static_cast<size_t>(tmp_time_step_index),
                                 batch_size * tmp_number_cell_units * sizeof(real));

                    memcpy(
                      tmp_ptr_layer_first_block_unit
                              ->ptr_delta_recurrent_inputs_gates +
                          this->batch_size * tmp_number_block_units *
                              static_cast<size_t>(tmp_time_step_index),
                      tmp_ptr_layer_first_block_unit->ptr_delta_inputs_gates +
                          this->batch_size * tmp_number_block_units *
                              static_cast<size_t>(tmp_time_step_index),
                                 batch_size * tmp_number_block_units * sizeof(real));

                    memcpy(tmp_ptr_layer_first_block_unit
                                   ->ptr_delta_recurrent_forgets_gates +
                               this->batch_size * tmp_number_block_units *
                                   static_cast<size_t>(tmp_time_step_index),
                           tmp_ptr_layer_first_block_unit
                                   ->ptr_delta_forgets_gates +
                               this->batch_size * tmp_number_block_units *
                                   static_cast<size_t>(tmp_time_step_index),
                                 batch_size * tmp_number_block_units * sizeof(real));
                }
            }
            // |END| memcpy. |END|
            
            // Normalization.
            switch(layer_it->type_normalization)
            {
                case LAYER_NORM::BATCH_NORMALIZATION:
                    // Block input, input.
                    this->Backward_Pass__Batch_Normalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                batch_size,
                                                                                                tmp_number_cell_units,
                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input_input,
                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input_input);

                    // Input gate, input.
                    this->Backward_Pass__Batch_Normalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                batch_size,
                                                                                                tmp_number_block_units,
                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_delta_input_inputs_gates,
                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_block_unit->ptr_delta_input_inputs_gates);

                    if(tmp_time_step_index != tmp_time_prediction_direction_end)
                    {
                        // Forget gate, input.
                        this->Backward_Pass__Batch_Normalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_input_forgets_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_input_forgets_gates);

                        // Block input, recurrent.
                        this->Backward_Pass__Batch_Normalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size,
                                                                                                    tmp_number_cell_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_delta_cell_recurrent_input,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_delta_cell_recurrent_input);
                        
                        // Input gate, recurrent.
                        this->Backward_Pass__Batch_Normalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_inputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_inputs_gates);

                        // Forget gate, recurrent.
                        this->Backward_Pass__Batch_Normalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_forgets_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_forgets_gates);
                    }
                        break;
                case LAYER_NORM::BATCH_RENORMALIZATION:
                    // Block input, input.
                    this->Backward_Pass__Batch_Renormalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size,
                                                                                                    tmp_number_cell_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_r_correction,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input_input,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_delta_cell_input_input);
                    
                    // Input gate, input.
                    this->Backward_Pass__Batch_Renormalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                    batch_size,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_r_correction,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_input_inputs_gates,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_block_unit->ptr_delta_input_inputs_gates);
                    
                    if(tmp_time_step_index != tmp_time_prediction_direction_end)
                    {
                        // Forget gate, input.
                        this->Backward_Pass__Batch_Renormalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_r_correction,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_input_forgets_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_delta_input_forgets_gates);
                        
                        // Block input, recurrent.
                        this->Backward_Pass__Batch_Renormalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_r_correction,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_delta_cell_recurrent_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_delta_cell_recurrent_input);
                        
                        // Input gate, recurrent.
                        this->Backward_Pass__Batch_Renormalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_inputs_gates);
                        
                        // Forget gate, recurrent.
                        this->Backward_Pass__Batch_Renormalization__OpenMP(static_cast<size_t>(tmp_time_step_index),
                                                                                                             batch_size,
                                                                                                             tmp_number_block_units,
                                                                                                             tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_r_correction,
                                                                                                             tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                             tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_forgets_gates,
                                                                                                             tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                             tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_derivatives_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                             tmp_ptr_layer_first_block_unit->ptr_delta_recurrent_forgets_gates);
                    }
                        break;
                default: break;
            }
        }
    }
}

void Model::Recurrent__Backward_Pass__Gradient__Residual__OpenMP(size_t const batch_size, Layer const *const layer)
{
  Layer const *const tmp_ptr_layer_end(layer + 1);
  Layer *layer_it(this->ptr_array_layers +
                  static_cast<size_t>(layer - this->ptr_array_layers) +
                  layer->block_depth);
    
    // Remaining layer(s).
    for(; layer_it != tmp_ptr_layer_end; --layer_it)
    {
        this->Recurrent__Backward_Pass__Gradient__Residual__Layer__OpenMP(false,
                                                                                                                batch_size,
                                                                                                                layer_it);
    }
    // |END| Remaining layer(s). |END|
    
    // First block layer.
    this->Recurrent__Backward_Pass__Gradient__Residual__Layer__OpenMP(true,
                                                                                                            batch_size,
                                                                                                            layer_it);
    // |END| First block layer. |END|
}

void Model::Recurrent__Backward_Pass__Gradient__Residual__Layer__OpenMP(bool const is_block_input_layer_received,
                                                                                                                                     size_t const batch_size,
                                                                                                                                     Layer *&layer_it)
{
    size_t const tmp_layer_number_outputs(*layer_it->ptr_number_outputs);
    
    real *const tmp_ptr_array_layer_gradients(layer_it->ptr_array_derivative_outputs);
    
    Layer const *const next_layer_it(layer_it->next_connected_layers[0]);
    
    // clear past error(s).
    #pragma omp single
    memset(tmp_ptr_array_layer_gradients,
                 0,
                 this->batch_size * tmp_layer_number_outputs * this->seq_w * sizeof(real));
    // |END| clear past error(s). |END|
    
    // Propagate the error(s) to the layer.
    switch(next_layer_it->type_layer)
    {
        case LAYER::AVERAGE_POOLING:
            this->Recurrent__Backward_Pass__Average_Pooling__OpenMP(batch_size,
                                                                                                        tmp_layer_number_outputs,
                                                                                                        tmp_ptr_array_layer_gradients,
                                                                                                        next_layer_it);
                break;
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Backward_Pass__FC__OpenMP(batch_size,
                                                                                      tmp_layer_number_outputs,
                                                                                      tmp_ptr_array_layer_gradients,
                                                                                      next_layer_it);
                break;
        case LAYER::LSTM:
            this->Recurrent__Backward_Pass__LSTM__OpenMP(batch_size,
                                                                                          tmp_layer_number_outputs,
                                                                                          tmp_ptr_array_layer_gradients,
                                                                                          next_layer_it);
                break;
        case LAYER::MAX_POOLING:
            this->Recurrent__Backward_Pass__Max_Pooling__OpenMP(batch_size,
                                                                                                    tmp_layer_number_outputs,
                                                                                                    tmp_ptr_array_layer_gradients,
                                                                                                    next_layer_it);
                break;
        case LAYER::RESIDUAL:
            this->Recurrent__Backward_Pass__Residual__Block__OpenMP(batch_size,
                                                                                                        tmp_layer_number_outputs,
                                                                                                        tmp_ptr_array_layer_gradients,
                                                                                                        next_layer_it);
                break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                                     next_layer_it->type_layer,
                                     LAYER_NAME[next_layer_it->type_layer].c_str());
                return;
    }
    // |END| Propagate the error(s) to the layer. |END|

    // Compute the gradients.
    switch(layer_it->type_layer)
    {
        case LAYER::AVERAGE_POOLING:
        case LAYER::MAX_POOLING: break;
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            this->Recurrent__Backward_Pass__Gradient__Residual__FC__OpenMP(is_block_input_layer_received,
                                                                                                                      batch_size,
                                                                                                                      layer_it);
                break;
        case LAYER::LSTM:
            if(layer_it->Use__Bidirectional())
            {
                this->Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(true,
                                                                                                            batch_size,
                                                                                                            tmp_layer_number_outputs,
                                                                                                            tmp_ptr_array_layer_gradients,
                                                                                                            &layer_it->ptr_Bidirectional_Layer->forward_layer);
                this->Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(false,
                                                                                                            batch_size,
                                                                                                            tmp_layer_number_outputs,
                                                                                                            tmp_ptr_array_layer_gradients,
                                                                                                            &layer_it->ptr_Bidirectional_Layer->backward_layer);
            }
            else
            {
                this->Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(true,
                                                                                                            batch_size,
                                                                                                            tmp_layer_number_outputs,
                                                                                                            tmp_ptr_array_layer_gradients,
                                                                                                            layer_it);
            }
                break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                                     layer_it->type_layer,
                                     LAYER_NAME[layer_it->type_layer].c_str());
                return;
    }
    // |END| Compute the gradients. |END|
}

void Model::Recurrent__Backward_Pass__Gradient__Residual__FC__OpenMP(bool const is_block_input_layer_received,
                                                                                                                                 size_t const batch_size,
                                                                                                                                 Layer const *const layer_it)
{
    for(size_t tmp_time_step_index(this->seq_w); tmp_time_step_index--;)
    {
        this->Backward_Pass__Gradient__Residual__FC__OpenMP(is_block_input_layer_received,
                                                                                                tmp_time_step_index,
                                                                                                batch_size,
                                                                                                layer_it);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Backward_Pass__LSTM__OpenMP(size_t const time_step_index_received,
                                                                                    size_t const batch_size,
                                                                                    size_t const derivative_input_size_received,
                                                                                    real const *const ptr_array_delta_input_block_inputs_received,
                                                                                    real const *const ptr_array_delta_input_input_gates_received,
                                                                                    real const *const ptr_array_delta_input_forget_gates_received,
                                                                                    real const *const ptr_array_delta_input_output_gates_received,
                                                                                    real *const ptr_array_derivative_inputs_received,
                                                                                    Layer const *const layer_it)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_number_blocks(static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units)),
                       tmp_number_cells(static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units));
    size_t tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index;

    var const *tmp_ptr_array_parameters;
    real *tmp_ptr_array_previous_layer_errors,
         tmp_error;
    
    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it;

    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * tmp_number_blocks + this->batch_size * tmp_number_blocks * time_step_index_received;

        tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * tmp_number_cells + this->batch_size * tmp_number_cells * time_step_index_received;

        tmp_ptr_block_unit_it = layer_it->ptr_array_block_units;

        tmp_ptr_array_previous_layer_errors = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * derivative_input_size_received + this->batch_size * derivative_input_size_received * time_step_index_received;

        for(tmp_cell_index = 0_UZ,
            tmp_block_index = 0_UZ; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                          ++tmp_block_index)
        {
            // Cells inputs to previous neurons.
            for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                   ++tmp_cell_index)
            {
                tmp_error = ptr_array_delta_input_block_inputs_received[tmp_cell_data_timed_index + tmp_cell_index];

                tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;
                
                for(tmp_connection_index = tmp_ptr_cell_unit_it->last_index_feedforward_connection_cell_input - tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input; tmp_connection_index--;)
                {
                    tmp_ptr_array_previous_layer_errors[tmp_connection_index] += tmp_error * cast(tmp_ptr_array_parameters[tmp_connection_index]);
                }
            }
            // |END| Cell input to previous neurons. |END|

            // Input gate to previous neurons.
            tmp_error = ptr_array_delta_input_input_gates_received[tmp_block_data_timed_index + tmp_block_index];

            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;
            
            for(tmp_connection_index = tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate; tmp_connection_index--;)
            {
                tmp_ptr_array_previous_layer_errors[tmp_connection_index] += tmp_error * cast(tmp_ptr_array_parameters[tmp_connection_index]);
            }
            // |END| Input gate to previous neurons. |END|

            // Forget gate to previous neurons.
            tmp_error = ptr_array_delta_input_forget_gates_received[tmp_block_data_timed_index + tmp_block_index];

            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;
            
            for(tmp_connection_index = tmp_ptr_block_unit_it->last_index_feedforward_connection_forget_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate; tmp_connection_index--;)
            {
                tmp_ptr_array_previous_layer_errors[tmp_connection_index] += tmp_error * cast(tmp_ptr_array_parameters[tmp_connection_index]);
            }
            // |END| Forget gate to previous neurons. |END|

            // Output gate to previous neurons.
            tmp_error = ptr_array_delta_input_output_gates_received[tmp_block_data_timed_index + tmp_block_index];

            tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;
            
            for(tmp_connection_index = tmp_ptr_block_unit_it->last_index_feedforward_connection_output_gate - tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate; tmp_connection_index--;)
            {
                tmp_ptr_array_previous_layer_errors[tmp_connection_index] += tmp_error * cast(tmp_ptr_array_parameters[tmp_connection_index]);
            }
            // |END| Output gate to previous neurons. |END|
        }
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Backward_Pass__LSTM_Derivative__Output__OpenMP(long long int const time_step_index_received,
                                                                                                                long long int const time_step_direction_received,
                                                                                                                long long int const time_step_prediction_end_received,
                                                                                                                size_t const batch_size,
                                                                                                                size_t const block_unit_size_received,
                                                                                                                size_t const cell_unit_size_received,
                                                                                                                real const *const ptr_array_delta_recurrent_block_inputs_received,
                                                                                                                real const *const ptr_array_delta_recurrent_input_gates_received,
                                                                                                                real const *const ptr_array_delta_recurrent_forget_gates_received,
                                                                                                                real const *const ptr_array_delta_recurrent_output_gates_received,
                                                                                                                Layer *const layer_it)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_number_recurrents_connection(layer_it->ptr_array_block_units->last_index_recurrent_connection_input_gate - layer_it->ptr_array_block_units->first_index_recurrent_connection_input_gate);
    size_t tmp_connection_index,
              tmp_block_index,
              tmp_block_data_timed_index,
              tmp_block_data_direction_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_direction_timed_index;

    var const *tmp_ptr_array_cell_input_parameters,
                  *tmp_ptr_array_input_gate_parameters,
                  *tmp_ptr_array_forget_gate_parameters,
                  *tmp_ptr_array_output_gate_parameters;
    real *tmp_ptr_array_delta_cells_outputs,
         tmp_activation,
         tmp_error,
         tmp_cell_input_error,
         tmp_input_gate_error,
         tmp_forget_gate_error,
         tmp_output_gate_error;
    
    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it;

    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;
    
    if(time_step_index_received != time_step_prediction_end_received)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_block_data_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_direction_received);

            tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);

            tmp_cell_data_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_direction_received);

            tmp_ptr_array_delta_cells_outputs = layer_it->ptr_array_block_units->ptr_array_delta_cells_outputs + tmp_cell_data_timed_index;

            // Cells inputs.
            for(tmp_cell_index = 0_UZ,
                tmp_ptr_last_cell_unit = layer_it->ptr_last_cell_unit,
                tmp_ptr_cell_unit_it = layer_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                  ++tmp_cell_index)
            {
                tmp_cell_input_error = ptr_array_delta_recurrent_block_inputs_received[tmp_cell_data_direction_timed_index + tmp_cell_index];
                
                tmp_ptr_array_cell_input_parameters = this->ptr_array_parameters + tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;

                for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
                {
                    tmp_ptr_array_delta_cells_outputs[tmp_connection_index] += tmp_cell_input_error * cast(tmp_ptr_array_cell_input_parameters[tmp_connection_index]);
                }
            }
            // |END| Cells inputs. |END|

            for(tmp_block_index = 0_UZ,
                tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                                                                                             ++tmp_block_index)
            {
                // Gates-recurrent.
                tmp_input_gate_error = ptr_array_delta_recurrent_input_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];
                tmp_forget_gate_error = ptr_array_delta_recurrent_forget_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];
                tmp_output_gate_error = ptr_array_delta_recurrent_output_gates_received[tmp_block_data_direction_timed_index + tmp_block_index];
                
                tmp_ptr_array_input_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;
                tmp_ptr_array_forget_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;
                tmp_ptr_array_output_gate_parameters = this->ptr_array_parameters + tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;
                
                for(tmp_connection_index = tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate - tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate; tmp_connection_index--;)
                {
                    tmp_error = tmp_input_gate_error * cast(tmp_ptr_array_input_gate_parameters[tmp_connection_index]);
                    tmp_error += tmp_forget_gate_error * cast(tmp_ptr_array_forget_gate_parameters[tmp_connection_index]);
                    tmp_error += tmp_output_gate_error * cast(tmp_ptr_array_output_gate_parameters[tmp_connection_index]);
                    
                    tmp_ptr_array_delta_cells_outputs[tmp_connection_index] += tmp_error;
                }
                // |END| Gates-recurrent. |END|
            }

            for(tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                // Output gate, activation.
                tmp_error = 0_r;

                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    tmp_error += tmp_ptr_cell_unit_it->ptr_delta_cell_output[tmp_cell_data_timed_index] * cast(tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index]);
                }
                
                tmp_activation = cast(tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index]);

                tmp_ptr_block_unit_it->ptr_delta_outputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_derive(tmp_activation) * tmp_error;
                // |END| Output gate, activation. |END|
            }
        }
    }
    else
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);

            for(tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                // Output gate, activation.
                tmp_error = 0_r;

                for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                    tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                {
                    tmp_error += tmp_ptr_cell_unit_it->ptr_delta_cell_output[tmp_cell_data_timed_index] * cast(tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index]);
                }
                
                tmp_activation = cast(tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index]);

                tmp_ptr_block_unit_it->ptr_delta_outputs_gates[tmp_block_data_timed_index] = AF_SIGMOID_derive(tmp_activation) * tmp_error;
                // |END| Output gate, activation. |END|
            }
        }
    }
}

void Model::Backward_Pass__LSTM_Derivative__Cell_State_AF__OpenMP(long long int const time_step_index_received,
                                                                                                                            long long int const time_step_direction_received,
                                                                                                                            long long int const time_step_prediction_start_received,
                                                                                                                            long long int const time_step_prediction_end_received,
                                                                                                                            size_t const batch_size,
                                                                                                                            size_t const block_unit_size_received,
                                                                                                                            size_t const cell_unit_size_received,
                                                                                                                            var const *const ptr_array_summation_cell_states_received,
                                                                                                                            Layer *const layer_it)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t tmp_block_data_timed_index,
              tmp_cell_layer_index,
              tmp_cell_data_timed_index;
    
    real tmp_output_gate;
    
    ACTIVATION::TYPE const tmp_type_activation_function_io(layer_it->ptr_array_block_units->activation_function_io);
    
    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it;

    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
        
        tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);
        
        for(tmp_cell_layer_index = 0_UZ,
            tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
        {
            tmp_output_gate = cast(tmp_ptr_block_unit_it->ptr_outputs_gates[tmp_block_data_timed_index]);

            for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                    ++tmp_cell_layer_index)
            {
                // Cell state.
                tmp_ptr_cell_unit_it->ptr_delta_cell_state[tmp_cell_data_timed_index] = tmp_ptr_cell_unit_it->ptr_delta_cell_output[tmp_cell_data_timed_index] * tmp_output_gate * this->activation_fn_derivative(tmp_type_activation_function_io,
                                                                                                                                                                                                                                                                                                               cast(ptr_array_summation_cell_states_received[tmp_cell_data_timed_index + tmp_cell_layer_index]),
                                                                                                                                                                                                                                                                                                               cast(tmp_ptr_cell_unit_it->ptr_cell_state_activate[tmp_cell_data_timed_index]));
                // |END| Cell state. |END|
            }
        }
    }
}

void Model::Backward_Pass__LSTM_Derivative__Gates_CIF_AF_State__OpenMP(long long int const time_step_index_received,
                                                                                                                                      long long int const time_step_direction_received,
                                                                                                                                      long long int const time_step_reverse_direction_received,
                                                                                                                                      long long int const time_step_prediction_start_received,
                                                                                                                                      long long int const time_step_prediction_end_received,
                                                                                                                                      size_t const batch_size,
                                                                                                                                      size_t const block_unit_size_received,
                                                                                                                                      size_t const cell_unit_size_received,
                                                                                                                                      Layer *const layer_it)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_number_cells_per_block(static_cast<size_t>(layer_it->ptr_array_block_units->ptr_last_cell_unit - layer_it->ptr_array_block_units->ptr_array_cell_units));
    size_t tmp_block_data_timed_index,
              tmp_block_data_direction_timed_index,
              tmp_cell_index,
              tmp_cell_data_timed_index,
              tmp_cell_data_direction_timed_index,
              tmp_cell_data_reverse_direction_timed_index,
              tmp_first_index_peephole_input_gate,
              tmp_first_index_peephole_forget_gate,
              tmp_first_index_peephole_output_gate;
    
    var const *tmp_ptr_array_cell_inputs, *tmp_ptr_array_cell_summation_inputs,
        *tmp_ptr_array_cell_states_reverse_direction_timed;
    real const *tmp_ptr_array_delta_cell_states_direction_timed;
    real *tmp_ptr_array_delta_cell_inputs,
        *tmp_ptr_array_delta_cell_states,
        tmp_input_gate_activation,
        tmp_forget_gate_activation,
        tmp_input_gate,
        tmp_forget_gate_dt,
        tmp_delta_input_gate_dt,
        tmp_delta_forget_gate_dt,
        tmp_delta_output_gate,
        tmp_cell_state_error,
        tmp_input_gate_error,
        tmp_forget_gate_error;
    
    ACTIVATION::TYPE const tmp_type_activation_function_io(layer_it->ptr_array_block_units->activation_function_io);

    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it;

    CellUnit *tmp_ptr_block_it_cell_unit;

    if(time_step_index_received != time_step_prediction_end_received && time_step_index_received != time_step_prediction_start_received)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
            tmp_block_data_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_direction_received);

            tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            tmp_cell_data_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_direction_received);
            tmp_cell_data_reverse_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);

            for(tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                tmp_ptr_block_it_cell_unit = tmp_ptr_block_unit_it->ptr_array_cell_units;

                tmp_first_index_peephole_input_gate = tmp_ptr_block_it_cell_unit->index_peephole_input_gate;
                tmp_first_index_peephole_forget_gate = tmp_ptr_block_it_cell_unit->index_peephole_forget_gate;
                tmp_first_index_peephole_output_gate = tmp_ptr_block_it_cell_unit->index_peephole_output_gate;

                tmp_ptr_array_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_cell_summation_inputs = tmp_ptr_block_it_cell_unit->ptr_summation_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_cell_states_reverse_direction_timed = tmp_ptr_block_it_cell_unit->ptr_cell_state + tmp_cell_data_reverse_direction_timed_index;
                tmp_ptr_array_delta_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_delta_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_states = tmp_ptr_block_it_cell_unit->ptr_delta_cell_state + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_states_direction_timed = tmp_ptr_block_it_cell_unit->ptr_delta_cell_state + tmp_cell_data_direction_timed_index;
                
                tmp_input_gate = cast(tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index]);
                tmp_forget_gate_dt = cast(tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_direction_timed_index]);
                
                tmp_delta_input_gate_dt = tmp_ptr_block_unit_it->ptr_delta_inputs_gates[tmp_block_data_direction_timed_index];
                tmp_delta_forget_gate_dt = tmp_ptr_block_unit_it->ptr_delta_forgets_gates[tmp_block_data_direction_timed_index];
                tmp_delta_output_gate = tmp_ptr_block_unit_it->ptr_delta_outputs_gates[tmp_block_data_timed_index];
                
                // Cells.
                for(tmp_cell_index = 0_UZ; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    // Cell state.
                    tmp_cell_state_error = tmp_ptr_array_delta_cell_states[tmp_cell_index];
                    
                #ifndef NO_PEEPHOLE
                    tmp_cell_state_error += cast(this->ptr_array_parameters[tmp_first_index_peephole_output_gate + tmp_cell_index]) * tmp_delta_output_gate;
                    
                    tmp_cell_state_error += cast(this->ptr_array_parameters[tmp_first_index_peephole_input_gate + tmp_cell_index]) * tmp_delta_input_gate_dt;

                    tmp_cell_state_error += cast(this->ptr_array_parameters[tmp_first_index_peephole_forget_gate + tmp_cell_index]) * tmp_delta_forget_gate_dt;
                #endif
                    
                    tmp_cell_state_error += tmp_ptr_array_delta_cell_states_direction_timed[tmp_cell_index] * tmp_forget_gate_dt;
                    
                    tmp_ptr_array_delta_cell_states[tmp_cell_index] = tmp_cell_state_error;
                    // |END| Cell state. |END|

                    // Cell input.
                    tmp_ptr_array_delta_cell_inputs[tmp_cell_index] = tmp_cell_state_error * tmp_input_gate * this->activation_fn_derivative(tmp_type_activation_function_io,
                                                                                                                                                                                                        cast(tmp_ptr_array_cell_summation_inputs[tmp_cell_index]),
                                                                                                                                                                                                        cast(tmp_ptr_array_cell_inputs[tmp_cell_index]));
                    // |END| Cell input. |END|
                }
                // |END| Cells. |END|
                
                // Gates.
                tmp_input_gate_error = 0_r;
                tmp_forget_gate_error = 0_r;
                
                for(tmp_cell_index = 0_UZ; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    tmp_cell_state_error = tmp_ptr_array_delta_cell_states[tmp_cell_index];

                    tmp_input_gate_error += tmp_cell_state_error * cast(tmp_ptr_array_cell_inputs[tmp_cell_index]);
                    tmp_forget_gate_error += tmp_cell_state_error * cast(tmp_ptr_array_cell_states_reverse_direction_timed[tmp_cell_index]);
                }
                
                tmp_input_gate_activation = cast(tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index]);
                tmp_forget_gate_activation = cast(tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index]);

                tmp_ptr_block_unit_it->ptr_delta_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_error * AF_SIGMOID_derive(tmp_input_gate_activation);
                tmp_ptr_block_unit_it->ptr_delta_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_error * AF_SIGMOID_derive(tmp_forget_gate_activation);
                // |END| Gates. |END|
            }
        }
    }
    else if(time_step_index_received != time_step_prediction_end_received)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
            tmp_block_data_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_direction_received);

            tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            tmp_cell_data_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_direction_received);

            for(tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                tmp_ptr_block_it_cell_unit = tmp_ptr_block_unit_it->ptr_array_cell_units;

                tmp_first_index_peephole_input_gate = tmp_ptr_block_it_cell_unit->index_peephole_input_gate;
                tmp_first_index_peephole_forget_gate = tmp_ptr_block_it_cell_unit->index_peephole_forget_gate;
                tmp_first_index_peephole_output_gate = tmp_ptr_block_it_cell_unit->index_peephole_output_gate;

                tmp_ptr_array_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_cell_summation_inputs = tmp_ptr_block_it_cell_unit->ptr_summation_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_delta_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_states = tmp_ptr_block_it_cell_unit->ptr_delta_cell_state + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_states_direction_timed = tmp_ptr_block_it_cell_unit->ptr_delta_cell_state + tmp_cell_data_direction_timed_index;
                
                tmp_input_gate = cast(tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index]);
                tmp_forget_gate_dt = cast(tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_direction_timed_index]);

                tmp_delta_input_gate_dt = tmp_ptr_block_unit_it->ptr_delta_inputs_gates[tmp_block_data_direction_timed_index];
                tmp_delta_forget_gate_dt = tmp_ptr_block_unit_it->ptr_delta_forgets_gates[tmp_block_data_direction_timed_index];
                tmp_delta_output_gate = tmp_ptr_block_unit_it->ptr_delta_outputs_gates[tmp_block_data_timed_index];
                
                // Cells.
                for(tmp_cell_index = 0_UZ; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    // Cell state.
                    tmp_cell_state_error = tmp_ptr_array_delta_cell_states[tmp_cell_index];
                    
                #ifndef NO_PEEPHOLE
                    tmp_cell_state_error += cast(this->ptr_array_parameters[tmp_first_index_peephole_output_gate + tmp_cell_index]) * tmp_delta_output_gate;
                    
                    tmp_cell_state_error += cast(this->ptr_array_parameters[tmp_first_index_peephole_input_gate + tmp_cell_index]) * tmp_delta_input_gate_dt;

                    tmp_cell_state_error += cast(this->ptr_array_parameters[tmp_first_index_peephole_forget_gate + tmp_cell_index]) * tmp_delta_forget_gate_dt;
                #endif

                    tmp_cell_state_error += tmp_ptr_array_delta_cell_states_direction_timed[tmp_cell_index] * tmp_forget_gate_dt;
                    
                    tmp_ptr_array_delta_cell_states[tmp_cell_index] = tmp_cell_state_error;
                    // |END| Cell state. |END|

                    // Cell input.
                    tmp_ptr_array_delta_cell_inputs[tmp_cell_index] = tmp_cell_state_error * tmp_input_gate * this->activation_fn_derivative(tmp_type_activation_function_io,
                                                                                                                                                                                                        cast(tmp_ptr_array_cell_summation_inputs[tmp_cell_index]),
                                                                                                                                                                                                        cast(tmp_ptr_array_cell_inputs[tmp_cell_index]));
                    // |END| Cell input. |END|
                }
                // |END| Cells. |END|
                
                // Gates.
                tmp_input_gate_error = 0_r;
                
                for(tmp_cell_index = 0_UZ; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    tmp_input_gate_error += tmp_ptr_array_delta_cell_states[tmp_cell_index] * cast(tmp_ptr_array_cell_inputs[tmp_cell_index]);
                }
                
                tmp_input_gate_activation = cast(tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index]);

                tmp_ptr_block_unit_it->ptr_delta_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_error * AF_SIGMOID_derive(tmp_input_gate_activation);
                // |END| Gates. |END|
            }
        }
    }
    else if(time_step_index_received != time_step_prediction_start_received)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_block_data_timed_index = static_cast<size_t>(tmp_example_index__int) * block_unit_size_received + this->batch_size * block_unit_size_received * static_cast<size_t>(time_step_index_received);
            
            tmp_cell_data_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_index_received);
            tmp_cell_data_reverse_direction_timed_index = static_cast<size_t>(tmp_example_index__int) * cell_unit_size_received + this->batch_size * cell_unit_size_received * static_cast<size_t>(time_step_reverse_direction_received);
            
            for(tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
            {
                tmp_ptr_block_it_cell_unit = tmp_ptr_block_unit_it->ptr_array_cell_units;

                tmp_first_index_peephole_output_gate = tmp_ptr_block_it_cell_unit->index_peephole_output_gate;

                tmp_ptr_array_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_cell_summation_inputs = tmp_ptr_block_it_cell_unit->ptr_summation_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_cell_states_reverse_direction_timed = tmp_ptr_block_it_cell_unit->ptr_cell_state + tmp_cell_data_reverse_direction_timed_index;
                tmp_ptr_array_delta_cell_inputs = tmp_ptr_block_it_cell_unit->ptr_delta_cell_input + tmp_cell_data_timed_index;
                tmp_ptr_array_delta_cell_states = tmp_ptr_block_it_cell_unit->ptr_delta_cell_state + tmp_cell_data_timed_index;
                
                tmp_input_gate = cast(tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index]);

                tmp_delta_output_gate = tmp_ptr_block_unit_it->ptr_delta_outputs_gates[tmp_block_data_timed_index];
                
                // Cells.
                for(tmp_cell_index = 0_UZ; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    // Cell state.
                    tmp_cell_state_error = tmp_ptr_array_delta_cell_states[tmp_cell_index];
                    
                #ifndef NO_PEEPHOLE
                    tmp_cell_state_error += cast(this->ptr_array_parameters[tmp_first_index_peephole_output_gate + tmp_cell_index]) * tmp_delta_output_gate;
                #endif

                    tmp_ptr_array_delta_cell_states[tmp_cell_index] = tmp_cell_state_error;
                    // |END| Cell state. |END|

                    // Cell input.
                    tmp_ptr_array_delta_cell_inputs[tmp_cell_index] = tmp_cell_state_error * tmp_input_gate * this->activation_fn_derivative(tmp_type_activation_function_io,
                                                                                                                                                                                                        cast(tmp_ptr_array_cell_summation_inputs[tmp_cell_index]),
                                                                                                                                                                                                        cast(tmp_ptr_array_cell_inputs[tmp_cell_index]));
                    // |END| Cell input. |END|
                }
                // |END| Cells. |END|
                
                // Gates.
                tmp_input_gate_error = 0_r;
                tmp_forget_gate_error = 0_r;
                
                for(tmp_cell_index = 0_UZ; tmp_cell_index != tmp_number_cells_per_block; ++tmp_cell_index)
                {
                    tmp_cell_state_error = tmp_ptr_array_delta_cell_states[tmp_cell_index];

                    tmp_input_gate_error += tmp_cell_state_error * cast(tmp_ptr_array_cell_inputs[tmp_cell_index]);
                    tmp_forget_gate_error += tmp_cell_state_error * cast(tmp_ptr_array_cell_states_reverse_direction_timed[tmp_cell_index]);
                }
                
                tmp_input_gate_activation = cast(tmp_ptr_block_unit_it->ptr_inputs_gates[tmp_block_data_timed_index]);
                tmp_forget_gate_activation = cast(tmp_ptr_block_unit_it->ptr_forgets_gates[tmp_block_data_timed_index]);

                tmp_ptr_block_unit_it->ptr_delta_inputs_gates[tmp_block_data_timed_index] = tmp_input_gate_error * AF_SIGMOID_derive(tmp_input_gate_activation);
                tmp_ptr_block_unit_it->ptr_delta_forgets_gates[tmp_block_data_timed_index] = tmp_forget_gate_error * AF_SIGMOID_derive(tmp_forget_gate_activation);
                // |END| Gates. |END|
            }
        }
    }
}
}
