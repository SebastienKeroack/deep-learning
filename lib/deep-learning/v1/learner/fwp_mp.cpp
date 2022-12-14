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

using namespace DL::Math;

namespace DL::v1 {
void Model::FF__Forward_Pass_Batch__OpenMP(size_t const batch_size,
                                                                                       real const *const *const ptr_array_inputs_received,
                                                                                       Layer *const ptr_first_layer_received,
                                                                                       Layer const *const last_layer)
{
    Layer const *prev_conn_layer;
    Layer *layer_it(ptr_first_layer_received + 1);
    
    // Training mode.
    #pragma omp single
    if(this->type_state_propagation >= PROPAGATION::TRAINING)
    {
        // If the network use normalization.
        if(this->Use__Normalization())
        {
            VARZERO(this->ptr_array_normalized_batch_units_means,
                        this->number_threads * this->total_normalized_units_allocated * sizeof(var));
            VARZERO(this->ptr_array_normalized_batch_units_variances,
                        this->number_threads * this->total_normalized_units_allocated * sizeof(var));
        }
    }

    // Input layer.
    this->assign_inputs_fwp_mp(batch_size, ptr_array_inputs_received);
    // |END| Input layer. |END|
    
    // Loop through each layer and do a forward propagation.
    for(; layer_it != last_layer; ++layer_it)
    {
        prev_conn_layer = layer_it->previous_connected_layers[0];

        switch(layer_it->type_layer)
        {
            case LAYER::AVERAGE_POOLING:
                this->Forward_Pass__Average_Pooling__OpenMP(0_UZ,
                                                                                          batch_size,
                                                                                          *prev_conn_layer->ptr_number_outputs,
                                                                                          prev_conn_layer->ptr_array_outputs,
                                                                                          layer_it);
                    break;
            case LAYER::FULLY_CONNECTED:
                this->Forward_Pass__FC__OpenMP(0_UZ,
                                                                     batch_size,
                                                                     *prev_conn_layer->ptr_number_outputs,
                                                                     prev_conn_layer->ptr_array_outputs,
                                                                     layer_it);
                    break;
            case LAYER::MAX_POOLING:
                this->Forward_Pass__Max_Pooling__OpenMP(0_UZ,
                                                                                    batch_size,
                                                                                    *prev_conn_layer->ptr_number_outputs,
                                                                                    prev_conn_layer->ptr_array_outputs,
                                                                                    layer_it);
                    break;
            case LAYER::RESIDUAL: this->Forward_Pass__Residual__OpenMP(batch_size, layer_it); break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         layer_it->type_layer,
                                         LAYER_NAME[layer_it->type_layer].c_str());
                    return;
        }
    }
}

void Model::FF__Forward_Pass_Batch__Pre_Training__OpenMP(
    size_t const batch_size,
    real const *const *const ptr_array_inputs_received) {
    Layer const *const last_layer(this->ptr_array_layers + this->pre_training_level),
                               *prev_conn_layer;
    Layer *layer_it(this->ptr_array_layers + 1);
    
    // Training mode.
    #pragma omp single
    if(this->type_state_propagation >= PROPAGATION::TRAINING)
    {
        // If the network use normalization.
        if(this->Use__Normalization())
        {
            VARZERO(this->ptr_array_normalized_batch_units_means,
                        this->number_threads * this->total_normalized_units_allocated * sizeof(var));
            VARZERO(this->ptr_array_normalized_batch_units_variances,
                        this->number_threads * this->total_normalized_units_allocated * sizeof(var));
        }
    }

    // Input layer.
    this->assign_inputs_pre_train_fwp_mp(batch_size, ptr_array_inputs_received);
    // |END| Input layer. |END|
    
    // Loop through each encoded layer and do a forward propagation.
    for(; layer_it != last_layer; ++layer_it)
    {
        prev_conn_layer = layer_it->previous_connected_layers[0];

        switch(layer_it->type_layer)
        {
            case LAYER::FULLY_CONNECTED:
                this->Forward_Pass__Encode__FC__OpenMP(0_UZ,
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

    // Code level part.
    prev_conn_layer = layer_it->previous_connected_layers[0];

    switch(layer_it->type_layer)
    {
        case LAYER::FULLY_CONNECTED:
            this->Forward_Pass__Code__FC__OpenMP(0_UZ,
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
    // |END| Code level part. |END|

    // Decode level part.
    prev_conn_layer = layer_it;
    layer_it = this->ptr_last_layer - static_cast<size_t>(layer_it - this->ptr_array_layers);
    
    switch(layer_it->type_layer)
    {
        case LAYER::FULLY_CONNECTED:
            this->Forward_Pass__Decode__FC__OpenMP(0_UZ,
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

void Model::Forward_Pass__Average_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                                 size_t const batch_size,
                                                                                                 size_t const input_size_received,
                                                                                                 var const *const ptr_array_inputs_received,
                                                                                                 Layer *const layer_it)
{
    Basic_unit *const tmp_ptr_layer_first_basic_unit(layer_it->ptr_array_basic_units);
    
    this->Forward_Pass__Average_Pooling__OpenMP(time_step_index_received,
                                                                              batch_size,
                                                                              input_size_received,
                                                                              *layer_it->ptr_number_outputs,
                                                                              layer_it->pooling_values[0],
                                                                              layer_it->pooling_values[1],
                                                                              layer_it->pooling_values[2],
                                                                              layer_it->pooling_values[3],
                                                                              ptr_array_inputs_received,
                                                                              tmp_ptr_layer_first_basic_unit->ptr_array_values);
    
    layer_it->ptr_array_outputs = tmp_ptr_layer_first_basic_unit->ptr_array_values;
}

void Model::Forward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                                                            size_t const batch_size,
                                                                            size_t const input_size_received,
                                                                            var const *const ptr_array_inputs_received,
                                                                            Layer *const layer_it)
{
    Neuron_unit *const tmp_ptr_layer_first_neuron_unit(layer_it->ptr_array_neuron_units);
    
    AF_unit *const tmp_ptr_layer_first_AF_unit(layer_it->ptr_array_AF_units);
    AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(layer_it->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(layer_it->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));

    var *tmp_ptr_array_inputs;

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= PROPAGATION::TRAINING)
    {
        // Weights.
        this->Forward_Pass__FC__OpenMP(time_step_index_received,
                                                              batch_size,
                                                              input_size_received,
                                                              tmp_output_size,
                                                              ptr_array_inputs_received,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                              tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(layer_it->Use__Bias())
        {
            this->Forward_Pass__Bias__OpenMP(time_step_index_received,
                                                                   batch_size,
                                                                   tmp_output_size,
                                                                   this->ptr_array_parameters + layer_it->first_bias_connection_index,
                                                                   tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
        
        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;

        // Normalization before activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation)
        {
            switch(layer_it->type_normalization)
            {
                case LAYER_NORM::BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(time_step_index_received,
                                                                                                                 batch_size,
                                                                                                                 tmp_output_size,
                                                                                                                 tmp_ptr_array_inputs,
                                                                                                                 tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                 tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                 tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                 tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                 tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                 tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                 tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                 tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case LAYER_NORM::BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(time_step_index_received,
                                                                                                                    batch_size,
                                                                                                                    tmp_output_size,
                                                                                                                    tmp_ptr_array_inputs,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    ERR(L"Layer normalization (%d | %ls) is not managed in",
                                             layer_it->type_normalization,
                                             LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                        break;
            }
                
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__OpenMP(time_step_index_received,
                                                                               batch_size,
                                                                               tmp_output_size,
                                                                               this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                               tmp_ptr_array_inputs,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation == false)
        {
            switch(layer_it->type_normalization)
            {
                case LAYER_NORM::BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(time_step_index_received,
                                                                                                                batch_size,
                                                                                                                tmp_output_size,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case LAYER_NORM::BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(time_step_index_received,
                                                                                                                    batch_size,
                                                                                                                    tmp_output_size,
                                                                                                                    tmp_ptr_array_inputs,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    ERR(L"Layer normalization (%d | %ls) is not managed in",
                                             layer_it->type_normalization,
                                             LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                        break;
            }
                
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }
    // Inference mode.
    else
    {
        // Weights.
        this->Forward_Pass__FC__OpenMP(time_step_index_received,
                                                              batch_size,
                                                              input_size_received,
                                                              tmp_output_size,
                                                              ptr_array_inputs_received,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                              tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(layer_it->Use__Bias())
        {
            this->Forward_Pass__Bias__OpenMP(time_step_index_received,
                                                                   batch_size,
                                                                   tmp_output_size,
                                                                   this->ptr_array_parameters + layer_it->first_bias_connection_index,
                                                                   tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
        
        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
        // Normalization before activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation)
        {
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(time_step_index_received,
                                                                                                           batch_size,
                                                                                                           tmp_output_size,
                                                                                                           tmp_ptr_array_inputs,
                                                                                                           tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                           tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                           tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                           tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                           tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__OpenMP(time_step_index_received,
                                                                               batch_size,
                                                                               tmp_output_size,
                                                                               this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                               tmp_ptr_array_inputs,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation == false)
        {
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(time_step_index_received,
                                                                                                       batch_size,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }
    
    // If the state of propagation is strictly at training.
    if(this->type_state_propagation == PROPAGATION::TRAINING)
    {
        // Dropout.
        switch(layer_it->type_dropout)
        {
            case LAYER_DROPOUT::BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(layer_it->ptr_array__mask__dropout__bernoulli,
                                                                                                            time_step_index_received,
                                                                                                            batch_size,
                                                                                                            tmp_output_size,
                                                                                                            tmp_ptr_array_inputs);
                    break;
            case LAYER_DROPOUT::BERNOULLI_INVERTED:
                this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(layer_it->ptr_array__mask__dropout__bernoulli,
                                                                                                          time_step_index_received,
                                                                                                          batch_size,
                                                                                                          tmp_output_size,
                                                                                                          layer_it->dropout_values[0] == 0_r ? 0_r : 1_r / layer_it->dropout_values[0],
                                                                                                          tmp_ptr_array_inputs);
                    break;
            case LAYER_DROPOUT::GAUSSIAN:
                this->Forward_Pass__Dropout__Gaussian__OpenMP(time_step_index_received,
                                                                                           batch_size,
                                                                                           tmp_output_size,
                                                                                           layer_it->dropout_values[0],
                                                                                           tmp_ptr_array_inputs);
                    break;
            case LAYER_DROPOUT::UOUT:
                this->Forward_Pass__Dropout__Uout__OpenMP(time_step_index_received,
                                                                                    batch_size,
                                                                                    tmp_output_size,
                                                                                    layer_it->dropout_values[0],
                                                                                    tmp_ptr_array_inputs);
                    break;
            default: break;
        }

        // k-Sparse.
        if(layer_it->Use__K_Sparsity())
        {
            this->Sparse_K_Filter__OpenMP(time_step_index_received,
                                                           batch_size,
                                                           tmp_output_size,
                                                           layer_it->k_sparsity,
                                                           layer_it->ptr_array_k_sparse_activities,
                                                           tmp_ptr_array_inputs);
        }
    }
    // Inference mode.
    else
    {
        // Dropout.
        switch(layer_it->type_dropout)
        {
            case LAYER_DROPOUT::BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(time_step_index_received,
                                                                                                            batch_size,
                                                                                                            tmp_output_size,
                                                                                                            layer_it->dropout_values[0],
                                                                                                            tmp_ptr_array_inputs);
                    break;
            default: break;
        }

        // k-Sparse.
        if(layer_it->Use__K_Sparsity())
        {
            this->Sparse_K_Filter__OpenMP(time_step_index_received,
                                                           batch_size,
                                                           tmp_output_size,
                                                           static_cast<size_t>(layer_it->alpha_sparsity * static_cast<real>(layer_it->k_sparsity)),
                                                           layer_it->ptr_array_k_sparse_activities,
                                                           tmp_ptr_array_inputs);
        }
    }
}

void Model::Forward_Pass__Encode__FC__OpenMP(size_t const time_step_index_received,
                                                                                      size_t const batch_size,
                                                                                      size_t const input_size_received,
                                                                                      var const *const ptr_array_inputs_received,
                                                                                      Layer *const layer_it)
{
    Neuron_unit *const tmp_ptr_layer_first_neuron_unit(layer_it->ptr_array_neuron_units);
    
    AF_unit *const tmp_ptr_layer_first_AF_unit(layer_it->ptr_array_AF_units);
    AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(layer_it->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(layer_it->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));
    
    var *tmp_ptr_array_inputs;

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // Weights.
    this->Forward_Pass__FC__OpenMP(time_step_index_received,
                                                            batch_size,
                                                            input_size_received,
                                                            tmp_output_size,
                                                            ptr_array_inputs_received,
                                                            this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                            tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
    // Bias.
    if(layer_it->Use__Bias())
    {
        this->Forward_Pass__Bias__OpenMP(time_step_index_received,
                                                                batch_size,
                                                                tmp_output_size,
                                                                this->ptr_array_parameters + layer_it->first_bias_connection_index,
                                                                tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
    }
        
    // Store the new inputs (summation).
    tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
    // Normalization before activation.
    if(layer_it->Use__Normalization()
      &&
      layer_it->use_layer_normalization_before_activation)
    {
        this->Forward_Pass__Batch_Normalization__Inference__OpenMP(time_step_index_received,
                                                                                                    batch_size,
                                                                                                    tmp_output_size,
                                                                                                    tmp_ptr_array_inputs,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
        // Store the new inputs (value normalize).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
    }
        
    if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
    {
        // Recurrent activation function.
        this->Forward_Pass__FC_Ind_RNN__OpenMP(time_step_index_received,
                                                                            batch_size,
                                                                            tmp_output_size,
                                                                            this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                            tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                            tmp_ptr_array_inputs,
                                                                            tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

        // Activation function.
        switch(layer_it->type_activation)
        {
            case LAYER_ACTIVATION::SYMMETRIC:
            case LAYER_ACTIVATION::ASYMMETRIC:
            case LAYER_ACTIVATION::RECTIFIER:
            case LAYER_ACTIVATION::SELF_NORMALIZATION:
                this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                        batch_size,
                                                                        tmp_output_size,
                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                    break;
            case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                        batch_size,
                                                                                        tmp_output_size,
                                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                        tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                    break;
            default:
                ERR(L"Layer activation (%d | %ls) is not managed in",
                                            layer_it->type_activation,
                                            LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                    break;
        }

        // Store the new inputs (value).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
    }
    else
    {
        // Activation function.
        switch(layer_it->type_activation)
        {
            case LAYER_ACTIVATION::SYMMETRIC:
            case LAYER_ACTIVATION::ASYMMETRIC:
            case LAYER_ACTIVATION::RECTIFIER:
            case LAYER_ACTIVATION::SELF_NORMALIZATION:
                this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                        batch_size,
                                                                        tmp_output_size,
                                                                        tmp_ptr_array_inputs,
                                                                        tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                        tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                    break;
            case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                        batch_size,
                                                                                        tmp_output_size,
                                                                                        tmp_ptr_array_inputs,
                                                                                        tmp_ptr_layer_first_AF_unit->ptr_array_values);
                    break;
            default:
                ERR(L"Layer activation (%d | %ls) is not managed in",
                                            layer_it->type_activation,
                                            LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                    break;
        }

        // Store the new inputs (value).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
    }

    // Normalization after activation.
    if(layer_it->Use__Normalization()
      &&
      layer_it->use_layer_normalization_before_activation == false)
        {
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(time_step_index_received,
                                                                                                       batch_size,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    
    // If the state of propagation is strictly at training && Input AE layer.
    if(this->type_state_propagation == PROPAGATION::TRAINING
      &&
      layer_it == this->ptr_array_layers + (this->pre_training_level - 1_UZ))
    {
        // Dropout.
        switch(layer_it->type_dropout)
        {
            case LAYER_DROPOUT::BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(layer_it->ptr_array__mask__dropout__bernoulli,
                                                                                                            time_step_index_received,
                                                                                                            batch_size,
                                                                                                            tmp_output_size,
                                                                                                            tmp_ptr_array_inputs);
                    break;
            case LAYER_DROPOUT::BERNOULLI_INVERTED:
                this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(layer_it->ptr_array__mask__dropout__bernoulli,
                                                                                                            time_step_index_received,
                                                                                                            batch_size,
                                                                                                            tmp_output_size,
                                                                                                            layer_it->dropout_values[0] == 0_r ? 0_r : 1_r / layer_it->dropout_values[0],
                                                                                                            tmp_ptr_array_inputs);
                    break;
            case LAYER_DROPOUT::GAUSSIAN:
                this->Forward_Pass__Dropout__Gaussian__OpenMP(time_step_index_received,
                                                                                                batch_size,
                                                                                                tmp_output_size,
                                                                                                layer_it->dropout_values[0],
                                                                                                tmp_ptr_array_inputs);
                    break;
            case LAYER_DROPOUT::UOUT:
                this->Forward_Pass__Dropout__Uout__OpenMP(time_step_index_received,
                                                                                        batch_size,
                                                                                        tmp_output_size,
                                                                                        layer_it->dropout_values[0],
                                                                                        tmp_ptr_array_inputs);
                    break;
            default: break;
        }
    }
    // Inference mode.
    else
    {
        // Dropout.
        switch(layer_it->type_dropout)
        {
            case LAYER_DROPOUT::BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(time_step_index_received,
                                                                                                              batch_size,
                                                                                                              tmp_output_size,
                                                                                                              layer_it->dropout_values[0],
                                                                                                              tmp_ptr_array_inputs);
                    break;
            default: break;
        }
    }

    // k-Sparse.
    if(layer_it->Use__K_Sparsity())
    {
        this->Sparse_K_Filter__OpenMP(time_step_index_received,
                                                   batch_size, tmp_output_size,
          static_cast<size_t>(layer_it->alpha_sparsity *
                              static_cast<real>(layer_it->k_sparsity)),
                                                   layer_it->ptr_array_k_sparse_activities,
                                                   tmp_ptr_array_inputs);
    }
}

void Model::Forward_Pass__Code__FC__OpenMP(size_t const time_step_index_received,
                                                                                        size_t const batch_size,
                                                                                        size_t const input_size_received,
                                                                                        var const *const ptr_array_inputs_received,
                                                                                        Layer *const layer_it)
{
    Neuron_unit *const tmp_ptr_layer_first_neuron_unit(layer_it->ptr_array_neuron_units);
    
    AF_unit *const tmp_ptr_layer_first_AF_unit(layer_it->ptr_array_AF_units);
    AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(layer_it->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(layer_it->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));
    
    var *tmp_ptr_array_inputs;

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= PROPAGATION::TRAINING)
    {
        // Weights.
        this->Forward_Pass__FC__OpenMP(time_step_index_received,
                                                              batch_size,
                                                              input_size_received,
                                                              tmp_output_size,
                                                              ptr_array_inputs_received,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                              tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(layer_it->Use__Bias())
        {
            this->Forward_Pass__Bias__OpenMP(time_step_index_received,
                                                                   batch_size,
                                                                   tmp_output_size,
                                                                   this->ptr_array_parameters + layer_it->first_bias_connection_index,
                                                                   tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
        
        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
        // Normalization before activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation)
        {
            switch(layer_it->type_normalization)
            {
                case LAYER_NORM::BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(time_step_index_received,
                                                                                                             batch_size,
                                                                                                             tmp_output_size,
                                                                                                             tmp_ptr_array_inputs,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case LAYER_NORM::BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(time_step_index_received,
                                                                                                                batch_size,
                                                                                                                tmp_output_size,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    ERR(L"Layer normalization (%d | %ls) is not managed in",
                                             layer_it->type_normalization,
                                             LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                        break;
            }
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__OpenMP(time_step_index_received,
                                                                               batch_size,
                                                                               tmp_output_size,
                                                                               this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                               tmp_ptr_array_inputs,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation == false)
        {
            switch(layer_it->type_normalization)
            {
                case LAYER_NORM::BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(time_step_index_received,
                                                                                                             batch_size,
                                                                                                             tmp_output_size,
                                                                                                             tmp_ptr_array_inputs,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case LAYER_NORM::BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(time_step_index_received,
                                                                                                                batch_size,
                                                                                                                tmp_output_size,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    ERR(L"Layer normalization (%d | %ls) is not managed in",
                                             layer_it->type_normalization,
                                             LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                        break;
            }
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }
    // Inference mode.
    else
    {
        // Weights.
        this->Forward_Pass__FC__OpenMP(time_step_index_received,
                                                              batch_size,
                                                              input_size_received,
                                                              tmp_output_size,
                                                              ptr_array_inputs_received,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                              tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(layer_it->Use__Bias())
        {
            this->Forward_Pass__Bias__OpenMP(time_step_index_received,
                                                                   batch_size,
                                                                   tmp_output_size,
                                                                   this->ptr_array_parameters + layer_it->first_bias_connection_index,
                                                                   tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
        
        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
        // Normalization before activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation)
        {
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(time_step_index_received,
                                                                                                       batch_size,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__OpenMP(time_step_index_received,
                                                                               batch_size,
                                                                               tmp_output_size,
                                                                               this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                               tmp_ptr_array_inputs,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation == false)
        {
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(time_step_index_received,
                                                                                                       batch_size,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }
    
    // If the state of propagation is strictly at training.
    if(this->type_state_propagation == PROPAGATION::TRAINING)
    {
        // Dropout.
        if(layer_it->Use__Coded_Dropout())
        {
            switch(layer_it->type_dropout)
            {
                case LAYER_DROPOUT::BERNOULLI:
                    this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(layer_it->ptr_array__mask__dropout__bernoulli,
                                                                                                              time_step_index_received,
                                                                                                              batch_size,
                                                                                                              tmp_output_size,
                                                                                                              tmp_ptr_array_inputs);
                        break;
                case LAYER_DROPOUT::BERNOULLI_INVERTED:
                    this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(layer_it->ptr_array__mask__dropout__bernoulli,
                                                                                                              time_step_index_received,
                                                                                                              batch_size,
                                                                                                              tmp_output_size,
                                                                                                              layer_it->dropout_values[0] == 0_r ? 0_r : 1_r / layer_it->dropout_values[0],
                                                                                                              tmp_ptr_array_inputs);
                        break;
                case LAYER_DROPOUT::GAUSSIAN:
                    this->Forward_Pass__Dropout__Gaussian__OpenMP(time_step_index_received,
                                                                                                 batch_size,
                                                                                                 tmp_output_size,
                                                                                                 layer_it->dropout_values[0],
                                                                                                 tmp_ptr_array_inputs);
                        break;
                case LAYER_DROPOUT::UOUT:
                    this->Forward_Pass__Dropout__Uout__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          layer_it->dropout_values[0],
                                                                                          tmp_ptr_array_inputs);
                        break;
                default: break;
            }
        }
        else
        {
            switch(layer_it->type_dropout)
            {
                case LAYER_DROPOUT::BERNOULLI:
                    this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(time_step_index_received,
                                                                                                                batch_size,
                                                                                                                tmp_output_size,
                                                                                                                layer_it->dropout_values[0],
                                                                                                                tmp_ptr_array_inputs);
                        break;
                default: break;
            }
        }

        // k-Sparse.
        if(layer_it->Use__K_Sparsity())
        {
            this->Sparse_K_Filter__OpenMP(time_step_index_received,
                                                           batch_size,
                                                           tmp_output_size,
                                                           layer_it->k_sparsity,
                                                           layer_it->ptr_array_k_sparse_activities,
                                                           tmp_ptr_array_inputs);
        }
    }
    // Inference mode.
    else
    {
        // Dropout.
        switch(layer_it->type_dropout)
        {
            case LAYER_DROPOUT::BERNOULLI:
                this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(time_step_index_received,
                                                                                                              batch_size,
                                                                                                              tmp_output_size,
                                                                                                              layer_it->dropout_values[0],
                                                                                                              tmp_ptr_array_inputs);
                    break;
            default: break;
        }

        // k-Sparse.
        if(layer_it->Use__K_Sparsity())
        {
            this->Sparse_K_Filter__OpenMP(time_step_index_received,
                                                       batch_size,
                                                       tmp_output_size,
                                                       static_cast<size_t>(layer_it->alpha_sparsity * static_cast<real>(layer_it->k_sparsity)),
                                                       layer_it->ptr_array_k_sparse_activities,
                                                       tmp_ptr_array_inputs);
        }
    }
}

void Model::Forward_Pass__Decode__FC__OpenMP(size_t const time_step_index_received,
                                                                                           size_t const batch_size,
                                                                                           size_t const input_size_received,
                                                                                           var const *const ptr_array_inputs_received,
                                                                                           Layer *const layer_it)
{
    Neuron_unit *const tmp_ptr_layer_first_neuron_unit(layer_it->ptr_array_neuron_units);
    
    AF_unit *const tmp_ptr_layer_first_AF_unit(layer_it->ptr_array_AF_units);
    AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(layer_it->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(layer_it->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));
    
    var *tmp_ptr_array_inputs;
    
    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= PROPAGATION::TRAINING)
    {
        // Weights.
        this->Forward_Pass__FC__OpenMP(time_step_index_received,
                                                              batch_size,
                                                              input_size_received,
                                                              tmp_output_size,
                                                              ptr_array_inputs_received,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                              tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(layer_it->Use__Bias())
        {
            this->Forward_Pass__Bias__OpenMP(time_step_index_received,
                                                                   batch_size,
                                                                   tmp_output_size,
                                                                   this->ptr_array_parameters + layer_it->first_bias_connection_index,
                                                                   tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
        
        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
        // Normalization before activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation)
        {
            switch(layer_it->type_normalization)
            {
                case LAYER_NORM::BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(time_step_index_received,
                                                                                                             batch_size,
                                                                                                             tmp_output_size,
                                                                                                             tmp_ptr_array_inputs,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case LAYER_NORM::BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(time_step_index_received,
                                                                                                                batch_size,
                                                                                                                tmp_output_size,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    ERR(L"Layer normalization (%d | %ls) is not managed in",
                                             layer_it->type_normalization,
                                             LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                        break;
            }
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__OpenMP(time_step_index_received,
                                                                               batch_size,
                                                                               tmp_output_size,
                                                                               this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                               tmp_ptr_array_inputs,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation == false)
        {
            switch(layer_it->type_normalization)
            {
                case LAYER_NORM::BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(time_step_index_received,
                                                                                                             batch_size,
                                                                                                             tmp_output_size,
                                                                                                             tmp_ptr_array_inputs,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case LAYER_NORM::BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(time_step_index_received,
                                                                                                                batch_size,
                                                                                                                tmp_output_size,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    ERR(L"Layer normalization (%d | %ls) is not managed in",
                                             layer_it->type_normalization,
                                             LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                        break;
            }
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }
    // Inference mode.
    else
    {
        // Weights.
        this->Forward_Pass__FC__OpenMP(time_step_index_received,
                                                              batch_size,
                                                              input_size_received,
                                                              tmp_output_size,
                                                              ptr_array_inputs_received,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                              tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(layer_it->Use__Bias())
        {
            this->Forward_Pass__Bias__OpenMP(time_step_index_received,
                                                                   batch_size,
                                                                   tmp_output_size,
                                                                   this->ptr_array_parameters + layer_it->first_bias_connection_index,
                                                                   tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
        
        // Store the new inputs (summation).
        tmp_ptr_array_inputs = tmp_ptr_layer_first_neuron_unit->ptr_array_summations;
        
        // Normalization before activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation)
        {
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(time_step_index_received,
                                                                                                       batch_size,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Forward_Pass__FC_Ind_RNN__OpenMP(time_step_index_received,
                                                                               batch_size,
                                                                               tmp_output_size,
                                                                               this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                               tmp_ptr_array_inputs,
                                                                               tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                          tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
        }
        else
        {
            // Activation function.
            switch(layer_it->type_activation)
            {
                case LAYER_ACTIVATION::SYMMETRIC:
                case LAYER_ACTIVATION::ASYMMETRIC:
                case LAYER_ACTIVATION::RECTIFIER:
                case LAYER_ACTIVATION::SELF_NORMALIZATION:
                    this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                          batch_size,
                                                                          tmp_output_size,
                                                                          tmp_ptr_array_inputs,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                          tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                        break;
                case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                    this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                          batch_size,
                                                                                          tmp_output_size,
                                                                                          tmp_ptr_array_inputs,
                                                                                          tmp_ptr_layer_first_AF_unit->ptr_array_values);
                        break;
                default:
                    ERR(L"Layer activation (%d | %ls) is not managed in",
                                             layer_it->type_activation,
                                             LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                        break;
            }

            // Store the new inputs (value).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
        }

        // Normalization after activation.
        if(layer_it->Use__Normalization()
          &&
          layer_it->use_layer_normalization_before_activation == false)
        {
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(time_step_index_received,
                                                                                                       batch_size,
                                                                                                       tmp_output_size,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
    }

    // Dropout.
    switch(layer_it->type_dropout)
    {
        case LAYER_DROPOUT::BERNOULLI:
            this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(time_step_index_received,
                                                                                                          batch_size,
                                                                                                          tmp_output_size,
                                                                                                          layer_it->dropout_values[0],
                                                                                                          tmp_ptr_array_inputs);
                break;
        default: break;
    }
}

void Model::Forward_Pass__LSTM__OpenMP(long long int const time_step_index_received,
                                                                                 long long int const tmp_time_step_reverse_direction,
                                                                                 long long int const tmp_time_step_start,
                                                                                 size_t const batch_size,
                                                                                 size_t const input_size_received,
                                                                                 var const *const ptr_array_inputs_received,
                                                                                 Layer *const layer_it)
{
    BlockUnit *const tmp_ptr_layer_first_block_unit(layer_it->ptr_array_block_units);
    
    CellUnit *const tmp_ptr_layer_first_cell_unit(layer_it->ptr_array_cell_units);
    
    size_t const tmp_number_block_units(static_cast<size_t>(layer_it->ptr_last_block_unit - tmp_ptr_layer_first_block_unit)),
                       tmp_number_cell_units(static_cast<size_t>(layer_it->ptr_last_cell_unit - tmp_ptr_layer_first_cell_unit));

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= PROPAGATION::TRAINING)
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__OpenMP(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                layer_it);
            
        // Normalization.
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }

        // If the state of propagation is strictly at training.
        // Gates activation cell, input, forget and state.
        if(this->type_state_propagation == PROPAGATION::TRAINING
          &&
          layer_it->Use__Dropout__Zoneout())
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__OpenMP(time_step_index_received,
                                                                                                                    tmp_time_step_reverse_direction,
                                                                                                                    tmp_time_step_start,
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                                    layer_it);
        }
        else
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__OpenMP(time_step_index_received,
                                                                                                    tmp_time_step_reverse_direction,
                                                                                                    tmp_time_step_start,
                                                                                                    batch_size,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_number_cell_units,
                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                    layer_it);
        }

        // Normalization.
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            default: break;
        }

        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__OpenMP(time_step_index_received,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                layer_it->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                layer_it);
            
        // Normalization.
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_normalizes);
                }    
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }
        
        // If the state of propagation is strictly at training.
        // Gate activation, output.
        if(this->type_state_propagation == PROPAGATION::TRAINING
          &&
          layer_it->Use__Dropout__Zoneout())
        {
            this->Forward_Pass__LSTM__Output__Zoneout__OpenMP(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                                layer_it);
        }
        else
        {
            this->Forward_Pass__LSTM__Output__OpenMP(time_step_index_received,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                layer_it);
        }
    }
    // Inference mode.
    else
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__OpenMP(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                layer_it);

        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Block input, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_normalizes);
                
            // Input gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Forget gate, input.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Block input, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Forget gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gates activation cell, input, forget and state.
        this->Forward_Pass__LSTM__Gates_CIF_AF_State__OpenMP(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                layer_it->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                layer_it);

        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Cell state activate.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_normalizes);
        }
            
        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__OpenMP(time_step_index_received,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                layer_it->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                layer_it);
            
        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Output gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Output gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gate activation, output.
        this->Forward_Pass__LSTM__Output__OpenMP(time_step_index_received,
                                                                            batch_size,
                                                                            tmp_number_block_units,
                                                                            tmp_number_cell_units,
                                                                            layer_it->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                            layer_it->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                            layer_it);
    }
}

void Model::Forward_Pass__Encode__LSTM__OpenMP(long long int const time_step_index_received,
                                                                                                long long int const tmp_time_step_reverse_direction,
                                                                                                long long int const tmp_time_step_start,
                                                                                                size_t const batch_size,
                                                                                                size_t const input_size_received,
                                                                                                var const *const ptr_array_inputs_received,
                                                                                                Layer *const layer_it)
{
    BlockUnit *const tmp_ptr_layer_first_block_unit(layer_it->ptr_array_block_units);
    
    CellUnit *const tmp_ptr_layer_first_cell_unit(layer_it->ptr_array_cell_units);
    
    size_t const tmp_number_block_units(static_cast<size_t>(layer_it->ptr_last_block_unit - tmp_ptr_layer_first_block_unit)),
                       tmp_number_cell_units(static_cast<size_t>(layer_it->ptr_last_cell_unit - tmp_ptr_layer_first_cell_unit));

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // Training mode && Input AE layer.
    if(this->type_state_propagation >= PROPAGATION::TRAINING
      &&
      layer_it == this->ptr_array_layers + (this->pre_training_level - 1_UZ))
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__OpenMP(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                layer_it);
            
        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Block input, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_normalizes);
                
            // Input gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Forget gate, input.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Block input, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Forget gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
        
        // If the state of propagation is strictly at training.
        // Gates activation cell, input, forget and state.
        if(this->type_state_propagation == PROPAGATION::TRAINING
          &&
          layer_it->Use__Dropout__Zoneout())
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__OpenMP(time_step_index_received,
                                                                                                                    tmp_time_step_reverse_direction,
                                                                                                                    tmp_time_step_start,
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                                    layer_it);
        }
        else
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__OpenMP(time_step_index_received,
                                                                                                    tmp_time_step_reverse_direction,
                                                                                                    tmp_time_step_start,
                                                                                                    batch_size,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_number_cell_units,
                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                    layer_it);
        }

        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Cell state activate.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_normalizes);
        }
            
        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__OpenMP(time_step_index_received,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                layer_it->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                layer_it);
            
        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Output gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Output gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
        
        // If the state of propagation is strictly at training.
        // Gate activation, output.
        if(this->type_state_propagation == PROPAGATION::TRAINING
          &&
          layer_it->Use__Dropout__Zoneout())
        {
            this->Forward_Pass__LSTM__Output__Zoneout__OpenMP(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                                layer_it);
        }
        else
        {
            this->Forward_Pass__LSTM__Output__OpenMP(time_step_index_received,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                layer_it);
        }
    }
    // Inference mode.
    else
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__OpenMP(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                layer_it);

        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Block input, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_normalizes);
                
            // Input gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Forget gate, input.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Block input, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Forget gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gates activation cell, input, forget and state.
        this->Forward_Pass__LSTM__Gates_CIF_AF_State__OpenMP(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                layer_it->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                layer_it);

        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Cell state activate.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_normalizes);
        }
            
        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__OpenMP(time_step_index_received,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                layer_it->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                layer_it);
            
        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Output gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Output gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gate activation, output.
        this->Forward_Pass__LSTM__Output__OpenMP(time_step_index_received,
                                                                            batch_size,
                                                                            tmp_number_block_units,
                                                                            tmp_number_cell_units,
                                                                            layer_it->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                            layer_it->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                            layer_it);
    }
}

void Model::Forward_Pass__Code__LSTM__OpenMP(long long int const time_step_index_received,
                                                                                       long long int const tmp_time_step_reverse_direction,
                                                                                       long long int const tmp_time_step_start,
                                                                                       size_t const batch_size,
                                                                                       size_t const input_size_received,
                                                                                       var const *const ptr_array_inputs_received,
                                                                                       Layer *const layer_it)
{
    BlockUnit *const tmp_ptr_layer_first_block_unit(layer_it->ptr_array_block_units);
    
    CellUnit *const tmp_ptr_layer_first_cell_unit(layer_it->ptr_array_cell_units);
    
    size_t const tmp_number_block_units(static_cast<size_t>(layer_it->ptr_last_block_unit - tmp_ptr_layer_first_block_unit)),
                       tmp_number_cell_units(static_cast<size_t>(layer_it->ptr_last_cell_unit - tmp_ptr_layer_first_cell_unit));

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= PROPAGATION::TRAINING)
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__OpenMP(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                layer_it);
            
        // Normalization.
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }
        
        // If the state of propagation is strictly at training.
        // Gates activation cell, input, forget and state.
        if(this->type_state_propagation == PROPAGATION::TRAINING
          &&
          layer_it->Use__Dropout__Zoneout() && layer_it->Use__Coded_Dropout())
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__OpenMP(time_step_index_received,
                                                                                                                    tmp_time_step_reverse_direction,
                                                                                                                    tmp_time_step_start,
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                                    layer_it);
        }
        else
        {
            this->Forward_Pass__LSTM__Gates_CIF_AF_State__OpenMP(time_step_index_received,
                                                                                                    tmp_time_step_reverse_direction,
                                                                                                    tmp_time_step_start,
                                                                                                    batch_size,
                                                                                                    tmp_number_block_units,
                                                                                                    tmp_number_cell_units,
                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                    layer_it->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                    layer_it);
        }

        // Normalization.
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            default: break;
        }

        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__OpenMP(time_step_index_received,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                layer_it->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                layer_it);
            
        // Normalization.
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_normalizes);
                }    
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }
        
        // If the state of propagation is strictly at training.
        // Gate activation, output.
        if(this->type_state_propagation == PROPAGATION::TRAINING
          &&
          layer_it->Use__Dropout__Zoneout() && layer_it->Use__Coded_Dropout())
        {
            this->Forward_Pass__LSTM__Output__Zoneout__OpenMP(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                                layer_it);
        }
        else
        {
            this->Forward_Pass__LSTM__Output__OpenMP(time_step_index_received,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                                layer_it->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                                layer_it);
        }
    }
    // Inference mode.
    else
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__OpenMP(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                layer_it);

        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Block input, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_normalizes);
                
            // Input gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Forget gate, input.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Block input, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Forget gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gates activation cell, input, forget and state.
        this->Forward_Pass__LSTM__Gates_CIF_AF_State__OpenMP(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                layer_it->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                layer_it);

        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Cell state activate.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_normalizes);
        }
            
        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__OpenMP(time_step_index_received,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                layer_it->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                layer_it);
            
        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Output gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Output gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gate activation, output.
        this->Forward_Pass__LSTM__Output__OpenMP(time_step_index_received,
                                                                            batch_size,
                                                                            tmp_number_block_units,
                                                                            tmp_number_cell_units,
                                                                            layer_it->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                            layer_it->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                            layer_it);
    }
}

void Model::Forward_Pass__Decode__LSTM__OpenMP(long long int const time_step_index_received,
                                                                                          long long int const tmp_time_step_reverse_direction,
                                                                                          long long int const tmp_time_step_start,
                                                                                          size_t const batch_size,
                                                                                          size_t const input_size_received,
                                                                                          var const *const ptr_array_inputs_received,
                                                                                          Layer *const layer_it)
{
    BlockUnit *const tmp_ptr_layer_first_block_unit(layer_it->ptr_array_block_units);
    
    CellUnit *const tmp_ptr_layer_first_cell_unit(layer_it->ptr_array_cell_units);
    
    size_t const tmp_number_block_units(static_cast<size_t>(layer_it->ptr_last_block_unit - tmp_ptr_layer_first_block_unit)),
                       tmp_number_cell_units(static_cast<size_t>(layer_it->ptr_last_cell_unit - tmp_ptr_layer_first_cell_unit));

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= PROPAGATION::TRAINING)
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__OpenMP(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                layer_it);
            
        // Normalization.
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                // Block input, input.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_cell_units,
                                                                                                                tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Forget gate, input.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Block input, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_cell_units,
                                                                                                                    tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Input gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_normalizes);
                        
                    // Forget gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                    batch_size,
                                                                                                                    tmp_number_block_units,
                                                                                                                    tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_shift,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_means,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_variances,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_mean_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_variance_average,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_r_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_d_correction,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_hats,
                                                                                                                    tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }

        // Gates activation cell, input, forget and state.
        this->Forward_Pass__LSTM__Gates_CIF_AF_State__OpenMP(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                layer_it->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                layer_it);

        // Normalization.
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_means,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_variances,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_hats,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                // Cell state activate.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_normalizes);
                    break;
            default: break;
        }

        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__OpenMP(time_step_index_received,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                layer_it->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                layer_it);
            
        // Normalization.
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_normalizes);
                }    
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                // Output gate, input.
                this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_means,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_variances,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_r_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_d_correction,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_hats,
                                                                                                            tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_normalizes);
                    
                if(time_step_index_received != tmp_time_step_start)
                {
                    // Output gate, recurrent.
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                                batch_size,
                                                                                                                tmp_number_block_units,
                                                                                                                tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_normalizes);
                }
                    break;
            default: break;
        }

        // Gate activation, output.
        this->Forward_Pass__LSTM__Output__OpenMP(time_step_index_received,
                                                                            batch_size,
                                                                            tmp_number_block_units,
                                                                            tmp_number_cell_units,
                                                                            layer_it->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                            layer_it->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                            layer_it);
    }
    // Inference mode.
    else
    {
        // Gates cell, input, forget.
        this->Forward_Pass__LSTM__Gates_CIFO__OpenMP(time_step_index_received,
                                                                                tmp_time_step_reverse_direction,
                                                                                tmp_time_step_start,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                input_size_received,
                                                                                ptr_array_inputs_received,
                                                                                layer_it);

        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Block input, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_summation_input_cell_input,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[0].normalized_batch_units.ptr_array_values_normalizes);
                
            // Input gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_inputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[3].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Forget gate, input.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_input_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[5].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Block input, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_cell_units,
                                                                                                            tmp_ptr_layer_first_cell_unit->ptr_summation_recurrent_cell_input,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[1].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Input gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_inputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[4].normalized_batch_units.ptr_array_values_normalizes);
                    
                // Forget gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_forgets_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[6].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gates activation cell, input, forget and state.
        this->Forward_Pass__LSTM__Gates_CIF_AF_State__OpenMP(time_step_index_received,
                                                                                                tmp_time_step_reverse_direction,
                                                                                                tmp_time_step_start,
                                                                                                batch_size,
                                                                                                tmp_number_block_units,
                                                                                                tmp_number_cell_units,
                                                                                                layer_it->Get__Array_Summations__Cell__Block_Input__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Input_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Forget_Gate__Input__Activation(),
                                                                                                layer_it->Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(),
                                                                                                layer_it);

        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Cell state activate.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_cell_units,
                                                                                                        tmp_ptr_layer_first_cell_unit->ptr_cell_state,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[2].normalized_batch_units.ptr_array_values_normalizes);
        }
            
        // Cell state activation.
        this->Forward_Pass__LSTM__States_AF__OpenMP(time_step_index_received,
                                                                                batch_size,
                                                                                tmp_number_block_units,
                                                                                tmp_number_cell_units,
                                                                                layer_it->Get__Array_Summations__Cell__Cell_State__Activation(),
                                                                                layer_it);
            
        // Batch normalization.
        if(layer_it->Use__Normalization())
        {
            // Output gate, input.
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                        batch_size,
                                                                                                        tmp_number_block_units,
                                                                                                        tmp_ptr_layer_first_block_unit->ptr_summation_input_outputs_gates,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_scale,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_shift,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_mean_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_variance_average,
                                                                                                        tmp_ptr_layer_first_normalized_unit[7].normalized_batch_units.ptr_array_values_normalizes);
                
            if(time_step_index_received != tmp_time_step_start)
            {
                // Output gate, recurrent.
                this->Forward_Pass__Batch_Normalization__Inference__OpenMP(static_cast<size_t>(time_step_index_received),
                                                                                                            batch_size,
                                                                                                            tmp_number_block_units,
                                                                                                            tmp_ptr_layer_first_block_unit->ptr_summation_recurrent_outputs_gates,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_scale,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_shift,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_mean_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_variance_average,
                                                                                                            tmp_ptr_layer_first_normalized_unit[8].normalized_batch_units.ptr_array_values_normalizes);
            }
        }
            
        // Gate activation, output.
        this->Forward_Pass__LSTM__Output__OpenMP(time_step_index_received,
                                                                            batch_size,
                                                                            tmp_number_block_units,
                                                                            tmp_number_cell_units,
                                                                            layer_it->Get__Array_Summations__Block__Output_Gate__Input__Activation(),
                                                                            layer_it->Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(),
                                                                            layer_it);
    }
}

void Model::Forward_Pass__Max_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                            size_t const batch_size,
                                                                                            size_t const input_size_received,
                                                                                            var const *const ptr_array_inputs_received,
                                                                                            Layer *const layer_it)
{
    Basic_indice_unit *const tmp_ptr_layer_first_basic_indice_unit(layer_it->ptr_array_basic_indice_units);
    
    this->Forward_Pass__Max_Pooling__OpenMP(time_step_index_received,
                                                                         batch_size,
                                                                         input_size_received,
                                                                         *layer_it->ptr_number_outputs,
                                                                         layer_it->pooling_values[0],
                                                                         layer_it->pooling_values[1],
                                                                         layer_it->pooling_values[2],
                                                                         layer_it->pooling_values[3],
                                                                         tmp_ptr_layer_first_basic_indice_unit->ptr_array_indices,
                                                                         ptr_array_inputs_received,
                                                                         tmp_ptr_layer_first_basic_indice_unit->ptr_array_values);
    
    layer_it->ptr_array_outputs = tmp_ptr_layer_first_basic_indice_unit->ptr_array_values;
}

void Model::Forward_Pass__Residual__OpenMP(size_t const batch_size, Layer *&layer_it)
{
    var *tmp_ptr_array_inputs;
    
    Layer const *const tmp_ptr_end_block_layer(layer_it + layer_it->block_depth + 1),
                               *prev_conn_layer;
    Layer *const tmp_ptr_residual_layer(layer_it);
    
    union Normalized_unit *const tmp_ptr_residual_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // First block layer.
    this->Forward_Pass__Residual__Layer__OpenMP(true,
                                                                              batch_size,
                                                                              ++layer_it);
    // |END| First block layer. |END|

    // Remaining layer(s).
    for(++layer_it; layer_it != tmp_ptr_end_block_layer; ++layer_it)
    {
        this->Forward_Pass__Residual__Layer__OpenMP(false,
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
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(0_UZ,
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
                        break;
                case LAYER_NORM::BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(0_UZ,
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
                        break;
                default:
                    ERR(L"Layer normalization (%d | %ls) is not managed in",
                                             layer_it->type_normalization,
                                             LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                        break;
            }
        }
        // Inference mode.
        else
        {
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(0_UZ,
                                                                                                      batch_size,
                                                                                                      *layer_it->ptr_number_outputs,
                                                                                                      tmp_ptr_array_inputs,
                                                                                                      tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                      tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                      tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                      tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                      tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
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
            this->Forward_Pass__Dropout__ShakeDrop__OpenMP(0_UZ,
                                                                                             batch_size,
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
            this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(0_UZ,
                                                                                                        batch_size,
                                                                                                        *layer_it->ptr_number_outputs,
                                                                                                        1_r - tmp_ptr_residual_layer->dropout_values[0],
                                                                                                        tmp_ptr_array_inputs);
        }
    }

    //  Zero-padded identity-mapping shortcut.
    this->Forward_Pass__Zero_Padded_Identity__OpenMP(0_UZ,
                                                                                   batch_size,
                                                                                   *prev_conn_layer->ptr_number_outputs, // Shortcut.
                                                                                   *layer_it->ptr_number_outputs, // Block, last layer.
                                                                                   tmp_ptr_residual_layer->pooling_values[2],
                                                                                   prev_conn_layer->ptr_array_outputs, // Shortcut.
                                                                                   tmp_ptr_array_inputs, // Block, last layer.
                                                                                   tmp_ptr_residual_layer->ptr_array_basic_units->ptr_array_values);
    // |END| Shortcut. |END|
}

void Model::Forward_Pass__Residual__Layer__OpenMP(bool const is_block_input_layer_received,
                                                                                                 size_t const batch_size,
                                                                                                 Layer *&layer_it)
{
    Layer const *const prev_conn_layer(layer_it->previous_connected_layers[0]);

    switch(layer_it->type_layer)
    {
        case LAYER::AVERAGE_POOLING:
            this->Forward_Pass__Average_Pooling__OpenMP(0_UZ,
                                                                                batch_size,
                                                                                *prev_conn_layer->ptr_number_outputs,
                                                                                prev_conn_layer->ptr_array_outputs,
                                                                                layer_it);
                break;
        case LAYER::FULLY_CONNECTED:
            this->Forward_Pass__Residual__FC__OpenMP(is_block_input_layer_received,
                                                                            0_UZ,
                                                                            batch_size,
                                                                            *prev_conn_layer->ptr_number_outputs,
                                                                            prev_conn_layer->ptr_array_outputs,
                                                                            layer_it);
                break;
        case LAYER::MAX_POOLING:
            this->Forward_Pass__Max_Pooling__OpenMP(0_UZ,
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

void Model::Forward_Pass__Residual__FC__OpenMP(bool const is_block_input_layer_received,
                                                                                             size_t const time_step_index_received,
                                                                                             size_t const batch_size,
                                                                                             size_t const input_size_received,
                                                                                             var const *const ptr_array_inputs_received,
                                                                                             Layer *const layer_it)
{
    Neuron_unit *const tmp_ptr_layer_first_neuron_unit(layer_it->ptr_array_neuron_units);
    
    AF_unit *const tmp_ptr_layer_first_AF_unit(layer_it->ptr_array_AF_units);
    AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(layer_it->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(layer_it->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));

    var *tmp_ptr_array_inputs(const_cast<var *>(ptr_array_inputs_received));

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // Training mode.
    if(this->type_state_propagation >= PROPAGATION::TRAINING)
    {
        // Normalization before activation.
        if(layer_it->Use__Normalization())
        {
            switch(layer_it->type_normalization)
            {
                case LAYER_NORM::BATCH_NORMALIZATION:
                    this->Forward_Pass__Batch_Normalization__Training__OpenMP(time_step_index_received,
                                                                                                             batch_size,
                                                                                                             input_size_received,
                                                                                                             tmp_ptr_array_inputs,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                             tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                case LAYER_NORM::BATCH_RENORMALIZATION:
                    this->Forward_Pass__Batch_Renormalization__Training__OpenMP(time_step_index_received,
                                                                                                                batch_size,
                                                                                                                input_size_received,
                                                                                                                tmp_ptr_array_inputs,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_d_correction,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
                        break;
                default:
                    ERR(L"Layer normalization (%d | %ls) is not managed in",
                                             layer_it->type_normalization,
                                             LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                        break;
            }
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(is_block_input_layer_received == false)
        {
            if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
            {
                // Recurrent activation function.
                this->Forward_Pass__FC_Ind_RNN__OpenMP(time_step_index_received,
                                                                                   batch_size,
                                                                                   input_size_received,
                                                                                   this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                                   tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                                   tmp_ptr_array_inputs,
                                                                                   tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

                // Activation function.
                switch(layer_it->type_activation)
                {
                    case LAYER_ACTIVATION::SYMMETRIC:
                    case LAYER_ACTIVATION::ASYMMETRIC:
                    case LAYER_ACTIVATION::RECTIFIER:
                    case LAYER_ACTIVATION::SELF_NORMALIZATION:
                        this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                              batch_size,
                                                                              input_size_received,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                            break;
                    case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                        this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                              batch_size,
                                                                                              input_size_received,
                                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                            break;
                    default:
                        ERR(L"Layer activation (%d | %ls) is not managed in",
                                                 layer_it->type_activation,
                                                 LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                            break;
                }

                // Store the new inputs (value).
                tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
            }
            else
            {
                // Activation function.
                switch(layer_it->type_activation)
                {
                    case LAYER_ACTIVATION::SYMMETRIC:
                    case LAYER_ACTIVATION::ASYMMETRIC:
                    case LAYER_ACTIVATION::RECTIFIER:
                    case LAYER_ACTIVATION::SELF_NORMALIZATION:
                        this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                              batch_size,
                                                                              input_size_received,
                                                                              tmp_ptr_array_inputs,
                                                                              tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                              tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                            break;
                    case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                        this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                              batch_size,
                                                                                              input_size_received,
                                                                                              tmp_ptr_array_inputs,
                                                                                              tmp_ptr_layer_first_AF_unit->ptr_array_values);
                            break;
                    default:
                        ERR(L"Layer activation (%d | %ls) is not managed in",
                                                 layer_it->type_activation,
                                                 LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                            break;
                }

                // Store the new inputs (value).
                tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
            }
            
            // If the state of propagation is strictly at training.
            if(this->type_state_propagation == PROPAGATION::TRAINING)
            {
                // Dropout.
                switch(layer_it->type_dropout)
                {
                    case LAYER_DROPOUT::BERNOULLI:
                        this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(layer_it->ptr_array__mask__dropout__bernoulli,
                                                                                                                    time_step_index_received,
                                                                                                                    batch_size,
                                                                                                                    input_size_received,
                                                                                                                    tmp_ptr_array_inputs);
                            break;
                    case LAYER_DROPOUT::BERNOULLI_INVERTED:
                        this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(layer_it->ptr_array__mask__dropout__bernoulli,
                                                                                                                  time_step_index_received,
                                                                                                                  batch_size,
                                                                                                                  input_size_received,
                                                                                                                  layer_it->dropout_values[0] == 0_r ? 0_r : 1_r / layer_it->dropout_values[0],
                                                                                                                  tmp_ptr_array_inputs);
                            break;
                    case LAYER_DROPOUT::GAUSSIAN:
                        this->Forward_Pass__Dropout__Gaussian__OpenMP(time_step_index_received,
                                                                                                       batch_size,
                                                                                                       input_size_received,
                                                                                                       layer_it->dropout_values[0],
                                                                                                       tmp_ptr_array_inputs);
                            break;
                    case LAYER_DROPOUT::UOUT:
                        this->Forward_Pass__Dropout__Uout__OpenMP(time_step_index_received,
                                                                                                batch_size,
                                                                                                input_size_received,
                                                                                                layer_it->dropout_values[0],
                                                                                                tmp_ptr_array_inputs);
                            break;
                    default: break;
                }

                // k-Sparse.
                if(layer_it->Use__K_Sparsity())
                {
                    this->Sparse_K_Filter__OpenMP(time_step_index_received,
                                                                     batch_size,
                                                                     input_size_received,
                                                                     layer_it->k_sparsity,
                                                                     layer_it->ptr_array_k_sparse_activities,
                                                                     tmp_ptr_array_inputs);
                }
            }
            // Inference mode.
            else
            {
                // Dropout.
                switch(layer_it->type_dropout)
                {
                    case LAYER_DROPOUT::BERNOULLI:
                        this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(time_step_index_received,
                                                                                                                      batch_size,
                                                                                                                      input_size_received,
                                                                                                                      layer_it->dropout_values[0],
                                                                                                                      tmp_ptr_array_inputs);
                            break;
                    default: break;
                }

                // k-Sparse.
                if(layer_it->Use__K_Sparsity())
                {
                    this->Sparse_K_Filter__OpenMP(time_step_index_received,
                                                                     batch_size,
                                                                     input_size_received,
                                                                     static_cast<size_t>(layer_it->alpha_sparsity * static_cast<real>(layer_it->k_sparsity)),
                                                                     layer_it->ptr_array_k_sparse_activities,
                                                                     tmp_ptr_array_inputs);
                }
            }
        }

        // Weights.
        this->Forward_Pass__FC__OpenMP(time_step_index_received,
                                                        batch_size,
                                                        input_size_received,
                                                        tmp_output_size,
                                                        tmp_ptr_array_inputs,
                                                        this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                        tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(layer_it->Use__Bias())
        {
            this->Forward_Pass__Bias__OpenMP(time_step_index_received,
                                                               batch_size,
                                                               tmp_output_size,
                                                               this->ptr_array_parameters + layer_it->first_bias_connection_index,
                                                               tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
    }
    // Inference mode.
    else
    {
        // Normalization before activation.
        if(layer_it->Use__Normalization())
        {
            this->Forward_Pass__Batch_Normalization__Inference__OpenMP(time_step_index_received,
                                                                                                       batch_size,
                                                                                                       input_size_received,
                                                                                                       tmp_ptr_array_inputs,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_shift,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_mean_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_variance_average,
                                                                                                       tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes);
            
            // Store the new inputs (value normalize).
            tmp_ptr_array_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_normalizes;
        }
        
        if(is_block_input_layer_received == false)
        {
            if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
            {
                // Recurrent activation function.
                this->Forward_Pass__FC_Ind_RNN__OpenMP(time_step_index_received,
                                                                                   batch_size,
                                                                                   input_size_received,
                                                                                   this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                                   tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                                   tmp_ptr_array_inputs,
                                                                                   tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs);

                // Activation function.
                switch(layer_it->type_activation)
                {
                    case LAYER_ACTIVATION::SYMMETRIC:
                    case LAYER_ACTIVATION::ASYMMETRIC:
                    case LAYER_ACTIVATION::RECTIFIER:
                    case LAYER_ACTIVATION::SELF_NORMALIZATION:
                        this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                              batch_size,
                                                                              input_size_received,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function);
                            break;
                    case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                        this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                              batch_size,
                                                                                              input_size_received,
                                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_pre_AFs,
                                                                                              tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs);
                            break;
                    default:
                        ERR(L"Layer activation (%d | %ls) is not managed in",
                                                 layer_it->type_activation,
                                                 LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                            break;
                }

                // Store the new inputs (value).
                tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs;
            }
            else
            {
                // Activation function.
                switch(layer_it->type_activation)
                {
                    case LAYER_ACTIVATION::SYMMETRIC:
                    case LAYER_ACTIVATION::ASYMMETRIC:
                    case LAYER_ACTIVATION::RECTIFIER:
                    case LAYER_ACTIVATION::SELF_NORMALIZATION:
                        this->Forward_Pass__FC_AF__OpenMP(time_step_index_received,
                                                                              batch_size,
                                                                              input_size_received,
                                                                              tmp_ptr_array_inputs,
                                                                              tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                              tmp_ptr_layer_first_AF_unit->ptr_type_activation_function);
                            break;
                    case LAYER_ACTIVATION::SOFTMAX: // Only output layer.
                        this->Forward_Pass__FC_AF__Softmax__OpenMP(time_step_index_received,
                                                                                              batch_size,
                                                                                              input_size_received,
                                                                                              tmp_ptr_array_inputs,
                                                                                              tmp_ptr_layer_first_AF_unit->ptr_array_values);
                            break;
                    default:
                        ERR(L"Layer activation (%d | %ls) is not managed in",
                                                 layer_it->type_activation,
                                                 LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
                            break;
                }

                // Store the new inputs (value).
                tmp_ptr_array_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_values;
            }

            // Dropout.
            switch(layer_it->type_dropout)
            {
                case LAYER_DROPOUT::BERNOULLI:
                    this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(time_step_index_received,
                                                                                                                  batch_size,
                                                                                                                  input_size_received,
                                                                                                                  layer_it->dropout_values[0],
                                                                                                                  tmp_ptr_array_inputs);
                        break;
                default: break;
            }

            // k-Sparse.
            if(layer_it->Use__K_Sparsity())
            {
                this->Sparse_K_Filter__OpenMP(time_step_index_received,
                                                           batch_size,
                                                           input_size_received,
                                                           static_cast<size_t>(layer_it->alpha_sparsity * static_cast<real>(layer_it->k_sparsity)),
                                                           layer_it->ptr_array_k_sparse_activities,
                                                           tmp_ptr_array_inputs);
            }
        }

        // Weights.
        this->Forward_Pass__FC__OpenMP(time_step_index_received,
                                                        batch_size,
                                                        input_size_received,
                                                        tmp_output_size,
                                                        tmp_ptr_array_inputs,
                                                        this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                        tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        
        // Bias.
        if(layer_it->Use__Bias())
        {
            this->Forward_Pass__Bias__OpenMP(time_step_index_received,
                                                               batch_size,
                                                               tmp_output_size,
                                                               this->ptr_array_parameters + layer_it->first_bias_connection_index,
                                                               tmp_ptr_layer_first_neuron_unit->ptr_array_summations);
        }
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Forward_Pass__Average_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                                 size_t const batch_size,
                                                                                                 size_t const input_size_received,
                                                                                                 size_t const output_size_received,
                                                                                                 size_t const kernel_size_received,
                                                                                                 size_t const stride_received,
                                                                                                 size_t const padding_received,
                                                                                                 size_t const dilation_received,
                                                                                                 var const *const ptr_array_inputs_received,
                                                                                                 var *const ptr_array_outputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_output_timed_batched_index(this->batch_size * output_size_received * time_step_index_received),
                       tmp_input_padded_half(input_size_received + padding_received);
    size_t tmp_kernel_index,
              tmp_shift_index,
              tmp_output_index;
    
    real const tmp_scale(1_r / static_cast<real>(kernel_size_received));
    var const *tmp_ptr_array_inputs;
    var *tmp_ptr_array_outputs,
         tmp_summation;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_outputs = ptr_array_outputs_received + static_cast<size_t>(tmp_example_index__int) * output_size_received + tmp_output_timed_batched_index;
        
        for(tmp_output_index = 0_UZ; tmp_output_index != output_size_received; ++tmp_output_index)
        {
            tmp_summation = 0_r;
            
            for(tmp_kernel_index = 0_UZ; tmp_kernel_index != kernel_size_received; ++tmp_kernel_index)
            {
                tmp_shift_index = tmp_output_index * stride_received + tmp_kernel_index * dilation_received;

                if(tmp_shift_index < padding_received || tmp_shift_index >= tmp_input_padded_half) { continue; }

                tmp_summation += tmp_ptr_array_inputs[tmp_shift_index - padding_received];
            }
            
            tmp_ptr_array_outputs[tmp_output_index] = tmp_summation * tmp_scale;
        }
    }
}

void Model::Forward_Pass__Bias__OpenMP(size_t const time_step_index_received,
                                                                               size_t const batch_size,
                                                                               size_t const output_size_received,
                                                                               var const *const ptr_array_bias_received,
                                                                               var *const ptr_array_outputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_unit_timed_batched_index(this->batch_size * output_size_received * time_step_index_received);
    size_t tmp_unit_index;
    
    var *tmp_ptr_array_layer_outputs;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_layer_outputs = ptr_array_outputs_received + static_cast<size_t>(tmp_example_index__int) * output_size_received + tmp_unit_timed_batched_index;
        
        for(tmp_unit_index = 0_UZ; tmp_unit_index != output_size_received; ++tmp_unit_index) { tmp_ptr_array_layer_outputs[tmp_unit_index] += ptr_array_bias_received[tmp_unit_index]; }
    }
}

void Model::Forward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                                                             size_t const batch_size,
                                                                             size_t const input_size_received,
                                                                             size_t const output_size_received,
                                                                             var const *const ptr_array_inputs_received,
                                                                             var const *const ptr_array_parameters_received,
                                                                             var *const ptr_array_outputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_output_timed_batched_index(this->batch_size * output_size_received * time_step_index_received);
    size_t tmp_output_index,
              tmp_connection_index;
    
    var const *tmp_ptr_array_inputs,
                  *tmp_ptr_array_parameters;
    var *tmp_ptr_array_outputs,
         tmp_summation;

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_parameters = ptr_array_parameters_received;
        
        tmp_ptr_array_outputs = ptr_array_outputs_received + static_cast<size_t>(tmp_example_index__int) * output_size_received + tmp_output_timed_batched_index;
        
        for(tmp_output_index = 0_UZ; tmp_output_index != output_size_received; ++tmp_output_index,
                                                                                                                  tmp_ptr_array_parameters += input_size_received)
        {
            tmp_summation = 0_r;

            for(tmp_connection_index = 0_UZ; tmp_connection_index != input_size_received; ++tmp_connection_index) { tmp_summation += tmp_ptr_array_inputs[tmp_connection_index] * tmp_ptr_array_parameters[tmp_connection_index]; }

            tmp_ptr_array_outputs[tmp_output_index] = tmp_summation;
        }
    }
}

void Model::Forward_Pass__Batch_Normalization__Inference__OpenMP(size_t const time_step_index_received,
                                                                                                                       size_t const batch_size,
                                                                                                                       size_t const input_size_received,
                                                                                                                       var const *const ptr_array_inputs_received,
                                                                                                                       var const *const ptr_array_scales_received,
                                                                                                                       var const *const ptr_array_shifts_received,
                                                                                                                       var const *const ptr_array_means_averages_received,
                                                                                                                       var const *const ptr_array_variances_averages_received,
                                                                                                                       var *const ptr_array_output_normalizes_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index,
              tmp_input_data_timed_index;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            // Normalize input, scale and shift.
            // value_normalize = scale * ( (summation - mean) / variance ) + shift
            ptr_array_output_normalizes_received[tmp_input_data_timed_index + tmp_input_index] = ptr_array_scales_received[tmp_input_index] * ( (ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index] - ptr_array_means_averages_received[tmp_input_timed_index + tmp_input_index]) / ptr_array_variances_averages_received[tmp_input_timed_index + tmp_input_index] ) + ptr_array_shifts_received[tmp_input_index];
        }
    }
}

void Model::Forward_Pass__Batch_Normalization__Training__OpenMP(size_t const time_step_index_received,
                                                                                                                     size_t const batch_size,
                                                                                                                     size_t const input_size_received,
                                                                                                                     var const *const ptr_array_inputs_received,
                                                                                                                     var const *const ptr_array_scales_received,
                                                                                                                     var const *const ptr_array_shifts_received,
                                                                                                                     var *const ptr_array_means_received,
                                                                                                                     var *const ptr_array_variances_received,
                                                                                                                     var *const ptr_array_means_averages_received,
                                                                                                                     var *const ptr_array_variances_averages_received,
                                                                                                                     var *const ptr_array_output_hats_received,
                                                                                                                     var *const ptr_array_output_normalizes_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size)),
                  tmp_units_size__int(static_cast<int>(input_size_received)),
                  tmp_number_threads__int(std::min<int>(tmp_batch_size__int, static_cast<int>(this->number_threads)));
    int tmp_thread_index__int,
        tmp_example_index__int,
        tmp_input_index__int;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index,
              tmp_input_data_timed_index,
              tmp_input_thread_timed_index;
    
    real const tmp_batch_scale(1_r / static_cast<real>(batch_size)),
                  tmp_epsilon(this->normalization_epsilon);
    var *tmp_ptr_array_mean,
        *tmp_ptr_array_variance,
        tmp_reduction_mean,
        tmp_reduction_variance,
        tmp_summation;
    
    // Summation.
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_input_thread_timed_index = static_cast<size_t>(omp_get_thread_num()) * input_size_received * this->seq_w + tmp_input_timed_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_summation = ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index];

            // mean += summation
            ptr_array_means_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_summation;
            // variance += pow(summation, 2)
            ptr_array_variances_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_summation * tmp_summation;
        }
    }
        
    // Average.
    #pragma omp for schedule(static)
    for(tmp_input_index__int = 0; tmp_input_index__int < tmp_units_size__int; ++tmp_input_index__int)
    {
        // Reduction.
        tmp_reduction_mean = 0_r;
        tmp_reduction_variance = 0_r;

        tmp_ptr_array_mean = ptr_array_means_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));
        tmp_ptr_array_variance = ptr_array_variances_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));

        // TODO: Transpose optimization.
        for(tmp_thread_index__int = 1; tmp_thread_index__int != tmp_number_threads__int; ++tmp_thread_index__int)
        {
            tmp_reduction_mean += tmp_ptr_array_mean[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
            tmp_reduction_variance += tmp_ptr_array_variance[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
        }

        *tmp_ptr_array_mean += tmp_reduction_mean;
        *tmp_ptr_array_variance += tmp_reduction_variance;
        // |END| Reduction. |END|

        // Average batch mean.
        // mean_b = sum(summation, N) / N
        *tmp_ptr_array_mean *= tmp_batch_scale;

        // Average exponentialy global mean.
        // mean += momentum * (mean_b - mean)
        ptr_array_means_averages_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)] += this->normalization_momentum_average * (*tmp_ptr_array_mean - ptr_array_means_averages_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)]); // Exponential moving average.
            
        // Average batch variance.
        // variance_b = sqrt( ((sum(pow(summation, 2), N) / N) - pow(mean_b, 2) + epsilon )
        *tmp_ptr_array_variance = op_sqrt(*tmp_ptr_array_variance * tmp_batch_scale - *tmp_ptr_array_mean * *tmp_ptr_array_mean + tmp_epsilon);
            
        // Average exponentialy global variance.
        // variance += momentum * (variance_b - variance)
        ptr_array_variances_averages_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)] += this->normalization_momentum_average * (*tmp_ptr_array_variance - ptr_array_variances_averages_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)]); // Exponential moving average.
    }

    // Activation function.
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_summation = ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index];

            // Normalize.
            // value_hat = (summation - mean_b) / variance_b * r_correction + d_correction
            ptr_array_output_hats_received[tmp_input_data_timed_index + tmp_input_index] = tmp_summation = (tmp_summation - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) / ptr_array_variances_received[tmp_input_timed_index + tmp_input_index];
                
            // Scale and shift.
            // value_normalize = scale * value_hat + shift
            ptr_array_output_normalizes_received[tmp_input_data_timed_index + tmp_input_index] = tmp_summation = ptr_array_scales_received[tmp_input_index] * tmp_summation + ptr_array_shifts_received[tmp_input_index];
        }
    }
}

void Model::Forward_Pass__Batch_Renormalization__Training__OpenMP(size_t const time_step_index_received,
                                                                                                                         size_t const batch_size,
                                                                                                                         size_t const input_size_received,
                                                                                                                         var const *const ptr_array_inputs_received,
                                                                                                                         var const *const ptr_array_scales_received,
                                                                                                                         var const *const ptr_array_shifts_received,
                                                                                                                         var *const ptr_array_means_received,
                                                                                                                         var *const ptr_array_variances_received,
                                                                                                                         var *const ptr_array_means_averages_received,
                                                                                                                         var *const ptr_array_variances_averages_received,
                                                                                                                         var *const ptr_array_r_corrections_received,
                                                                                                                         var *const ptr_array_d_corrections_received,
                                                                                                                         var *const ptr_array_output_hats_received,
                                                                                                                         var *const ptr_array_output_normalizes_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size)),
                  tmp_units_size__int(static_cast<int>(input_size_received)),
                  tmp_number_threads__int(std::min<int>(tmp_batch_size__int, static_cast<int>(this->number_threads)));
    int tmp_thread_index__int,
        tmp_example_index__int,
        tmp_input_index__int;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index,
              tmp_input_data_timed_index,
              tmp_input_thread_timed_index;
    
    real const tmp_batch_scale(1_r / static_cast<real>(batch_size)),
                  tmp_r_correction_maximum(this->batch_renormalization_r_correction_maximum),
                  tmp_d_correction_maximum(this->batch_renormalization_d_correction_maximum),
                  tmp_epsilon(this->normalization_epsilon);
    var *tmp_ptr_array_mean,
        *tmp_ptr_array_variance,
        tmp_reduction_mean,
        tmp_reduction_variance,
        tmp_summation,
        tmp_gamma,
        tmp_r_correction,
        tmp_d_correction;
    
    // Summation.
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_input_thread_timed_index = static_cast<size_t>(omp_get_thread_num()) * input_size_received * this->seq_w + tmp_input_timed_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_summation = ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index];

            // mean += summation
            ptr_array_means_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_summation;

            // variance += pow(summation, 2)
            ptr_array_variances_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_summation * tmp_summation;
        }
    }

    // Average.
    #pragma omp for schedule(static)
    for(tmp_input_index__int = 0; tmp_input_index__int < tmp_units_size__int; ++tmp_input_index__int)
    {
        // Reduction.
        tmp_reduction_mean = 0_r;
        tmp_reduction_variance = 0_r;

        tmp_ptr_array_mean = ptr_array_means_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));
        tmp_ptr_array_variance = ptr_array_variances_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));
            
        for(tmp_thread_index__int = 1; tmp_thread_index__int != tmp_number_threads__int; ++tmp_thread_index__int)
        {
            tmp_reduction_mean += tmp_ptr_array_mean[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
            tmp_reduction_variance += tmp_ptr_array_variance[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
        }

        *tmp_ptr_array_mean += tmp_reduction_mean;
        *tmp_ptr_array_variance += tmp_reduction_variance;
        // |END| Reduction. |END|

        // Average batch mean.
        // mean_b = sum(summation, N) / N
        *tmp_ptr_array_mean *= tmp_batch_scale;

        // Average exponentialy global mean.
        // mean += momentum * (mean_b - mean)
        ptr_array_means_averages_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)] += this->normalization_momentum_average * (*tmp_ptr_array_mean - ptr_array_means_averages_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)]); // Exponential moving average.
            
        // Average batch variance.
        // variance_b = sqrt( ((sum(pow(summation, 2), N) / N) - pow(mean_b, 2) + epsilon )
        *tmp_ptr_array_variance = op_sqrt(*tmp_ptr_array_variance * tmp_batch_scale - *tmp_ptr_array_mean * *tmp_ptr_array_mean + tmp_epsilon);
            
        // Average exponentialy global variance.
        // variance += momentum * (variance_b - variance)
        ptr_array_variances_averages_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)] += this->normalization_momentum_average * (*tmp_ptr_array_variance - ptr_array_variances_averages_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)]); // Exponential moving average.
            
        // r correction.
        // value = variance_b / variance
        tmp_gamma = *tmp_ptr_array_variance / ptr_array_variances_averages_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)];
        // low = 1 / r_correction_max
        tmp_r_correction = 1_r / tmp_r_correction_maximum;
        // high = r_correction_max
        // r_correction = clip(value, low, high)
        ptr_array_r_corrections_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)] = clip<var>(tmp_gamma, tmp_r_correction, tmp_r_correction_maximum);
            
        // d correction.
        // value = (mean_b - mean) / variance
        tmp_d_correction = (*tmp_ptr_array_mean - ptr_array_means_averages_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)]) / ptr_array_variances_averages_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)];
        // low = -d_correction_max
        // high = d_correction_max
        // d_correction = clip(value, low, high)
        ptr_array_d_corrections_received[tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int)] = clip<var>(tmp_d_correction, -tmp_d_correction_maximum, tmp_d_correction_maximum);
    }

    // Activation function.
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_summation = ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index];

            // Normalize.
            // value_hat = (summation - mean_b) / variance_b * r_correction + d_correction
            ptr_array_output_hats_received[tmp_input_data_timed_index + tmp_input_index] = tmp_summation = (tmp_summation - ptr_array_means_received[tmp_input_timed_index + tmp_input_index]) / ptr_array_variances_received[tmp_input_timed_index + tmp_input_index] * ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index] + ptr_array_d_corrections_received[tmp_input_timed_index + tmp_input_index];
                
            // Scale and shift.
            // value_normalize = scale * value_hat + shift
            ptr_array_output_normalizes_received[tmp_input_data_timed_index + tmp_input_index] = tmp_summation = ptr_array_scales_received[tmp_input_index] * tmp_summation + ptr_array_shifts_received[tmp_input_index];
        }
    }
}

void Model::Forward_Pass__FC_AF__OpenMP(size_t const time_step_index_received,
                                                                                   size_t const batch_size,
                                                                                   size_t const input_size_received,
                                                                                   var const *const ptr_array_inputs_received,
                                                                                   var *const ptr_array_outputs_received,
                                                                                   ACTIVATION::TYPE const *const ptr_array_type_activations_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received);
    size_t tmp_input_index;
    
    var const *tmp_ptr_array_inputs;
    var *tmp_ptr_array_outputs;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_ptr_array_outputs = ptr_array_outputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            AF_FIRE(ptr_array_type_activations_received[tmp_input_index],
                          tmp_ptr_array_inputs[tmp_input_index],
                          tmp_ptr_array_outputs[tmp_input_index]);
        }
    }
}

void Model::Forward_Pass__FC_AF__Softmax__OpenMP(size_t const time_step_index_received,
                                                                                                  size_t const batch_size,
                                                                                                  size_t const input_size_received,
                                                                                                  var const *const ptr_array_inputs_received,
                                                                                                  var *const ptr_array_outputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;

    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received);
    size_t tmp_input_index;
    
    var const *tmp_ptr_array_inputs;
    var *tmp_ptr_array_outputs,
        tmp_layer_maximum_summation,
        tmp_summation;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_summation = 0_r;

        tmp_layer_maximum_summation = -(std::numeric_limits<real>::max)();
        
        tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_ptr_array_outputs = ptr_array_outputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_layer_maximum_summation = op_max(tmp_layer_maximum_summation, tmp_ptr_array_inputs[tmp_input_index]); }
        
        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_summation += tmp_ptr_array_outputs[tmp_input_index] = op_exp(tmp_ptr_array_inputs[tmp_input_index] - tmp_layer_maximum_summation); }

        tmp_summation = 1_r / tmp_summation;
        
        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_outputs[tmp_input_index] *= tmp_summation; }
    }
}

void Model::Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(bool const *const ptr_array__mask__dropout__bernoulli_received,
                                                                                                                   size_t const time_step_index_received,
                                                                                                                   size_t const batch_size,
                                                                                                                   size_t const input_size_received,
    real const inverse_retention_probability_divided_received,
                                                                                                                   var *const ptr_array_inputs_received)
{
    bool const *tmp_ptr_timed_mask_dropout_bernoulli;

    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index;
    
    var *tmp_ptr_array_inputs;

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_timed_mask_dropout_bernoulli = ptr_array__mask__dropout__bernoulli_received + tmp_input_timed_index;
        
        tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            if(tmp_ptr_timed_mask_dropout_bernoulli[tmp_input_index]) { tmp_ptr_array_inputs[tmp_input_index] *= inverse_retention_probability_divided_received; }
            else { tmp_ptr_array_inputs[tmp_input_index] = 0_r; }
        }
    }
}

void Model::Forward_Pass__Dropout__Bernoulli__Training__OpenMP(bool const *const ptr_array__mask__dropout__bernoulli_received,
                                                                                                                  size_t const time_step_index_received,
                                                                                                                  size_t const batch_size,
                                                                                                                  size_t const input_size_received,
                                                                                                                  var *const ptr_array_inputs_received)
{
    bool const *tmp_ptr_timed_mask_dropout_bernoulli;

    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index;
    
    var *tmp_ptr_array_inputs;

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_timed_mask_dropout_bernoulli = ptr_array__mask__dropout__bernoulli_received + tmp_input_timed_index;
        
        tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index) { if(tmp_ptr_timed_mask_dropout_bernoulli[tmp_input_index] == false) { tmp_ptr_array_inputs[tmp_input_index] = 0_r; } }
    }
}

void Model::Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(size_t const time_step_index_received,
                                                                                                                    size_t const batch_size,
                                                                                                                    size_t const input_size_received,
                                                                                                                    real const retention_probability_received,
                                                                                                                    var *const ptr_array_inputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index;
    
    var *tmp_ptr_array_inputs;

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_inputs[tmp_input_index] *= retention_probability_received; }
    }
}

void Model::Forward_Pass__Dropout__Gaussian__OpenMP(size_t const time_step_index_received,
                                                             size_t const batch_size,
                                                             size_t const input_size_received, real const variance,
                                                             var *const ptr_array_inputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int,
        tmp_thread_index__int(omp_get_thread_num());
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index;
    
    var *tmp_ptr_array_inputs;

    this->ptr_array_Class_Generator_Real_Gaussian[tmp_thread_index__int].range(1_r, variance);

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();
        
        tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_inputs[tmp_input_index] *= this->ptr_array_Class_Generator_Real_Gaussian[tmp_thread_index__int](); }
    }
}

void Model::Forward_Pass__Dropout__ShakeDrop__OpenMP(size_t const time_step_index_received,
                                                              size_t const batch_size,
                                                              size_t const input_size_received,
                                                              bool *const ptr_array_mask_dopout_shakedrop_received,
    real const lower_bound,
    real const upper_bound,
    real const dropout_probability_received,
                                                              var *const ptr_array_inputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int,
        tmp_thread_index__int(omp_get_thread_num());
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index),
                       tmp_layer_timed_batched_index(this->batch_size * time_step_index_received);
    size_t tmp_input_index;
    
    var *tmp_ptr_array_inputs;

    this->ptr_array_Class_Generator_Bernoulli_ShakeDrop[tmp_thread_index__int].probability(dropout_probability_received);
    this->ptr_array_Class_Generator_Real_ShakeDrop[tmp_thread_index__int].range(lower_bound, upper_bound);
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();
        
        if((ptr_array_mask_dopout_shakedrop_received[tmp_layer_timed_batched_index + static_cast<size_t>(tmp_example_index__int)] = this->ptr_array_Class_Generator_Bernoulli_ShakeDrop[tmp_thread_index__int]()))
        {
            tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

            for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_inputs[tmp_input_index] *= this->ptr_array_Class_Generator_Real_ShakeDrop[tmp_thread_index__int](); }
        }
    }
}

void Model::Forward_Pass__Dropout__Uout__OpenMP(size_t const time_step_index_received,
                                                                                              size_t const batch_size,
                                                                                              size_t const input_size_received, real const beta_received,
                                                                                              var *const ptr_array_inputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int,
        tmp_thread_index__int;
    
    size_t const tmp_input_timed_index(input_size_received * time_step_index_received),
                       tmp_input_timed_batched_index(this->batch_size * tmp_input_timed_index);
    size_t tmp_input_index;

    var *tmp_ptr_array_inputs;

    this->ptr_array_Class_Generator_Real_Uout[omp_get_thread_num()].range(-beta_received, beta_received);
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();
        
        tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_inputs[tmp_input_index] += tmp_ptr_array_inputs[tmp_input_index] * this->ptr_array_Class_Generator_Real_Uout[tmp_thread_index__int](); }
    }
}

void Model::Forward_Pass__Max_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                            size_t const batch_size,
                                                                                            size_t const input_size_received,
                                                                                            size_t const output_size_received,
                                                                                            size_t const kernel_size_received,
                                                                                            size_t const stride_received,
                                                                                            size_t const padding_received,
                                                                                            size_t const dilation_received,
                                                                                            size_t *const ptr_array_indices_received,
                                                                                            var const *const ptr_array_inputs_received,
                                                                                            var *const ptr_array_outputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_output_timed_batched_index(this->batch_size * output_size_received * time_step_index_received),
                       tmp_input_padded_half(input_size_received + padding_received);
    size_t *tmp_ptr_array_indices,
              tmp_kernel_index,
              tmp_shift_index,
              tmp_indice,
              tmp_output_index;
    
    var const *tmp_ptr_array_inputs;
    var *tmp_ptr_array_outputs,
         tmp_max;

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_inputs = ptr_array_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_indices = ptr_array_indices_received + static_cast<size_t>(tmp_example_index__int) * output_size_received + tmp_output_timed_batched_index;
        
        tmp_ptr_array_outputs = ptr_array_outputs_received + static_cast<size_t>(tmp_example_index__int) * output_size_received + tmp_output_timed_batched_index;
        
        for(tmp_output_index = 0_UZ; tmp_output_index != output_size_received; ++tmp_output_index)
        {
            tmp_indice = 0;

            tmp_max = -(std::numeric_limits<real>::max)();
            
            for(tmp_kernel_index = 0_UZ; tmp_kernel_index != kernel_size_received; ++tmp_kernel_index)
            {
                tmp_shift_index = tmp_output_index * stride_received + tmp_kernel_index * dilation_received;

                if(tmp_shift_index < padding_received || tmp_shift_index >= tmp_input_padded_half)
                {
                    if(tmp_max < 0.0)
                    {
                        tmp_indice = tmp_shift_index;

                        tmp_max = 0.0;
                    }
                }
                else if(tmp_max < tmp_ptr_array_inputs[tmp_shift_index - padding_received])
                {
                    tmp_indice = tmp_shift_index;

                    tmp_max = tmp_ptr_array_inputs[tmp_shift_index - padding_received];
                }
            }
            
            tmp_ptr_array_indices[tmp_output_index] = tmp_indice;

            tmp_ptr_array_outputs[tmp_output_index] = tmp_max;
        }
    }
}

void Model::Forward_Pass__Zero_Padded_Identity__OpenMP(size_t const time_step_index_received,
                                                                                                        size_t const batch_size,
                                                                                                        size_t const A_unit_size_received,
                                                                                                        size_t const B_unit_size_received,
                                                                                                        size_t const padding_received,
                                                                                                        var const *const ptr_array_A_received,
                                                                                                        var const *const ptr_array_B_received,
                                                                                                        var *const ptr_array_outputs_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_A_unit_timed_index(A_unit_size_received * time_step_index_received),
                       tmp_B_unit_timed_index(B_unit_size_received * time_step_index_received);
    size_t tmp_unit_index;
    
    var const *tmp_ptr_array_A_outputs,
                  *tmp_ptr_array_B_outputs;
    var *tmp_ptr_array_outputs;

    if(padding_received == 0_UZ)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_A_outputs = ptr_array_A_received + static_cast<size_t>(tmp_example_index__int) * A_unit_size_received + this->batch_size * tmp_A_unit_timed_index;
            tmp_ptr_array_B_outputs = ptr_array_B_received + static_cast<size_t>(tmp_example_index__int) * B_unit_size_received + this->batch_size * tmp_B_unit_timed_index;
            tmp_ptr_array_outputs = ptr_array_outputs_received + static_cast<size_t>(tmp_example_index__int) * A_unit_size_received + this->batch_size * tmp_A_unit_timed_index;
            
            for(tmp_unit_index = 0_UZ; tmp_unit_index != A_unit_size_received; ++tmp_unit_index)
            {
                tmp_ptr_array_outputs[tmp_unit_index] = tmp_ptr_array_A_outputs[tmp_unit_index]
                                                                                    +
                                                                             tmp_ptr_array_B_outputs[tmp_unit_index];
            }
        }
    }
    else if(A_unit_size_received > B_unit_size_received)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_A_outputs = ptr_array_A_received + static_cast<size_t>(tmp_example_index__int) * A_unit_size_received + this->batch_size * tmp_A_unit_timed_index;
            tmp_ptr_array_B_outputs = ptr_array_B_received + static_cast<size_t>(tmp_example_index__int) * B_unit_size_received + this->batch_size * tmp_B_unit_timed_index;
            tmp_ptr_array_outputs = ptr_array_outputs_received + static_cast<size_t>(tmp_example_index__int) * A_unit_size_received + this->batch_size * tmp_A_unit_timed_index;
            
            for(tmp_unit_index = 0_UZ; tmp_unit_index != A_unit_size_received; ++tmp_unit_index) { tmp_ptr_array_outputs[tmp_unit_index] = tmp_ptr_array_A_outputs[tmp_unit_index]; }

            for(tmp_unit_index = 0_UZ; tmp_unit_index != B_unit_size_received; ++tmp_unit_index) { tmp_ptr_array_outputs[tmp_unit_index + padding_received] += tmp_ptr_array_B_outputs[tmp_unit_index]; }
        }
    }
    else // if(A_unit_size_received < B_unit_size_received)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_A_outputs = ptr_array_A_received + static_cast<size_t>(tmp_example_index__int) * A_unit_size_received + this->batch_size * tmp_A_unit_timed_index;
            tmp_ptr_array_B_outputs = ptr_array_B_received + static_cast<size_t>(tmp_example_index__int) * B_unit_size_received + this->batch_size * tmp_B_unit_timed_index;
            tmp_ptr_array_outputs = ptr_array_outputs_received + static_cast<size_t>(tmp_example_index__int) * B_unit_size_received + this->batch_size * tmp_B_unit_timed_index;
            
            for(tmp_unit_index = 0_UZ; tmp_unit_index != B_unit_size_received; ++tmp_unit_index) { tmp_ptr_array_outputs[tmp_unit_index] = tmp_ptr_array_B_outputs[tmp_unit_index]; }

            for(tmp_unit_index = 0_UZ; tmp_unit_index != A_unit_size_received; ++tmp_unit_index) { tmp_ptr_array_outputs[tmp_unit_index + padding_received] += tmp_ptr_array_A_outputs[tmp_unit_index]; }
        }
    }
}
}
