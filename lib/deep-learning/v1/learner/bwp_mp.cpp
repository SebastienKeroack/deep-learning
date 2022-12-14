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
#include "deep-learning/v1/mem/reallocate.hpp"

#include <omp.h>

namespace DL::v1 {
void Model::FF__Backward_Pass_Batch__OpenMP(size_t const batch_size)
{
    size_t tmp_layer_number_outputs;
    
    real *tmp_ptr_array_layer_gradients;
    
    Layer const *const tmp_ptr_second_layer(this->ptr_array_layers + 1);

    Layer const *tmp_ptr_next_layer_end,
                               *tmp_ptr_next_layer_it;
    Layer *tmp_ptr_gradient_layer_it(this->ptr_last_layer - 1),
                      *layer_it;

    // If the network use normalization.
    #pragma omp single
    if(this->Use__Normalization())
    {
        memset(this->ptr_array_normalized_batch_units_derivatives_means, 0,
              this->number_threads * this->total_normalized_units_allocated *
                  sizeof(real));
        memset(this->ptr_array_normalized_batch_units_derivatives_variances, 0,
                this->number_threads * this->total_normalized_units_allocated *
                    sizeof(real));
    }
    
    // Loop through each layer and do a backward propagation.
    for(; tmp_ptr_gradient_layer_it != tmp_ptr_second_layer; --tmp_ptr_gradient_layer_it)
    {
        layer_it = this->ptr_array_layers + static_cast<size_t>(tmp_ptr_gradient_layer_it->previous_connected_layers[0] - this->ptr_array_layers);
        
        // clear past error(s).
        tmp_layer_number_outputs = *layer_it->ptr_number_outputs;

        tmp_ptr_array_layer_gradients = layer_it->ptr_array_derivative_outputs;

        #pragma omp single
        memset(tmp_ptr_array_layer_gradients, 0,
                this->batch_size * tmp_layer_number_outputs * sizeof(real));
        // |END| clear past error(s). |END|
        
        // Propagate the error(s) to the layer.
        for(tmp_ptr_next_layer_it = layer_it->next_connected_layers[0],
            tmp_ptr_next_layer_end = tmp_ptr_next_layer_it + layer_it->next_connected_layers.size(); tmp_ptr_next_layer_it != tmp_ptr_next_layer_end; ++tmp_ptr_next_layer_it)
        {
            switch(tmp_ptr_next_layer_it->type_layer)
            {
                case LAYER::AVERAGE_POOLING:
                    this->Backward_Pass__Average_Pooling__OpenMP(0_UZ,
                                                                                                 batch_size,
                                                                                                 tmp_layer_number_outputs,
                                                                                                 tmp_ptr_array_layer_gradients,
                                                                                                 tmp_ptr_next_layer_it);
                        break;
                case LAYER::FULLY_CONNECTED:
                    this->Backward_Pass__FC__OpenMP(batch_size,
                                                                             tmp_layer_number_outputs,
                                                                             tmp_ptr_array_layer_gradients,
                                                                             tmp_ptr_next_layer_it);
                        break;
                case LAYER::MAX_POOLING:
                    this->Backward_Pass__Max_Pooling__OpenMP(0_UZ,
                                                                                           batch_size,
                                                                                           tmp_layer_number_outputs,
                                                                                           tmp_ptr_array_layer_gradients,
                                                                                           tmp_ptr_next_layer_it);
                        break;
                case LAYER::RESIDUAL:
                    this->Backward_Pass__Residual__OpenMP(0_UZ,
                                                                                     batch_size,
                                                                                     tmp_layer_number_outputs,
                                                                                     tmp_ptr_array_layer_gradients,
                                                                                     tmp_ptr_next_layer_it);
                        break;
                default:
                    ERR(L"Layer type (%d | %ls) is not managed in",
                        tmp_ptr_next_layer_it->type_layer,
                        LAYER_NAME[tmp_ptr_next_layer_it->type_layer].c_str());
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
                this->Backward_Pass__Gradient__FC__OpenMP(0_UZ,
                                                                                         batch_size,
                                                                                         layer_it);
                    break;
            case LAYER::RESIDUAL:
                this->Backward_Pass__Gradient__Residual__OpenMP(batch_size, layer_it);

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

void Model::FF__Backward_Pass_Batch__Pre_Training__OpenMP(size_t const batch_size)
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
               this->number_threads * this->total_normalized_units_allocated *
                   sizeof(real));
        memset(this->ptr_array_normalized_batch_units_derivatives_variances,
                    0,
               this->number_threads * this->total_normalized_units_allocated *
                   sizeof(real));
    }

    // clear past error(s).
    tmp_layer_number_outputs = *tmp_ptr_coded_layer->ptr_number_outputs;

    tmp_ptr_array_layer_gradients = tmp_ptr_coded_layer->ptr_array_derivative_outputs;

    #pragma omp single
    memset(tmp_ptr_array_layer_gradients,
                   0,
                   this->batch_size * tmp_layer_number_outputs * sizeof(real));
    // |END| clear past error(s). |END|
    
    // Propagate the error(s) to the layer.
    switch(tmp_ptr_decoded_layer->type_layer)
    {
        case LAYER::FULLY_CONNECTED:
            this->Backward_Pass__FC__OpenMP(batch_size,
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
            this->Backward_Pass__Gradient__FC__OpenMP(0_UZ,
                                                                                     batch_size,
                                                                                     tmp_ptr_coded_layer);
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

void Model::Backward_Pass__FC__OpenMP(size_t const batch_size,
                                                                                size_t const derivative_size_received,
                                                                                real *const ptr_array_derivatives_received,
                                                                                Layer const *const layer_it)
{
    if(layer_it->type_group == GROUP::RESIDUAL)
    {
        this->Backward_Pass__Residual__FC__OpenMP(0_UZ,
                                                                                 batch_size,
                                                                                 derivative_size_received,
                                                                                 ptr_array_derivatives_received,
                                                                                 layer_it);
    }
    else
    {
        this->Backward_Pass__FC__OpenMP(0_UZ,
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

void Model::Backward_Pass__Average_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                                    size_t const batch_size,
                                                                                                    size_t const derivative_size_received,
                                                                                                    real *const ptr_array_derivatives_received,
                                                                                                    Layer const *const layer_it)
{
    this->Backward_Pass__Average_Pooling__OpenMP(time_step_index_received,
                                                                                 batch_size,
                                                                                 *layer_it->ptr_number_outputs,
                                                                                 derivative_size_received,
                                                                                 layer_it->pooling_values[0],
                                                                                 layer_it->pooling_values[1],
                                                                                 layer_it->pooling_values[2],
                                                                                 layer_it->pooling_values[3],
                                                                                 layer_it->ptr_array_basic_units->ptr_array_errors,
                                                                                 ptr_array_derivatives_received);
}

void Model::Backward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                                                                size_t const batch_size,
                                                                                size_t const derivative_size_received,
                                                                                real *const ptr_array_derivatives_received,
                                                                                Layer const *const layer_it)
{
    Neuron_unit *const tmp_ptr_layer_first_neuron_unit(layer_it->ptr_array_neuron_units);
    
    this->Backward_Pass__FC__OpenMP(time_step_index_received,
                                                            batch_size,
                                                            static_cast<size_t>(layer_it->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit),
                                                            derivative_size_received,
                                                            tmp_ptr_layer_first_neuron_unit->ptr_array_errors,
                                                            this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                            ptr_array_derivatives_received);
}

void Model::Backward_Pass__Max_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                              size_t const batch_size,
                                                                                              size_t const derivative_size_received,
                                                                                              real *const ptr_array_derivatives_received,
                                                                                              Layer const *const layer_it)
{
    Basic_indice_unit *const tmp_ptr_layer_first_basic_indice_unit(layer_it->ptr_array_basic_indice_units);
    
    this->Backward_Pass__Max_Pooling__OpenMP(time_step_index_received,
                                                                           batch_size,
                                                                           *layer_it->ptr_number_outputs,
                                                                           derivative_size_received,
                                                                           layer_it->pooling_values[2],
                                                                           tmp_ptr_layer_first_basic_indice_unit->ptr_array_indices,
                                                                           tmp_ptr_layer_first_basic_indice_unit->ptr_array_errors,
                                                                           ptr_array_derivatives_received);
}

void Model::Backward_Pass__Residual__OpenMP(size_t const time_step_index_received,
                                                                                        size_t const batch_size,
                                                                                        size_t const derivative_size_received,
                                                                                        real *const ptr_array_derivatives_received,
                                                                                        Layer const *const layer_it)
{
    this->Backward_Pass__Residual__OpenMP(time_step_index_received,
                                                                     batch_size,
                                                                     *layer_it->ptr_number_outputs,
                                                                     derivative_size_received,
                                                                     layer_it->pooling_values[2],
                                                                     layer_it->ptr_array_basic_units->ptr_array_errors,
                                                                     ptr_array_derivatives_received);
}

void Model::Backward_Pass__Residual__Block__OpenMP(size_t const time_step_index_received,
                                                                                                    size_t const batch_size,
                                                                                                    size_t const derivative_size_received,
                                                                                                    real *const ptr_array_derivatives_received,
                                                                                                    Layer const *const layer_it)
{
    union Normalized_unit *const tmp_ptr_residual_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    real *tmp_ptr_array_derivatives;
    
    if(layer_it->Use__Normalization())
    {
        tmp_ptr_array_derivatives = tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_errors;

        #pragma omp single
        memset(tmp_ptr_array_derivatives + this->batch_size * derivative_size_received * time_step_index_received,
                       0, this->batch_size * derivative_size_received * sizeof(real));
    }
    else { tmp_ptr_array_derivatives = ptr_array_derivatives_received; }

    this->Backward_Pass__Residual__OpenMP(time_step_index_received,
                                                                     batch_size,
                                                                     *layer_it->ptr_number_outputs,
                                                                     derivative_size_received,
                                                                     layer_it->pooling_values[2],
                                                                     layer_it->ptr_array_basic_units->ptr_array_errors,
                                                                     tmp_ptr_array_derivatives);
    
    // Dropout, ShakeDrop.
    if(layer_it->type_dropout == LAYER_DROPOUT::SHAKEDROP)
    {
        this->Backward_Pass__Dropout__ShakeDrop__OpenMP(time_step_index_received,
                                                                                           batch_size,
                                                                                           derivative_size_received,
                                                                                           layer_it->ptr_array__mask__dropout__shakedrop,
                                                                                           0_r,
                                                                                           1_r,
                                                                                           tmp_ptr_array_derivatives);
    }
    
    // Normalization.
    if(layer_it->Use__Normalization())
    {
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                this->Backward_Pass__Batch_Normalization__OpenMP(time_step_index_received,
                                                                                            batch_size,
                                                                                            derivative_size_received,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                            layer_it->ptr_array_pre_normalization,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                            tmp_ptr_array_derivatives,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                            tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                this->Backward_Pass__Batch_Renormalization__OpenMP(time_step_index_received,
                                                                                                batch_size,
                                                                                                derivative_size_received,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                layer_it->ptr_array_pre_normalization,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_array_derivatives,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            default:
                ERR(L"Layer normalization (%d | %ls) is not managed in",
                    layer_it->type_normalization,
                    LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                return;
        }
        
        // Store the new derivative inputs (normalized derivative).
        tmp_ptr_array_derivatives = tmp_ptr_residual_first_normalized_unit->normalized_batch_units.ptr_array_errors;
        
        //     --------------> FC --> ... --> {FC} --> [ResNet-BN]
        //    /                                                        /
        // FC --> ResNet -----------------------------------------------> ...
        #pragma omp single
        memcpy(ptr_array_derivatives_received + this->batch_size * derivative_size_received * time_step_index_received,
                       tmp_ptr_array_derivatives + this->batch_size * derivative_size_received * time_step_index_received,
                       this->batch_size * derivative_size_received * sizeof(real));
    }
}

void Model::Backward_Pass__Residual__FC__OpenMP(size_t const time_step_index_received,
                                                                                                size_t const batch_size,
                                                                                                size_t const derivative_size_received,
                                                                                                real *const ptr_array_derivatives_received,
                                                                                                Layer const *const layer_it)
{
    bool const tmp_is_input_layer(static_cast<size_t>(layer_it->ptr_last_AF_unit - layer_it->ptr_array_AF_units) + static_cast<size_t>(layer_it->ptr_last_AF_Ind_recurrent_unit - layer_it->ptr_array_AF_Ind_recurrent_units) == 0_UZ);

    if(layer_it->Use__Normalization())
    {
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
            case LAYER_NORM::BATCH_RENORMALIZATION:
                //     --------------> {FC} --> [FC] ...
                //    /
                // FC --> ResNet ---> ...
                if(tmp_is_input_layer == false)
                {
                    #pragma omp single
                    memcpy(ptr_array_derivatives_received + this->batch_size * derivative_size_received * time_step_index_received,
                                  layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_array_errors + this->batch_size * derivative_size_received * time_step_index_received,
                         this->batch_size * derivative_size_received *
                             sizeof(real));
                }
                //     --------------> [FC] --> FC ...
                //    /
                // {FC} --> ResNet ---> ...
                else
                {
                    this->Backward_Pass__Identity__OpenMP(time_step_index_received,
                                                                                   batch_size,
                                                                                   derivative_size_received,
                                                                                   layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_array_errors,
                                                                                   ptr_array_derivatives_received);
                }
                    break;
            default:
                ERR(L"Layer normalization (%d | %ls) is not managed in",
                    layer_it->type_normalization,
                    LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                return;
        }
    }
    //     --------------> {FC} --> [FC] ...
    //    /
    // FC --> ResNet ---> ...
    else if(tmp_is_input_layer == false)
    {
        if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            #pragma omp single
            memcpy(ptr_array_derivatives_received + this->batch_size * derivative_size_received * time_step_index_received,
                           layer_it->ptr_array_AF_Ind_recurrent_units->ptr_array_dAFs + this->batch_size * derivative_size_received * time_step_index_received,
                 this->batch_size * derivative_size_received * sizeof(real));
        }
        else
        {
            #pragma omp single
            memcpy(ptr_array_derivatives_received + this->batch_size * derivative_size_received * time_step_index_received,
                           layer_it->ptr_array_AF_units->ptr_array_errors + this->batch_size * derivative_size_received * time_step_index_received,
                 this->batch_size * derivative_size_received * sizeof(real));
        }
    }
    //     --------------> [FC] --> FC ...
    //    /
    // {FC} --> ResNet ---> ...
    else
    {
        Neuron_unit *const tmp_ptr_layer_first_neuron_unit(layer_it->ptr_array_neuron_units);
        
        this->Backward_Pass__FC__OpenMP(time_step_index_received,
                                                                 batch_size,
                                                                 static_cast<size_t>(layer_it->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit),
                                                                 derivative_size_received,
                                                                 tmp_ptr_layer_first_neuron_unit->ptr_array_errors,
                                                                 this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                                 ptr_array_derivatives_received);
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Backward_Pass__Gradient__FC__OpenMP(size_t const time_step_index_received,
                                                                                                size_t const batch_size,
                                                                                                Layer const *const layer_it)
{
    Neuron_unit *const tmp_ptr_layer_first_neuron_unit(layer_it->ptr_array_neuron_units);
    
    AF_unit *const tmp_ptr_layer_first_AF_unit(layer_it->ptr_array_AF_units);
    AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(layer_it->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_output_size(static_cast<size_t>(layer_it->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));
    
    real const *tmp_ptr_array_derivative_inputs(layer_it->ptr_array_derivative_outputs);

    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    // k-Sparse.
    // ...

    // Normalization after activation.
    if(layer_it->Use__Normalization()
      &&
      layer_it->use_layer_normalization_before_activation == false)
    {
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                this->Backward_Pass__Batch_Normalization__OpenMP(time_step_index_received,
                                                                                            batch_size,
                                                                                            tmp_output_size,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                            layer_it->ptr_array_pre_normalization,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                            tmp_ptr_array_derivative_inputs,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                this->Backward_Pass__Batch_Renormalization__OpenMP(time_step_index_received,
                                                                                                batch_size,
                                                                                                tmp_output_size,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                layer_it->ptr_array_pre_normalization,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_array_derivative_inputs,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            default:
                ERR(L"Layer normalization (%d | %ls) is not managed in",
                    layer_it->type_normalization,
                    LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                return;
        }
        
        // Store the new derivative inputs (normalized derivative).
        tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors;
    }
    
    if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
    {
        // Recurrent activation function.
        this->Backward_Pass__FC__DF_Ind_RNN__OpenMP(time_step_index_received,
                                                                                     batch_size,
                                                                                     tmp_output_size,
                                                                                     this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                                     tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function,
                                                                                     layer_it->ptr_array_pre_activation_functions,
                                                                                     tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                                     tmp_ptr_array_derivative_inputs,
                                                                                     tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_dAFs,
                                                                                     tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_errors);

        // Store the new derivative inputs (recurrent activation function derivative).
        tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_dAFs;
    }
    else
    {
        // Activation function.
        this->Backward_Pass__FC__DF__OpenMP(time_step_index_received,
                                                                       batch_size,
                                                                       tmp_output_size,
                                                                       tmp_ptr_layer_first_AF_unit->ptr_type_activation_function,
                                                                       layer_it->ptr_array_pre_activation_functions,
                                                                       tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                       tmp_ptr_array_derivative_inputs,
                                                                       tmp_ptr_layer_first_AF_unit->ptr_array_errors);

        // Store the new derivative inputs (activation function derivative).
        tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_errors;
    }

    // Normalization before activation.
    if(layer_it->Use__Normalization()
      &&
      layer_it->use_layer_normalization_before_activation)
    {
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                this->Backward_Pass__Batch_Normalization__OpenMP(time_step_index_received,
                                                                                                batch_size,
                                                                                                tmp_output_size,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                layer_it->ptr_array_pre_normalization,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_array_derivative_inputs,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                this->Backward_Pass__Batch_Renormalization__OpenMP(time_step_index_received,
                                                                                                    batch_size,
                                                                                                    tmp_output_size,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                    layer_it->ptr_array_pre_normalization,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                    tmp_ptr_array_derivative_inputs,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                    tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            default:
                ERR(L"Layer normalization (%d | %ls) is not managed in",
                    layer_it->type_normalization,
                    LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                return;
        }
        
        // Store the new derivative inputs (normalized derivative).
        tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors;
    }
    
    // copy derivative to derivative neurons.
    #pragma omp single
    memcpy(tmp_ptr_layer_first_neuron_unit->ptr_array_errors + this->batch_size * tmp_output_size * time_step_index_received,
                  tmp_ptr_array_derivative_inputs + this->batch_size * tmp_output_size * time_step_index_received,
         this->batch_size * tmp_output_size * sizeof(real));
    // |END| copy derivative to derivative neurons. |END|
}

void Model::Backward_Pass__Gradient__Residual__OpenMP(size_t const batch_size, Layer const *const layer)
{
  Layer const *const tmp_ptr_layer_end(layer + 1);
  Layer *layer_it(this->ptr_array_layers +
                  static_cast<size_t>(layer - this->ptr_array_layers) +
                  layer->block_depth);
    
    // Remaining layer(s).
    for(; layer_it != tmp_ptr_layer_end; --layer_it)
    {
        this->Backward_Pass__Gradient__Residual__Layer__OpenMP(false,
                                                                                                    batch_size,
                                                                                                    layer_it);
    }
    // |END| Remaining layer(s). |END|
    
    // First block layer.
    this->Backward_Pass__Gradient__Residual__Layer__OpenMP(true,
                                                                                                batch_size,
                                                                                                layer_it);
    // |END| First block layer. |END|
}

void Model::Backward_Pass__Gradient__Residual__Layer__OpenMP(bool const is_block_input_layer_received,
                                                                                                                   size_t const batch_size,
                                                                                                                   Layer *&layer_it)
{
    size_t const tmp_layer_number_outputs(*layer_it->ptr_number_outputs);
    
    real *const tmp_ptr_array_layer_gradients(layer_it->ptr_array_derivative_outputs);
    
    Layer const *const tmp_ptr_next_layer_it(layer_it->next_connected_layers[0]);
    
    #pragma omp single
    memset(tmp_ptr_array_layer_gradients,
                  0,
            this->batch_size * tmp_layer_number_outputs * sizeof(real));
    
    switch(tmp_ptr_next_layer_it->type_layer)
    {
        case LAYER::AVERAGE_POOLING:
            this->Backward_Pass__Average_Pooling__OpenMP(0_UZ,
                                                                                         batch_size,
                                                                                         tmp_layer_number_outputs,
                                                                                         tmp_ptr_array_layer_gradients,
                                                                                         tmp_ptr_next_layer_it);
                break;
        case LAYER::FULLY_CONNECTED:
            this->Backward_Pass__Residual__FC__OpenMP(0_UZ,
                                                                                     batch_size,
                                                                                     tmp_layer_number_outputs,
                                                                                     tmp_ptr_array_layer_gradients,
                                                                                     tmp_ptr_next_layer_it);
                break;
        case LAYER::MAX_POOLING:
            this->Backward_Pass__Max_Pooling__OpenMP(0_UZ,
                                                                                   batch_size,
                                                                                   tmp_layer_number_outputs,
                                                                                   tmp_ptr_array_layer_gradients,
                                                                                   tmp_ptr_next_layer_it);
                break;
        case LAYER::RESIDUAL:
            this->Backward_Pass__Residual__Block__OpenMP(0_UZ,
                                                                                         batch_size,
                                                                                         tmp_layer_number_outputs,
                                                                                         tmp_ptr_array_layer_gradients,
                                                                                         tmp_ptr_next_layer_it);
                break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                tmp_ptr_next_layer_it->type_layer,
                LAYER_NAME[tmp_ptr_next_layer_it->type_layer].c_str());
            return;
    }
    // |END| Propagate the error(s) to the layer. |END|

    // Compute the gradients.
    switch(layer_it->type_layer)
    {
        case LAYER::AVERAGE_POOLING:
        case LAYER::MAX_POOLING: break;
        case LAYER::FULLY_CONNECTED:
            this->Backward_Pass__Gradient__Residual__FC__OpenMP(is_block_input_layer_received,
                                                                                                    0_UZ,
                                                                                                    batch_size,
                                                                                                    layer_it);
                break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
            return;
    }
    // |END| Compute the gradients. |END|
}

void Model::Backward_Pass__Gradient__Residual__FC__OpenMP(bool const is_block_input_layer_received,
                                                                                                                size_t const time_step_index_received,
                                                                                                                size_t const batch_size,
                                                                                                                Layer const *const layer_it)
{
    Neuron_unit *const tmp_ptr_layer_first_neuron_unit(layer_it->ptr_array_neuron_units);
    
    AF_unit *const tmp_ptr_layer_first_AF_unit(layer_it->ptr_array_AF_units);
    AF_Ind_recurrent_unit *const tmp_ptr_layer_first_AF_Ind_recurrent_unit(layer_it->ptr_array_AF_Ind_recurrent_units);
    
    size_t const tmp_input_size(*tmp_ptr_layer_first_neuron_unit->ptr_number_connections),
                       tmp_output_size(static_cast<size_t>(layer_it->ptr_last_neuron_unit - tmp_ptr_layer_first_neuron_unit));
    
    union Normalized_unit *const tmp_ptr_layer_first_normalized_unit(layer_it->ptr_array_normalized_units);
    
    real *tmp_ptr_array_derivative_inputs;
    
    if(is_block_input_layer_received == false)
    {
        if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT) { tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_errors; }
        else { tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_errors; }
    }
    else if(layer_it->Use__Normalization()) { tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors; }
    else { return; }

    #pragma omp single
    memset(tmp_ptr_array_derivative_inputs + this->batch_size * tmp_input_size * time_step_index_received,
                  0, this->batch_size * tmp_input_size * sizeof(real));
    
    this->Backward_Pass__FC__OpenMP(time_step_index_received,
                                                             batch_size,
                                                             tmp_output_size,
                                                             tmp_input_size,
                                                             tmp_ptr_layer_first_neuron_unit->ptr_array_errors,
                                                             this->ptr_array_parameters + *tmp_ptr_layer_first_neuron_unit->ptr_first_connection_index,
                                                             tmp_ptr_array_derivative_inputs);

    if(is_block_input_layer_received == false)
    {
        // k-Sparse.
        // ...
        
        if(layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT)
        {
            // Recurrent activation function.
            this->Backward_Pass__FC__DF_Ind_RNN__OpenMP(time_step_index_received,
                                                                                           batch_size,
                                                                                           tmp_output_size,
                                                                                           this->ptr_array_parameters + *tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_recurrent_connection_index,
                                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_type_activation_function,
                                                                                           layer_it->ptr_array_pre_activation_functions,
                                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_AFs,
                                                                                           tmp_ptr_array_derivative_inputs,
                                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_dAFs,
                                                                                           tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_errors);

            // Store the new derivative inputs (recurrent activation function derivative).
            tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_Ind_recurrent_unit->ptr_array_dAFs;
        }
        else
        {
            // Activation function.
            this->Backward_Pass__FC__DF__OpenMP(time_step_index_received,
                                                                           batch_size,
                                                                           tmp_input_size,
                                                                           tmp_ptr_layer_first_AF_unit->ptr_type_activation_function,
                                                                           layer_it->ptr_array_pre_activation_functions,
                                                                           tmp_ptr_layer_first_AF_unit->ptr_array_values,
                                                                           tmp_ptr_array_derivative_inputs,
                                                                           tmp_ptr_layer_first_AF_unit->ptr_array_errors);

            // Store the new derivative inputs (activation function derivative).
            tmp_ptr_array_derivative_inputs = tmp_ptr_layer_first_AF_unit->ptr_array_errors;
        }
    }

    // Normalization.
    if(layer_it->Use__Normalization())
    {
        switch(layer_it->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
                this->Backward_Pass__Batch_Normalization__OpenMP(time_step_index_received,
                                                                                            batch_size,
                                                                                            tmp_input_size,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                            layer_it->ptr_array_pre_normalization,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                            tmp_ptr_array_derivative_inputs,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                            tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            case LAYER_NORM::BATCH_RENORMALIZATION:
                this->Backward_Pass__Batch_Renormalization__OpenMP(time_step_index_received,
                                                                                                batch_size,
                                                                                                tmp_input_size,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_scale,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_r_correction,
                                                                                                layer_it->ptr_array_pre_normalization,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_values_hats,
                                                                                                tmp_ptr_array_derivative_inputs,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_scales,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_shifts,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_means,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_derivatives_variances,
                                                                                                tmp_ptr_layer_first_normalized_unit->normalized_batch_units.ptr_array_errors);
                    break;
            default:
                ERR(L"Layer normalization (%d | %ls) is not managed in",
                    layer_it->type_normalization,
                    LAYER_NORM_NAME[layer_it->type_normalization].c_str());
                return;
        }
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Backward_Pass__Average_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                                    size_t const batch_size,
                                                                                                    size_t const input_size_received,
                                                                                                    size_t const derivative_size_received,
                                                                                                    size_t const kernel_size_received,
                                                                                                    size_t const stride_received,
                                                                                                    size_t const padding_received,
                                                                                                    size_t const dilation_received,
                                                                                                    real const *const ptr_array_derivative_inputs_received,
                                                                                                    real *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received),
                       tmp_derivative_padded_half(derivative_size_received + padding_received);
    size_t tmp_kernel_index,
              tmp_index,
              tmp_input_index;
    
    real const *tmp_ptr_array_derivative_inputs,
                  tmp_scale(1_r / static_cast<real>(kernel_size_received));
    real *tmp_ptr_array_derivatives,
         tmp_error;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
        
        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = tmp_ptr_array_derivative_inputs[tmp_input_index] * tmp_scale;
            
            for(tmp_kernel_index = 0_UZ; tmp_kernel_index != kernel_size_received; ++tmp_kernel_index)
            {
                tmp_index = tmp_input_index * stride_received + tmp_kernel_index * dilation_received;

                if(tmp_index < padding_received || tmp_index >= tmp_derivative_padded_half) { continue; }

                tmp_ptr_array_derivatives[tmp_index - padding_received] += tmp_error;
            }
        }
    }
}

void Model::Backward_Pass__Dropout__ShakeDrop__OpenMP(size_t const time_step_index_received,
                                                                                                          size_t const batch_size,
                                                                                                          size_t const derivative_size_received,
                                                                                                          bool const *const ptr_array_mask_dopout_shakedrop_received,
                                                                                                          real const lower_bound,
                                                                                                          real const upper_bound,
                                                                                                          real *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int,
        tmp_thread_index__int(omp_get_thread_num());
    
    size_t const tmp_derivative_timed_index(derivative_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * tmp_derivative_timed_index),
                       tmp_layer_timed_batched_index(this->batch_size * time_step_index_received);
    size_t tmp_derivative_index;
    
    real *tmp_ptr_array_derivatives;

    this->ptr_array_Class_Generator_Real_ShakeDrop[tmp_thread_index__int].range(lower_bound, upper_bound);
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();
        
        if(ptr_array_mask_dopout_shakedrop_received[tmp_layer_timed_batched_index + static_cast<size_t>(tmp_example_index__int)])
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;

            for(tmp_derivative_index = 0_UZ; tmp_derivative_index != derivative_size_received; ++tmp_derivative_index) { tmp_ptr_array_derivatives[tmp_derivative_index] *= this->ptr_array_Class_Generator_Real_ShakeDrop[tmp_thread_index__int](); }
        }
    }
}

void Model::Backward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                                                                size_t const batch_size,
                                                                                size_t const input_size_received,
                                                                                size_t const derivative_size_received,
                                                                                real const *const ptr_array_derivative_inputs_received,
                                                                                var const *const ptr_array_parameters_received,
                                                                                real *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_input_index,
              tmp_connection_index;
    
    real const *tmp_ptr_array_derivative_inputs;
    real *tmp_ptr_array_derivatives,
         tmp_error;

    var const *tmp_ptr_array_parameters;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_parameters = ptr_array_parameters_received;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index,
                                                                                                            tmp_ptr_array_parameters += derivative_size_received)
        {
            tmp_error = tmp_ptr_array_derivative_inputs[tmp_input_index];
            
            for(tmp_connection_index = 0_UZ; tmp_connection_index != derivative_size_received; ++tmp_connection_index) { tmp_ptr_array_derivatives[tmp_connection_index] += tmp_error * cast(tmp_ptr_array_parameters[tmp_connection_index]); }
        }
    }
}

void Model::Backward_Pass__Identity__OpenMP(size_t const time_step_index_received,
                                                                                      size_t const batch_size,
                                                                                      size_t const input_size_received,
                                                                                      real const *const ptr_array_derivative_inputs_received,
                                                                                      real *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received);
    size_t tmp_input_index;
    
    real const *tmp_ptr_array_derivative_inputs;
    real *tmp_ptr_array_derivatives;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index) { tmp_ptr_array_derivatives[tmp_input_index] += tmp_ptr_array_derivative_inputs[tmp_input_index]; }
    }
}

void Model::Backward_Pass__Max_Pooling__OpenMP(size_t const time_step_index_received,
                                                                                              size_t const batch_size,
                                                                                              size_t const input_size_received,
                                                                                              size_t const derivative_size_received,
                                                                                              size_t const padding_received,
                                                                                              size_t const *const ptr_array_indices_received,
                                                                                              real const *const ptr_array_derivative_inputs_received,
                                                                                              real *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const *tmp_ptr_array_indices,
                       tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received),
                       tmp_derivative_padded_half(derivative_size_received + padding_received);
    size_t tmp_indice,
              tmp_input_index;
    
    real const *tmp_ptr_array_derivative_inputs;
    real *tmp_ptr_array_derivatives,
         tmp_error;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_indices = ptr_array_indices_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        
        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
        
        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_indice = tmp_ptr_array_indices[tmp_input_index];
            
            if(tmp_indice < padding_received || tmp_indice >= tmp_derivative_padded_half) { continue; }

            tmp_error = tmp_ptr_array_derivative_inputs[tmp_input_index];
            
            tmp_ptr_array_derivatives[tmp_indice - padding_received] += tmp_error;
        }
    }
}

void Model::Backward_Pass__Residual__OpenMP(size_t const time_step_index_received,
                                                                                        size_t const batch_size,
                                                                                        size_t const input_size_received,
                                                                                        size_t const derivative_size_received,
                                                                                        size_t const padding_received,
                                                                                        real const *const ptr_array_derivative_inputs_received,
                                                                                        real *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_derivative_timed_batched_index(this->batch_size * derivative_size_received * time_step_index_received);
    size_t tmp_input_index;
    
    real const *tmp_ptr_array_derivative_inputs;
    real *tmp_ptr_array_derivatives;
    
    if(input_size_received == derivative_size_received)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

            for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_derivatives[tmp_input_index] += tmp_ptr_array_derivative_inputs[tmp_input_index];
            }
        }
    }
    else if(input_size_received > derivative_size_received)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            
            for(tmp_input_index = 0_UZ; tmp_input_index != derivative_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_derivatives[tmp_input_index] += tmp_ptr_array_derivative_inputs[tmp_input_index + padding_received];
            }
        }
    }
    else
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * derivative_size_received + tmp_derivative_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            
            for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_derivatives[tmp_input_index + padding_received] += tmp_ptr_array_derivative_inputs[tmp_input_index];
            }
        }
    }
}

// ======================================

// ======================================

// ======================================

// ======================================

void Model::Backward_Pass__FC__DF__OpenMP(size_t const time_step_index_received,
                                                                                        size_t const batch_size,
                                                                                        size_t const input_size_received,
                                                                                        ACTIVATION::TYPE const *const ptr_array_type_activations_functions_received,
                                                                                        var const *const ptr_array_pre_AFs_received,
                                                                                        var const *const ptr_array_AFs_received,
                                                                                        real const *const ptr_array_derivative_inputs_received,
                                                                                        real *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received);
    size_t tmp_input_index;
    
    real const *tmp_ptr_array_derivative_inputs;
    real *tmp_ptr_array_derivatives;

    var const *tmp_ptr_array_pre_AFs, *tmp_ptr_array_AFs;

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_ptr_array_pre_AFs = ptr_array_pre_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_ptr_array_AFs = ptr_array_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_ptr_array_derivatives[tmp_input_index] = tmp_ptr_array_derivative_inputs[tmp_input_index] * this->activation_fn_derivative(ptr_array_type_activations_functions_received[tmp_input_index],
                                                                                                                                                                                                         cast(tmp_ptr_array_pre_AFs[tmp_input_index]),
                                                                                                                                                                                                         cast(tmp_ptr_array_AFs[tmp_input_index]));
        }
    }
}

void Model::Backward_Pass__FC__DF_Ind_RNN__OpenMP(size_t const time_step_index_received,
                                                                                                      size_t const batch_size,
                                                                                                      size_t const input_size_received,
                                                                                                      var const *const ptr_array_parameters_received,
                                                                                                      ACTIVATION::TYPE const *const ptr_array_type_activations_functions_received,
                                                                                                      var const *const ptr_array_pre_AFs_received,
                                                                                                      var const *const ptr_array_AFs_received,
                                                                                                      real const *const ptr_array_derivative_inputs_received,
                                                                                                      real *const ptr_array_dAFs_received,
                                                                                                      real *const ptr_array_derivatives_received)
{
    int const tmp_batch_size__int(static_cast<int>(batch_size));
    int tmp_example_index__int;
    
    size_t const tmp_input_timed_batched_index(this->batch_size * input_size_received * time_step_index_received),
                       tmp_input_next_timed_batched_index(this->batch_size * input_size_received * (time_step_index_received + 1_UZ));
    size_t tmp_input_index;
    
    var const *tmp_ptr_array_pre_AFs, *tmp_ptr_array_AFs;

    real const *tmp_ptr_array_derivative_inputs, *tmp_ptr_array_next_timed_dAFs;
    real *tmp_ptr_array_dAFs,
         *tmp_ptr_array_derivatives;
    
    if(time_step_index_received + 1_UZ != this->seq_w)
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_pre_AFs = ptr_array_pre_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_AFs = ptr_array_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

            tmp_ptr_array_dAFs = ptr_array_dAFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_next_timed_dAFs = ptr_array_dAFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_next_timed_batched_index;
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            
            for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_dAFs[tmp_input_index] = this->activation_fn_derivative(ptr_array_type_activations_functions_received[tmp_input_index],
                                                                                                                           cast(tmp_ptr_array_pre_AFs[tmp_input_index]),
                                                                                                                           cast(tmp_ptr_array_AFs[tmp_input_index]));
                
                tmp_ptr_array_derivatives[tmp_input_index] = tmp_ptr_array_derivative_inputs[tmp_input_index] * tmp_ptr_array_dAFs[tmp_input_index];
                
                tmp_ptr_array_dAFs[tmp_input_index] = tmp_ptr_array_derivatives[tmp_input_index]
                                                                                                        +
                                                                           cast(ptr_array_parameters_received[tmp_input_index])
                                                                                                        *
                                                                           tmp_ptr_array_dAFs[tmp_input_index]
                                                                                                        *
                                                                           tmp_ptr_array_next_timed_dAFs[tmp_input_index];
            }
        }
    }
    else
    {
        #pragma omp for schedule(static)
        for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
        {
            tmp_ptr_array_pre_AFs = ptr_array_pre_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_AFs = ptr_array_AFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_derivative_inputs = ptr_array_derivative_inputs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

            tmp_ptr_array_dAFs = ptr_array_dAFs_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            tmp_ptr_array_derivatives = ptr_array_derivatives_received + static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
            
            for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
            {
                tmp_ptr_array_dAFs[tmp_input_index] = tmp_ptr_array_derivatives[tmp_input_index] = tmp_ptr_array_derivative_inputs[tmp_input_index] * this->activation_fn_derivative(ptr_array_type_activations_functions_received[tmp_input_index],
                                                                                                                                                                                                                                                                        cast(tmp_ptr_array_pre_AFs[tmp_input_index]),
                                                                                                                                                                                                                                                                        cast(tmp_ptr_array_AFs[tmp_input_index]));
            }
        }
    }
}

void Model::Backward_Pass__Batch_Normalization__OpenMP(size_t const time_step_index_received,
                                                                                                        size_t const batch_size,
                                                                                                        size_t const input_size_received,
                                                                                                        var const *const ptr_array_means_received,
                                                                                                        var const *const ptr_array_variances_received,
                                                                                                        var const *const ptr_array_scales_received,
                                                                                                        var const *const ptr_array_inputs_received,
                                                                                                        var const *const ptr_array_inputs_hats_received,
                                                                                                        real const *const ptr_array_derivative_inputs_received,
                                                                                                        real *const ptr_array_derivatives_scales_received,
                                                                                                        real *const ptr_array_derivatives_shifts_received,
                                                                                                        real *const ptr_array_derivatives_means_received,
                                                                                                        real *const ptr_array_derivatives_variances_received,
                                                                                                        real *const ptr_array_derivatives_received)
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
              tmp_input_thread_timed_index,
              tmp_input_thread_parameter_index;
    
    real const tmp_batch_scale(1_r / static_cast<real>(batch_size));
    real *tmp_ptr_array_derivative_mean,
         *tmp_ptr_array_derivative_variance,
         tmp_reduction_derivative_mean,
         tmp_reduction_derivative_variance,
        tmp_error, tmp_variance_b;

    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();

        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_input_thread_timed_index = static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w + tmp_input_timed_index;
        tmp_input_thread_parameter_index = static_cast<size_t>(tmp_thread_index__int) * this->total_parameters_allocated;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = cast(ptr_array_variances_received[tmp_input_timed_index + tmp_input_index]);

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error * cast(ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index]);
                
            // Derivative shift.
            // dShift += dY
            ptr_array_derivatives_shifts_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error;

            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= cast(ptr_array_scales_received[tmp_input_index]);
                
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * ( -1_r / tmp_variance_b );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * (cast(ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index]) - cast(ptr_array_means_received[tmp_input_timed_index + tmp_input_index])) * ( -1_r / (tmp_variance_b * tmp_variance_b) );
                
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
        
    // Reduction.
    #pragma omp for schedule(static)
    for(tmp_input_index__int = 0; tmp_input_index__int < tmp_units_size__int; ++tmp_input_index__int)
    {
        tmp_reduction_derivative_mean = 0_r;
        tmp_reduction_derivative_variance = 0_r;
            
        tmp_ptr_array_derivative_mean = ptr_array_derivatives_means_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));
        tmp_ptr_array_derivative_variance = ptr_array_derivatives_variances_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));

        // TODO: Transpose optimization.
        for(tmp_thread_index__int = 1; tmp_thread_index__int != tmp_number_threads__int; ++tmp_thread_index__int)
        {
            tmp_reduction_derivative_mean += tmp_ptr_array_derivative_mean[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
            tmp_reduction_derivative_variance += tmp_ptr_array_derivative_variance[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
        }

        *tmp_ptr_array_derivative_mean += tmp_reduction_derivative_mean;
        *tmp_ptr_array_derivative_variance += tmp_reduction_derivative_variance;
    }
        
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = cast(ptr_array_variances_received[tmp_input_timed_index + tmp_input_index]);

            // First
            // dX_h *= r_correction / variance_b
            tmp_error *= 1_r / tmp_variance_b;
                
            // Middle
            // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
            tmp_error += ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] * ( (cast(ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index]) - cast(ptr_array_means_received[tmp_input_timed_index + tmp_input_index])) / (static_cast<real>(batch_size) * tmp_variance_b) );

            // Last
            // dX_h += dMean_b * 1 / N
            // dX_h += dMean_b / N
            tmp_error += ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] * tmp_batch_scale;

            // dX = dX_h
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
}

void Model::Backward_Pass__Batch_Normalization__OpenMP(size_t const time_step_index_received,
                                                                                                        size_t const batch_size,
                                                                                                        size_t const input_size_received,
                                                                                                        var const *const ptr_array_means_received,
                                                                                                        var const *const ptr_array_variances_received,
                                                                                                        var const *const ptr_array_scales_received,
                                                                                                        var const *const ptr_array_inputs_received,
                                                                                                        var const *const ptr_array_inputs_hats_received,
                                                                                                        real const *const ptr_array_derivative_inputs_received,
                                                                                                        real *const ptr_array_derivatives_scales_received,
                                                                                                        real *const ptr_array_derivatives_means_received,
                                                                                                        real *const ptr_array_derivatives_variances_received,
                                                                                                        real *const ptr_array_derivatives_received)
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
              tmp_input_thread_timed_index,
              tmp_input_thread_parameter_index;
    
    real const tmp_batch_scale(1_r / static_cast<real>(batch_size));
    real *tmp_ptr_array_derivative_mean,
         *tmp_ptr_array_derivative_variance,
         tmp_reduction_derivative_mean,
         tmp_reduction_derivative_variance,
         tmp_error,
         tmp_variance_b;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();

        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_input_thread_timed_index = static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w + tmp_input_timed_index;
        tmp_input_thread_parameter_index = static_cast<size_t>(tmp_thread_index__int) * this->total_parameters_allocated;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = cast(ptr_array_variances_received[tmp_input_timed_index + tmp_input_index]);

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error * cast(ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index]);
            
            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= cast(ptr_array_scales_received[tmp_input_index]);
                
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * ( -1_r / tmp_variance_b );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * (cast(ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index]) - cast(ptr_array_means_received[tmp_input_timed_index + tmp_input_index])) * ( -1_r / (tmp_variance_b * tmp_variance_b) );
                
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
        
    // Reduction.
    #pragma omp for schedule(static)
    for(tmp_input_index__int = 0; tmp_input_index__int < tmp_units_size__int; ++tmp_input_index__int)
    {
        tmp_reduction_derivative_mean = 0_r;
        tmp_reduction_derivative_variance = 0_r;
            
        tmp_ptr_array_derivative_mean = ptr_array_derivatives_means_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));
        tmp_ptr_array_derivative_variance = ptr_array_derivatives_variances_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));

        for(tmp_thread_index__int = 1; tmp_thread_index__int != tmp_number_threads__int; ++tmp_thread_index__int)
        {
            tmp_reduction_derivative_mean += tmp_ptr_array_derivative_mean[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
            tmp_reduction_derivative_variance += tmp_ptr_array_derivative_variance[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
        }

        *tmp_ptr_array_derivative_mean += tmp_reduction_derivative_mean;
        *tmp_ptr_array_derivative_variance += tmp_reduction_derivative_variance;
    }
        
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = cast(ptr_array_variances_received[tmp_input_timed_index + tmp_input_index]);

            // First
            // dX_h *= r_correction / variance_b
            tmp_error *= 1_r / tmp_variance_b;
                
            // Middle
            // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
            tmp_error += ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] * ( (cast(ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index]) - cast(ptr_array_means_received[tmp_input_timed_index + tmp_input_index])) / (static_cast<real>(batch_size) * tmp_variance_b) );

            // Last
            // dX_h += dMean_b * 1 / N
            // dX_h += dMean_b / N
            tmp_error += ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] * tmp_batch_scale;

            // dX = dX_h
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
}

void Model::Backward_Pass__Batch_Renormalization__OpenMP(size_t const time_step_index_received,
                                                                                                             size_t const batch_size,
                                                                                                             size_t const input_size_received,
                                                                                                             var const *const ptr_array_means_received,
                                                                                                             var const *const ptr_array_variances_received,
                                                                                                             var const *const ptr_array_scales_received,
                                                                                                             var const *const ptr_array_r_corrections_received,
                                                                                                             var const *const ptr_array_inputs_received,
                                                                                                             var const *const ptr_array_inputs_hats_received,
                                                                                                             real const *const ptr_array_derivative_inputs_received,
                                                                                                             real *const ptr_array_derivatives_scales_received,
                                                                                                             real *const ptr_array_derivatives_shifts_received,
                                                                                                             real *const ptr_array_derivatives_means_received,
                                                                                                             real *const ptr_array_derivatives_variances_received,
                                                                                                             real *const ptr_array_derivatives_received)
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
              tmp_input_thread_timed_index,
              tmp_input_thread_parameter_index;
    
    real const tmp_batch_scale(1_r / static_cast<real>(batch_size));
    real *tmp_ptr_array_derivative_mean,
         *tmp_ptr_array_derivative_variance,
         tmp_reduction_derivative_mean,
         tmp_reduction_derivative_variance,
         tmp_error,
         tmp_variance_b,
         tmp_negate_r_correction;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();

        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_input_thread_timed_index = static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w + tmp_input_timed_index;
        tmp_input_thread_parameter_index = static_cast<size_t>(tmp_thread_index__int) * this->total_parameters_allocated;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = cast(ptr_array_variances_received[tmp_input_timed_index + tmp_input_index]);
            tmp_negate_r_correction = -cast(ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index]); // Negate.

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error * cast(ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index]);
                
            // Derivative shift.
            // dShift += dY
            ptr_array_derivatives_shifts_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error;

            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= cast(ptr_array_scales_received[tmp_input_index]);
                
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * ( tmp_negate_r_correction / tmp_variance_b );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * (cast(ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index]) - cast(ptr_array_means_received[tmp_input_timed_index + tmp_input_index])) * ( tmp_negate_r_correction / (tmp_variance_b * tmp_variance_b) );
                
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
        
    // Reduction.
    #pragma omp for schedule(static)
    for(tmp_input_index__int = 0; tmp_input_index__int < tmp_units_size__int; ++tmp_input_index__int)
    {
        tmp_reduction_derivative_mean = 0_r;
        tmp_reduction_derivative_variance = 0_r;
            
        tmp_ptr_array_derivative_mean = ptr_array_derivatives_means_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));
        tmp_ptr_array_derivative_variance = ptr_array_derivatives_variances_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));

        for(tmp_thread_index__int = 1; tmp_thread_index__int != tmp_number_threads__int; ++tmp_thread_index__int)
        {
            tmp_reduction_derivative_mean += tmp_ptr_array_derivative_mean[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
            tmp_reduction_derivative_variance += tmp_ptr_array_derivative_variance[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
        }

        *tmp_ptr_array_derivative_mean += tmp_reduction_derivative_mean;
        *tmp_ptr_array_derivative_variance += tmp_reduction_derivative_variance;
    }
        
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = cast(ptr_array_variances_received[tmp_input_timed_index + tmp_input_index]);

            // First
            // dX_h *= r_correction / variance_b
            tmp_error *= cast(ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index]) / tmp_variance_b;
                
            // Middle
            // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
            tmp_error += ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] * ( (cast(ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index]) - cast(ptr_array_means_received[tmp_input_timed_index + tmp_input_index])) / (static_cast<real>(batch_size) * tmp_variance_b) );

            // Last
            // dX_h += dMean_b * 1 / N
            // dX_h += dMean_b / N
            tmp_error += ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] * tmp_batch_scale;

            // dX = dX_h
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
}

void Model::Backward_Pass__Batch_Renormalization__OpenMP(size_t const time_step_index_received,
                                                                                                             size_t const batch_size,
                                                                                                             size_t const input_size_received,
                                                                                                             var const *const ptr_array_means_received,
                                                                                                             var const *const ptr_array_variances_received,
                                                                                                             var const *const ptr_array_scales_received,
                                                                                                             var const *const ptr_array_r_corrections_received,
                                                                                                             var const *const ptr_array_inputs_received,
                                                                                                             var const *const ptr_array_inputs_hats_received,
                                                                                                             real const *const ptr_array_derivative_inputs_received,
                                                                                                             real *const ptr_array_derivatives_scales_received,
                                                                                                             real *const ptr_array_derivatives_means_received,
                                                                                                             real *const ptr_array_derivatives_variances_received,
                                                                                                             real *const ptr_array_derivatives_received)
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
              tmp_input_thread_timed_index,
              tmp_input_thread_parameter_index;
    
    real const tmp_batch_scale(1_r / static_cast<real>(batch_size));
    real *tmp_ptr_array_derivative_mean,
         *tmp_ptr_array_derivative_variance,
         tmp_reduction_derivative_mean,
         tmp_reduction_derivative_variance,
         tmp_error,
         tmp_variance_b,
         tmp_negate_r_correction;
    
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_thread_index__int = omp_get_thread_num();

        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;
        tmp_input_thread_timed_index = static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w + tmp_input_timed_index;
        tmp_input_thread_parameter_index = static_cast<size_t>(tmp_thread_index__int) * this->total_parameters_allocated;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivative_inputs_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = cast(ptr_array_variances_received[tmp_input_timed_index + tmp_input_index]);
            tmp_negate_r_correction = -cast(ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index]); // Negate.

            // Derivative scale.
            // dScale += dY * value_hat
            ptr_array_derivatives_scales_received[tmp_input_thread_parameter_index + tmp_input_index] += tmp_error * cast(ptr_array_inputs_hats_received[tmp_input_data_timed_index + tmp_input_index]);

            // Derivative value hat.
            // dX_h = dY * scale
            tmp_error *= cast(ptr_array_scales_received[tmp_input_index]);
                
            // dMean_b += dX_h * ( -r_correction / variance_b )
            ptr_array_derivatives_means_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * ( tmp_negate_r_correction / tmp_variance_b );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            ptr_array_derivatives_variances_received[tmp_input_thread_timed_index + tmp_input_index] += tmp_error * (cast(ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index]) - cast(ptr_array_means_received[tmp_input_timed_index + tmp_input_index])) * ( tmp_negate_r_correction / (tmp_variance_b * tmp_variance_b) );
                
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
        
    // Reduction.
    #pragma omp for schedule(static)
    for(tmp_input_index__int = 0; tmp_input_index__int < tmp_units_size__int; ++tmp_input_index__int)
    {
        tmp_reduction_derivative_mean = 0_r;
        tmp_reduction_derivative_variance = 0_r;
            
        tmp_ptr_array_derivative_mean = ptr_array_derivatives_means_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));
        tmp_ptr_array_derivative_variance = ptr_array_derivatives_variances_received + (tmp_input_timed_index + static_cast<size_t>(tmp_input_index__int));

        for(tmp_thread_index__int = 1; tmp_thread_index__int != tmp_number_threads__int; ++tmp_thread_index__int)
        {
            tmp_reduction_derivative_mean += tmp_ptr_array_derivative_mean[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
            tmp_reduction_derivative_variance += tmp_ptr_array_derivative_variance[static_cast<size_t>(tmp_thread_index__int) * input_size_received * this->seq_w];
        }

        *tmp_ptr_array_derivative_mean += tmp_reduction_derivative_mean;
        *tmp_ptr_array_derivative_variance += tmp_reduction_derivative_variance;
    }
        
    #pragma omp for schedule(static)
    for(tmp_example_index__int = 0; tmp_example_index__int < tmp_batch_size__int; ++tmp_example_index__int)
    {
        tmp_input_data_timed_index = static_cast<size_t>(tmp_example_index__int) * input_size_received + tmp_input_timed_batched_index;

        for(tmp_input_index = 0_UZ; tmp_input_index != input_size_received; ++tmp_input_index)
        {
            tmp_error = ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index];
            tmp_variance_b = cast(ptr_array_variances_received[tmp_input_timed_index + tmp_input_index]);

            // First
            // dX_h *= r_correction / variance_b
            tmp_error *= cast(ptr_array_r_corrections_received[tmp_input_timed_index + tmp_input_index]) / tmp_variance_b;
                
            // Middle
            // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
            tmp_error += ptr_array_derivatives_variances_received[tmp_input_timed_index + tmp_input_index] * ( (cast(ptr_array_inputs_received[tmp_input_data_timed_index + tmp_input_index]) - cast(ptr_array_means_received[tmp_input_timed_index + tmp_input_index])) / (static_cast<real>(batch_size) * tmp_variance_b) );

            // Last
            // dX_h += dMean_b * 1 / N
            // dX_h += dMean_b / N
            tmp_error += ptr_array_derivatives_means_received[tmp_input_timed_index + tmp_input_index] * tmp_batch_scale;

            // dX = dX_h
            ptr_array_derivatives_received[tmp_input_data_timed_index + tmp_input_index] = tmp_error;
        }
    }
}
}
