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
#include "deep-learning/v1/ops/activations/functions.hpp"

#include <array>

namespace DL::v1 {
real Model::Initialization__Gain__Scale(
    ACTIVATION::TYPE const type) {
    switch(type)
    {
        case ACTIVATION::COSINE:
        case ACTIVATION::COSINE_SYMMETRIC:
        case ACTIVATION::ELU:
        case ACTIVATION::ELLIOT:
        case ACTIVATION::ELLIOT_SYMMETRIC:
        case ACTIVATION::GAUSSIAN:
        case ACTIVATION::GAUSSIAN_STEPWISE:
        case ACTIVATION::GAUSSIAN_SYMMETRIC:
        case ACTIVATION::ISRU:
        case ACTIVATION::ISRLU:
        case ACTIVATION::LINEAR:
        case ACTIVATION::LINEAR_PIECE:
        case ACTIVATION::LINEAR_PIECE_SYMMETRIC:
        case ACTIVATION::SELU:
        case ACTIVATION::SIGMOID:
        case ACTIVATION::SINE:
        case ACTIVATION::SIGMOID_STEPWISE:
        case ACTIVATION::SINE_SYMMETRIC:
        case ACTIVATION::SOFTMAX:
        case ACTIVATION::THRESHOLD:
        case ACTIVATION::THRESHOLD_SYMMETRIC: return 1_r;
        case ACTIVATION::LEAKY_RELU:
          return std::sqrt(2_r / (1_r + std::pow(AF_LRELU_ALPHA, 2_r)));
        case ACTIVATION::PARAMETRIC_RELU:
          return std::sqrt(2_r / (1_r + std::pow(AF_PRELU_ALPHA, 2_r)));
        case ACTIVATION::RELU:
          return std::sqrt(2_r);
        case ACTIVATION::TANH:
        case ACTIVATION::TANH_STEPWISE: return 5_r / 3_r;
        default:
            ERR(L"Activation function type (%d | %ls) is not managed in the switch.",
                                     type,
                                     ACTIVATION_NAME[type].c_str());
                return 1_r;
    }
}

real Model::Initialization__Gaussian__Variance(size_t const n_inp,
                                                                                size_t const n_out,
                                                                                LAYER_ACTIVATION::TYPE const type)
{
    switch(type)
    {
        case LAYER_ACTIVATION::SYMMETRIC: 
        case LAYER_ACTIVATION::ASYMMETRIC:
        case LAYER_ACTIVATION::SOFTMAX: // Xavier Glorot & Yoshua Bengio.
                return std::sqrt(2_r / static_cast<real>(n_inp + n_out));
        case LAYER_ACTIVATION::RECTIFIER: // Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
        case LAYER_ACTIVATION::SELF_NORMALIZATION: // Self-Normalizing Neural Networks.
                return std::sqrt(1_r / static_cast<real>(n_inp));
        default:
            ERR(L"Can not get variance with (%d | %ls) as the type activation layer.",
                                     type,
                                     LAYER_ACTIVATION_NAME[type].c_str());
                return 1_r;
    }
}

real Model::Initialization__Uniform__Variance(size_t const n_inp,
                                                                              size_t const n_out,
                                                                              LAYER_ACTIVATION::TYPE const type)
{
    switch(type)
    {
        case LAYER_ACTIVATION::SYMMETRIC:
        case LAYER_ACTIVATION::ASYMMETRIC:
        case LAYER_ACTIVATION::SOFTMAX: // Xavier Glorot & Yoshua Bengio.
                return std::sqrt(6_r / static_cast<real>(n_inp + n_out));
        case LAYER_ACTIVATION::RECTIFIER: 
        case LAYER_ACTIVATION::SELF_NORMALIZATION: // Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
                return std::sqrt(6_r / static_cast<real>(n_inp));
        default:
            ERR(L"Can not get variance with (%d | %ls) as the type activation layer.",
                                     type,
                                     LAYER_ACTIVATION_NAME[type].c_str());
                return 1_r;
    }
}

void Model::Initialization__Glorot__Gaussian(real const bias)
{
    size_t n_inp,
              n_out;
    
    Layer const *const last_layer(this->ptr_last_layer),
                               *prev_conn_layer,
                               *next_layer_end,
                               *next_layer_it;
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
        
        prev_conn_layer = layer_it->previous_connected_layers[0];
        
        if((n_inp = *prev_conn_layer->ptr_number_outputs) == 0_UZ)
        {
            ERR(L"Can not get \"fan_in\" with (%d | %ls) as the type layer.",
                                     prev_conn_layer->type_layer,
                                     LAYER_NAME[prev_conn_layer->type_layer].c_str());

            continue;
        }

        if(layer_it + 1 != this->ptr_last_layer)
        {
            n_out = 0_UZ;
            
            for(next_layer_it = layer_it->next_connected_layers[0],
                next_layer_end = next_layer_it + 1; next_layer_it != next_layer_end; ++next_layer_it)
            {
                if((n_out += *next_layer_it->ptr_number_outputs) == 0_UZ)
                {
                    ERR(L"Can not get \"fan_out\" with (%d | %ls) as the type layer.",
                                             next_layer_it->type_layer,
                                             LAYER_NAME[next_layer_it->type_layer].c_str());
                }
            }

            n_out /= layer_it->next_connected_layers.size();
        }
        else { n_out = n_inp; }

        switch(layer_it->type_layer)
        {
            case LAYER::FULLY_CONNECTED:
                this->weights_initialize_gaussian(this->ptr_array_parameters + *layer_it->ptr_first_connection_index,
                                                      this->ptr_array_parameters + *layer_it->ptr_last_connection_index,
                                                      this->Initialization__Gain__Scale(*layer_it->ptr_array_AF_units->ptr_type_activation_function) * this->Initialization__Gaussian__Variance(n_inp,
                                                                                                                                                                                                                                                                        n_out,
                                                                                                                                                                                                                                                                        layer_it->type_activation));

                this->layer_initialize_const_bias(bias, layer_it);
                    break;
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                this->weights_initialize_gaussian(this->ptr_array_parameters + *layer_it->ptr_first_connection_index,
                                                      this->ptr_array_parameters + *layer_it->ptr_last_connection_index,
                                                      this->Initialization__Gain__Scale(*layer_it->ptr_array_AF_Ind_recurrent_units->ptr_type_activation_function) * this->Initialization__Gaussian__Variance(n_inp,
                                                                                                                                                                                                                                                                                             n_out,
                                                                                                                                                                                                                                                                                             layer_it->type_activation));
                
                this->indrec_initialize_uniform(layer_it);
                
                this->layer_initialize_const_bias(bias, layer_it);
                    break;
            case LAYER::LSTM:
                this->lstm_initialize_gaussian(this->Initialization__Gain__Scale(layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Gaussian__Variance(n_inp,
                                                                                                                                                                                                                                                                             n_out,
                                                                                                                                                                                                                                                                             layer_it->type_activation),
                                                                  this->Initialization__Gain__Scale(layer_it->ptr_array_block_units->activation_function_gate) * this->Initialization__Gaussian__Variance(n_inp,
                                                                                                                                                                                                                                                                                 n_out,
                                                                                                                                                                                                                                                                                 this->Activation_Function__To__Class_Activation_Function(layer_it->ptr_array_block_units->activation_function_gate)),
                                                                  this->Initialization__Gain__Scale(layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Gaussian__Variance(static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                                             static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                                             layer_it->type_activation),
                                                                 this->Initialization__Gain__Scale(layer_it->ptr_array_block_units->activation_function_gate) * this->Initialization__Gaussian__Variance(static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                                                static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                                                this->Activation_Function__To__Class_Activation_Function(layer_it->ptr_array_block_units->activation_function_gate)),
                                                                 this->Initialization__Gain__Scale(layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Gaussian__Variance(static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units) / static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units),
                                                                                                                                                                                                                                                                            static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units) / static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units),
                                                                                                                                                                                                                                                                            layer_it->type_activation),
                                                                  layer_it);

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

    if(this->ptr_array_derivatives_parameters != nullptr) this->clear_training_arrays();

    if(this->Use__Normalization()) this->Clear__Parameter__Normalized_Unit();

    this->_initialized__weight = true;
    this->_type_weights_initializer = INITIALIZER::GLOROT_GAUSSIAN;
}

void Model::initialize_weights_with_glorot_uniform(real const bias)
{
    size_t n_inp,
              n_out;
    
    real variances[5];

    Layer const *const last_layer(this->ptr_last_layer),
                               *prev_conn_layer,
                               *next_layer_end,
                               *next_layer_it;
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
        
        prev_conn_layer = layer_it->previous_connected_layers[0];
        
        if((n_inp = *prev_conn_layer->ptr_number_outputs) == 0_UZ)
        {
            ERR(L"Can not get \"fan_in\" with (%d | %ls) as the type layer.",
                                     prev_conn_layer->type_layer,
                                     LAYER_NAME[prev_conn_layer->type_layer].c_str());

            continue;
        }
        
        if(layer_it + 1 != this->ptr_last_layer)
        {
            n_out = 0_UZ;
            
            for(next_layer_it = layer_it->next_connected_layers[0],
                next_layer_end = next_layer_it + layer_it->next_connected_layers.size(); next_layer_it != next_layer_end; ++next_layer_it)
            {
                if((n_out += *next_layer_it->ptr_number_outputs) == 0_UZ)
                {
                    ERR(L"Can not get \"fan_out\" with (%d | %ls) as the type layer.",
                                             next_layer_it->type_layer,
                                             LAYER_NAME[next_layer_it->type_layer].c_str());
                }
            }

            n_out /= layer_it->next_connected_layers.size();
        }
        else { n_out = n_inp; }

        switch(layer_it->type_layer)
        {
            case LAYER::FULLY_CONNECTED:
                variances[0] = this->Initialization__Gain__Scale(*layer_it->ptr_array_AF_units->ptr_type_activation_function) * this->Initialization__Uniform__Variance(n_inp,
                                                                                                                                                                                                                                                            n_out,
                                                                                                                                                                                                                                                            layer_it->type_activation);
                
                this->weights_initialize_uniform(this->ptr_array_parameters + *layer_it->ptr_first_connection_index,
                                                    this->ptr_array_parameters + *layer_it->ptr_last_connection_index,
                                                    -variances[0],
                                                    variances[0]);
                
                this->layer_initialize_const_bias(bias, layer_it);
                    break;
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                variances[0] = this->Initialization__Gain__Scale(*layer_it->ptr_array_AF_Ind_recurrent_units->ptr_type_activation_function) * this->Initialization__Uniform__Variance(n_inp,
                                                                                                                                                                                                                                                                                n_out,
                                                                                                                                                                                                                                                                                layer_it->type_activation);
                
                this->weights_initialize_uniform(this->ptr_array_parameters + *layer_it->ptr_first_connection_index,
                                                    this->ptr_array_parameters + *layer_it->ptr_last_connection_index,
                                                    -variances[0],
                                                    variances[0]);

                this->indrec_initialize_uniform(layer_it);

                this->layer_initialize_const_bias(bias, layer_it);
                    break;
            case LAYER::LSTM:
                variances[0] = this->Initialization__Gain__Scale(layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Uniform__Variance(n_inp,
                                                                                                                                                                                                                                                     n_out,
                                                                                                                                                                                                                                                     layer_it->type_activation);
                
                variances[1] = this->Initialization__Gain__Scale(layer_it->ptr_array_block_units->activation_function_gate) * this->Initialization__Uniform__Variance(n_inp,
                                                                                                                                                                                                                                                         n_out,
                                                                                                                                                                                                                                                         this->Activation_Function__To__Class_Activation_Function(layer_it->ptr_array_block_units->activation_function_gate));
                
                variances[2] = this->Initialization__Gain__Scale(layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Uniform__Variance(static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                     static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                     layer_it->type_activation);
                
                variances[3] = this->Initialization__Gain__Scale(layer_it->ptr_array_block_units->activation_function_gate) * this->Initialization__Uniform__Variance(static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                         static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units),
                                                                                                                                                                                                                                                         this->Activation_Function__To__Class_Activation_Function(layer_it->ptr_array_block_units->activation_function_gate));
                
                variances[4] = this->Initialization__Gain__Scale(layer_it->ptr_array_block_units->activation_function_io) * this->Initialization__Uniform__Variance(static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units) / static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units),
                                                                                                                                                                                                                                                     static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units) / static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units),
                                                                                                                                                                                                                                                     layer_it->type_activation);
                
                this->lstm_initialize_uniform(std::array<real, 5_UZ>{-variances[0], -variances[1], -variances[2], -variances[3], -variances[4]}.data(),
                    std::array<real, 5_UZ>{variances[0], variances[1],
                                           variances[2], variances[3],
                                           variances[4]}
                        .data(),
                                                               layer_it);

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

    if(this->ptr_array_derivatives_parameters != nullptr) this->clear_training_arrays();

    if(this->Use__Normalization()) this->Clear__Parameter__Normalized_Unit();

    this->_initialized__weight = true;
    this->_type_weights_initializer = INITIALIZER::GLOROT_UNIFORM;
}
}
