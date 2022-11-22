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

#include <iostream>

namespace DL::v1 {
bool Model::transfer_learning(Model *&model_dst) const
{
    // copy connections.
    Layer const *tmp_ptr_source_last_layer,
                               *tmp_ptr_destination_last_layer,
                               *tmp_ptr_source_layer_it;
    Layer *tmp_ptr_destination_layer_it;

    if(this->type == MODEL::AUTOENCODER
      &&
      model_dst->type == MODEL::AUTOENCODER)
    {
        // Input ... Coded.
        tmp_ptr_source_last_layer = this->ptr_array_layers + std::min((this->total_layers - 3_UZ) / 2_UZ + 2_UZ, (model_dst->total_layers - 3_UZ) / 2_UZ + 2_UZ); // First decoded layer.
        tmp_ptr_destination_last_layer = model_dst->ptr_array_layers + std::min((this->total_layers - 3_UZ) / 2_UZ + 2_UZ, (model_dst->total_layers - 3_UZ) / 2_UZ + 2_UZ); // First decoded layer.

        //  Compare dimensions, type activation, bias.
        for(tmp_ptr_destination_layer_it = model_dst->ptr_array_layers,
            tmp_ptr_source_layer_it = this->ptr_array_layers; tmp_ptr_source_layer_it != tmp_ptr_source_last_layer; ++tmp_ptr_source_layer_it,
                                                                                                                                                                    ++tmp_ptr_destination_layer_it)
        {
            if(tmp_ptr_source_layer_it->Compare__Dimensions(*tmp_ptr_destination_layer_it) == false)
            {
                ERR(L"Layer dimensions unequal.",);

                return false;
            }

            tmp_ptr_destination_layer_it->type_activation = tmp_ptr_source_layer_it->type_activation;
            
            //  Bias.
            if(tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index != 0_UZ)
            {
                VARCOPY(model_dst->ptr_array_parameters + tmp_ptr_destination_layer_it->first_bias_connection_index,
                               this->ptr_array_parameters + tmp_ptr_source_layer_it->first_bias_connection_index,
                               (tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index) * sizeof(var));
            }
            //  |END| Bias. |END|
        }
        //  |END| Compare dimensions, type activation, bias. |END|

        //  Unit(s).
        model_dst->Copy__AF_Units(0_UZ,
                                                                                                static_cast<size_t>(tmp_ptr_source_last_layer->ptr_array_AF_units - this->ptr_array_AF_units),
                                                                                                this->ptr_array_AF_units);
        
        model_dst->Copy__AF_Ind_Recurrent_Units(0_UZ,
                                                                                                                      static_cast<size_t>(tmp_ptr_source_last_layer->ptr_array_AF_Ind_recurrent_units - this->ptr_array_AF_Ind_recurrent_units),
                                                                                                                      this->ptr_array_AF_Ind_recurrent_units,
                                                                                                                      false);
        
        model_dst->Copy__Blocks(0_UZ,
                                                                                            static_cast<size_t>(tmp_ptr_source_last_layer->ptr_array_block_units - this->ptr_array_block_units),
                                                                                            this->ptr_array_block_units,
                                                                                            false);
        //  |END| Unit(s). |END|
        
        //  Dropout.
        model_dst->Copy__Dropout(this->ptr_array_layers,
                                                                                              tmp_ptr_source_last_layer,
                                                                                              model_dst->ptr_array_layers);
        //  |END| Dropout. |END|
        
        //  Normalization.
        model_dst->Copy__Normalization(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                      tmp_ptr_source_last_layer,
                                                                                                      model_dst->ptr_array_layers + 1); // Skip input layer.

        model_dst->Copy__Normalized_Units(0_UZ,
                                                                                                            static_cast<size_t>(tmp_ptr_source_last_layer->ptr_array_normalized_units - this->ptr_array_normalized_units),
                                                                                                            this->ptr_array_normalized_units);
        //  |END| Normalization. |END|

        //  k-Sparse.
        model_dst->Copy__Sparse_K_Filters(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                           tmp_ptr_source_last_layer,
                                                                                                           model_dst->ptr_array_layers + 1); // Skip input layer.
        //  |END| k-Sparse. |END|
        
        //  Constraint recurrent weight.
        model_dst->Copy__Constraint_Recurrent_Weight(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                                             tmp_ptr_source_last_layer,
                                                                                                                             model_dst->ptr_array_layers + 1); // Skip input layer.
        //  |END| Constraint recurrent weight. |END|
        
        //  Weights.
        VARCOPY(model_dst->ptr_array_parameters,
                        this->ptr_array_parameters,
                        *tmp_ptr_source_last_layer->ptr_first_connection_index * sizeof(var));
        //  |END| Weights. |END|
        // |END| Input ... Coded. |END|
        
        // Coded ... Output.
        Layer const *tmp_ptr_source_layer_begin(this->ptr_last_layer - std::min((this->total_layers - 3_UZ) / 2_UZ + 1_UZ, (model_dst->total_layers - 3_UZ) / 2_UZ + 1_UZ)); // First decoded layer.
        Layer *tmp_ptr_destination_layer_begin(model_dst->ptr_last_layer - std::min((this->total_layers - 3_UZ) / 2_UZ + 1_UZ, (model_dst->total_layers - 3_UZ) / 2_UZ + 1_UZ)); // First decoded layer.

        tmp_ptr_source_last_layer = this->ptr_last_layer - 1; // Get output layer.
        tmp_ptr_destination_last_layer = model_dst->ptr_last_layer - 1; // Get output layer.
        
        // Compare dimensions, type activation, bias.
        for(tmp_ptr_destination_layer_it = tmp_ptr_destination_layer_begin,
            tmp_ptr_source_layer_it = tmp_ptr_source_layer_begin; tmp_ptr_source_layer_it != tmp_ptr_source_last_layer; ++tmp_ptr_source_layer_it,
                                                                                                                                                                             ++tmp_ptr_destination_layer_it)
        {
            if(tmp_ptr_source_layer_it->Compare__Dimensions(*tmp_ptr_destination_layer_it) == false)
            {
                ERR(L"Layer dimensions unequal.",);

                return false;
            }

            tmp_ptr_destination_layer_it->type_activation = tmp_ptr_source_layer_it->type_activation;
            
            //  Bias.
            if(tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index != 0_UZ)
            {
                VARCOPY(model_dst->ptr_array_parameters + tmp_ptr_destination_layer_it->first_bias_connection_index,
                               this->ptr_array_parameters + tmp_ptr_source_layer_it->first_bias_connection_index,
                               (tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index) * sizeof(var));
            }
            //  |END| Bias. |END|
        }

        //  Output layer, Bias.
        if(tmp_ptr_source_last_layer->Compare__Dimensions(*tmp_ptr_destination_last_layer))
        {
            VARCOPY(model_dst->ptr_array_parameters + tmp_ptr_destination_last_layer->first_bias_connection_index,
                            this->ptr_array_parameters + tmp_ptr_source_last_layer->first_bias_connection_index,
                            (tmp_ptr_source_last_layer->last_bias_connection_index - tmp_ptr_source_last_layer->first_bias_connection_index) * sizeof(var));
        }
        //  |END| Output layer, Bias. |END|
        // |END| Compare dimensions, type activation, bias. |END|
        
        //  Unit(s).
        if(tmp_ptr_source_last_layer != tmp_ptr_source_layer_begin)
        {
            model_dst->Copy__AF_Units(static_cast<size_t>(tmp_ptr_destination_layer_begin->ptr_array_AF_units - model_dst->ptr_array_AF_units),
                                                                                                    static_cast<size_t>(tmp_ptr_destination_last_layer->ptr_array_AF_units - model_dst->ptr_array_AF_units),
                                                                                                    this->ptr_array_AF_units + static_cast<size_t>(tmp_ptr_source_layer_begin->ptr_array_AF_units - this->ptr_array_AF_units));
            
            model_dst->Copy__AF_Ind_Recurrent_Units(static_cast<size_t>(tmp_ptr_destination_layer_begin->ptr_array_AF_Ind_recurrent_units - model_dst->ptr_array_AF_Ind_recurrent_units),
                                                                                                                          static_cast<size_t>(tmp_ptr_destination_last_layer->ptr_array_AF_Ind_recurrent_units - model_dst->ptr_array_AF_Ind_recurrent_units),
                                                                                                                          this->ptr_array_AF_Ind_recurrent_units + static_cast<size_t>(tmp_ptr_source_layer_begin->ptr_array_AF_Ind_recurrent_units - this->ptr_array_AF_Ind_recurrent_units),
                                                                                                                          false);
            
            model_dst->Copy__Blocks(static_cast<size_t>(tmp_ptr_destination_layer_begin->ptr_array_block_units - model_dst->ptr_array_block_units),
                                                                                                static_cast<size_t>(tmp_ptr_destination_last_layer->ptr_array_block_units - model_dst->ptr_array_block_units),
                                                                                                this->ptr_array_block_units + static_cast<size_t>(tmp_ptr_source_layer_begin->ptr_array_block_units - this->ptr_array_block_units),
                                                                                                false);
        }
        //  |END| Unit(s). |END|
        
        //  Weights.
        if(tmp_ptr_source_last_layer->Compare__Dimensions(*tmp_ptr_destination_last_layer))
        {
            VARCOPY(model_dst->ptr_array_parameters + *tmp_ptr_destination_layer_begin->ptr_first_connection_index,
                            this->ptr_array_parameters + *tmp_ptr_source_layer_begin->ptr_first_connection_index,
                            (*tmp_ptr_source_last_layer->ptr_last_connection_index - *tmp_ptr_source_layer_begin->ptr_first_connection_index) * sizeof(var));
        }
        else if(*tmp_ptr_source_last_layer->ptr_first_connection_index - *tmp_ptr_source_layer_begin->ptr_first_connection_index != 0_UZ)
        {
            VARCOPY(model_dst->ptr_array_parameters + *tmp_ptr_destination_layer_begin->ptr_first_connection_index,
                            this->ptr_array_parameters + *tmp_ptr_source_layer_begin->ptr_first_connection_index,
                            (*tmp_ptr_source_last_layer->ptr_first_connection_index - *tmp_ptr_source_layer_begin->ptr_first_connection_index) * sizeof(var));
        }
        //  |END| Weights. |END|
        // |END| Coded ... Output. |END|
    }
    else
    {
        tmp_ptr_source_last_layer = this->ptr_array_layers + std::min(this->total_layers, model_dst->total_layers) - 1_UZ; // Get output layer.
        tmp_ptr_destination_last_layer = model_dst->ptr_array_layers + std::min(this->total_layers, model_dst->total_layers) - 1_UZ; // Get output layer.

        // Compare dimensions, type activation, bias.
        for(tmp_ptr_destination_layer_it = model_dst->ptr_array_layers,
            tmp_ptr_source_layer_it = this->ptr_array_layers; tmp_ptr_source_layer_it != tmp_ptr_source_last_layer; ++tmp_ptr_source_layer_it,
                                                                                                                                                                    ++tmp_ptr_destination_layer_it)
        {
            if(tmp_ptr_source_layer_it->Compare__Dimensions(*tmp_ptr_destination_layer_it) == false)
            {
                ERR(L"Layer dimensions unequal.",);

                return false;
            }

            tmp_ptr_destination_layer_it->type_activation = tmp_ptr_source_layer_it->type_activation;
            
            //  Bias.
            if(tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index != 0_UZ)
            {
                VARCOPY(model_dst->ptr_array_parameters + tmp_ptr_destination_layer_it->first_bias_connection_index,
                               this->ptr_array_parameters + tmp_ptr_source_layer_it->first_bias_connection_index,
                               (tmp_ptr_source_layer_it->last_bias_connection_index - tmp_ptr_source_layer_it->first_bias_connection_index) * sizeof(var));
            }
            //  |END| Bias. |END|
        }

        //  Output layer, Bias.
        if(tmp_ptr_source_last_layer->Compare__Dimensions(*tmp_ptr_destination_last_layer))
        {
            VARCOPY(model_dst->ptr_array_parameters + tmp_ptr_destination_last_layer->first_bias_connection_index,
                            this->ptr_array_parameters + tmp_ptr_source_last_layer->first_bias_connection_index,
                            (tmp_ptr_source_last_layer->last_bias_connection_index - tmp_ptr_source_last_layer->first_bias_connection_index) * sizeof(var));
        }
        //  |END| Output layer, Bias. |END|
        // |END| Compare dimensions, type activation, bias. |END|
        
        // Unit(s).
        model_dst->Copy__AF_Units(0_UZ,
                                                                                                model_dst->total_AF_units,
                                                                                                this->ptr_array_AF_units);
        
        model_dst->Copy__AF_Ind_Recurrent_Units(0_UZ,
                                                                                                                      model_dst->total_AF_Ind_recurrent_units,
                                                                                                                      this->ptr_array_AF_Ind_recurrent_units,
                                                                                                                      false);
        
        model_dst->Copy__Blocks(0_UZ,
                                                                                            model_dst->total_block_units,
                                                                                            this->ptr_array_block_units,
                                                                                            false);
        // |END| Unit(s). |END|

        // Dropout.
        model_dst->Copy__Dropout(this->ptr_array_layers,
                                                                                              tmp_ptr_source_last_layer,
                                                                                              model_dst->ptr_array_layers);
        // |END| Dropout. |END|
        
        // Normalization.
        model_dst->Copy__Normalization(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                      tmp_ptr_source_last_layer,
                                                                                                      model_dst->ptr_array_layers + 1); // Skip input layer.

        model_dst->Copy__Normalized_Units(0_UZ,
                                                                                                            model_dst->total_normalized_units,
                                                                                                            this->ptr_array_normalized_units);
        // |END| Normalization. |END|
        
        // k-Sparse.
        model_dst->Copy__Sparse_K_Filters(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                           tmp_ptr_source_last_layer,
                                                                                                           model_dst->ptr_array_layers + 1); // Skip input layer.
        // |END| k-Sparse. |END|
        
        // Constraint recurrent weight.
        model_dst->Copy__Constraint_Recurrent_Weight(this->ptr_array_layers + 1, // Skip input layer.
                                                                                                                             tmp_ptr_source_last_layer,
                                                                                                                             model_dst->ptr_array_layers + 1); // Skip input layer.
        // |END| Constraint recurrent weight. |END|
        
        // Weights.
        if(tmp_ptr_source_last_layer->Compare__Dimensions(*tmp_ptr_destination_last_layer))
        {
            VARCOPY(model_dst->ptr_array_parameters,
                           this->ptr_array_parameters,
                           *tmp_ptr_source_last_layer->ptr_last_connection_index * sizeof(var));
        }
        else
        {
            VARCOPY(model_dst->ptr_array_parameters,
                           this->ptr_array_parameters,
                           *tmp_ptr_source_last_layer->ptr_first_connection_index * sizeof(var));
        }
        // |END| Weights. |END|
    }
    // |END| copy connections. |END|
    
    // Normalization.
    model_dst->Copy__Normalization(this);
    // |END| Normalization. |END|
    
    // Initializer weight parameters.
    model_dst->Copy__Initializer__Weight_Parameter(*this);
    // |END| Initializer weight parameters. |END|
    
    // Training parameters.
    model_dst->Copy__Training_Parameters(this);
    // |END| Training parameters. |END|
    
    // Optimizer parameters.
    if(model_dst->Copy__Optimizer_Parameters(this) == false)
    {
        ERR(L"An error has been triggered from the \"Copy__Optimizer_Parameters(ptr)\" function.",);

        return false;
    }
    // |END| Optimizer parameters. |END|

    // Regularization parameters.
    model_dst->Copy__Regularization(this);
    // |END| Regularization parameters. |END|
    
    // Compute parameters.
    model_dst->maximum_allowable_memory_bytes = this->maximum_allowable_memory_bytes;
    
    if(model_dst->Set__Maximum_Thread_Usage(this->pct_threads) == false)
    {
        ERR(L"An error has been triggered from the \"Set__Maximum_Thread_Usage(%f)\" function.",
                                    this->pct_threads);

        return false;
    }
    else if(model_dst->set_max_batch_size(this->maximum_batch_size) == false)
    {
        ERR(L"An error has been triggered from the \"set_max_batch_size(%zu)\" function.",
                                 this->maximum_batch_size);

        return false;
    }
    // |END| Compute parameters. |END|

    return true;
}
}
