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

// PCH:
#include "pch.hpp"

// File header:
#include "deep-learning/v1/learner/model.hpp"

// Deep learning:
#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/io/logger.hpp"

using namespace DL::Str;

namespace DL::v1 {
bool Model::Prepare__Normalized__Layers(void)
{
    this->total_normalized_units = 0_UZ;

    Layer const *const last_layer(this->ptr_last_layer);
    Layer *layer_it(this->ptr_array_layers);
    
    for(; layer_it != last_layer; ++layer_it)
    {
        if(this->Prepare__Normalized__Layer(layer_it) == false)
        {
            ERR(L"An error has been triggered from the "
                L"`Prepare__Normalized__Layer()` function.");
            return false;
        }
    }

    return true;
}

bool Model::Prepare__Normalized__Layer(Layer *&layer_it)
{
    Layer const *tmp_ptr_residual_block_last_layer;

    switch(layer_it->type_layer)
    {
        case LAYER::AVERAGE_POOLING:
        case LAYER::MAX_POOLING: break;
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case LAYER::FULLY_CONNECTED_RECURRENT:
            layer_it->ptr_array_normalized_units = nullptr;
            layer_it->ptr_last_normalized_unit = layer_it->ptr_array_normalized_units + *layer_it->ptr_number_outputs;

            this->total_normalized_units += static_cast<size_t>(layer_it->ptr_last_normalized_unit - layer_it->ptr_array_normalized_units);
                break;
        case LAYER::LSTM:
            layer_it->ptr_array_normalized_units = nullptr;
            layer_it->ptr_last_normalized_unit = layer_it->ptr_array_normalized_units + 3_UZ * static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units)
                                                                                                                                                                  +
                                                                                                                                                               6_UZ * static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units);

            this->total_normalized_units += static_cast<size_t>(layer_it->ptr_last_normalized_unit - layer_it->ptr_array_normalized_units);
                break;
        case LAYER::RESIDUAL:
            tmp_ptr_residual_block_last_layer = layer_it + layer_it->block_depth;

            layer_it->ptr_array_normalized_units = nullptr;
            layer_it->ptr_last_normalized_unit = layer_it->ptr_array_normalized_units + *tmp_ptr_residual_block_last_layer->ptr_number_outputs;

            this->total_normalized_units += static_cast<size_t>(layer_it->ptr_last_normalized_unit - layer_it->ptr_array_normalized_units);

            if(this->Prepare__Normalized__Residual_Block(layer_it) == false)
            {
                ERR(L"An error has been triggered from the "
                    L"`Prepare__Normalized__Residual_Block()` function.");
                return false;
            }
                break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in the switch.",
                layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
            return false;
    }

    return true;
}

bool Model::Prepare__Normalized__Residual_Block(Layer *&layer_it)
{
    if(layer_it->type_layer != LAYER::RESIDUAL)
    {
        ERR(L"Layer received as argument is not a residual unit.");
        return false;
    }
    
    Layer const *const tmp_ptr_residual_block_end(layer_it + layer_it->block_depth + 1);
    
    for(++layer_it; layer_it != tmp_ptr_residual_block_end; ++layer_it)
    {
        if(this->Prepare__Normalized__Residual_Layer(layer_it) == false)
        {
            ERR(L"An error has been triggered from the "
                L"`Prepare__Normalized__Residual_Layer()` function.");
            return false;
        }
    }
    
    // Assign layer iterator to the last layer inside the block.
    --layer_it;

    return true;
}

bool Model::Prepare__Normalized__Residual_Layer(Layer *&layer_it)
{
    Layer const *const tmp_ptr_previous_layer_connected(layer_it->previous_connected_layers[0]),
                               *tmp_ptr_residual_block_last_layer;

    switch(layer_it->type_layer)
    {
        case LAYER::AVERAGE_POOLING:
        case LAYER::MAX_POOLING: break;
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case LAYER::FULLY_CONNECTED_RECURRENT:
            layer_it->ptr_array_normalized_units = nullptr;
            layer_it->ptr_last_normalized_unit = layer_it->ptr_array_normalized_units + *tmp_ptr_previous_layer_connected->ptr_number_outputs;

            this->total_normalized_units += static_cast<size_t>(layer_it->ptr_last_normalized_unit - layer_it->ptr_array_normalized_units);
                break;
        case LAYER::LSTM:
            layer_it->ptr_array_normalized_units = nullptr;
            layer_it->ptr_last_normalized_unit = layer_it->ptr_array_normalized_units + 3_UZ * static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units)
                                                                                                                                                                  +
                                                                                                                                                               6_UZ * static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units);

            this->total_normalized_units += static_cast<size_t>(layer_it->ptr_last_normalized_unit - layer_it->ptr_array_normalized_units);
                break;
        case LAYER::RESIDUAL:
            tmp_ptr_residual_block_last_layer = layer_it + layer_it->block_depth;

            layer_it->ptr_array_normalized_units = nullptr;
            layer_it->ptr_last_normalized_unit = layer_it->ptr_array_normalized_units + *tmp_ptr_residual_block_last_layer->ptr_number_outputs;

            this->total_normalized_units += static_cast<size_t>(layer_it->ptr_last_normalized_unit - layer_it->ptr_array_normalized_units);

            if(this->Prepare__Normalized__Residual_Block(layer_it) == false)
            {
                ERR(L"An error has been triggered from the "
                    L"`Prepare__Normalized__Residual_Block()` function.");
                return false;
            }
                break;
        default:
            ERR(L"Layer type (%d | %ls) is not managed in the switch.",
                layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
            return false;
    }

    return true;
}

bool Model::Set__Layer_Normalization(size_t const index_layer_received,
                                                                       LAYER_NORM::TYPE const type_layer_normalization_received,
                                                                       bool const reallocate_dimension_parameters_received,
                                                                       bool const organize_pointers_received)
{
    if(index_layer_received >= this->total_layers) {
        ERR(L"Layer received (%zu) as argument overflow the "
            L"number of layers (%zu) in the neural network.",
            index_layer_received, this->total_layers);
        return false;
    }
    else if(this->ptr_array_layers == nullptr)
    {
        ERR(L"Layer received as argument is a nullptr.");
        return false;
    }

    return(this->Set__Layer_Normalization(this->ptr_array_layers + index_layer_received,
                                                            type_layer_normalization_received,
                                                            reallocate_dimension_parameters_received,
                                                            organize_pointers_received));
}

bool Model::Set__Layer_Normalization(Layer *const ptr_layer_received,
                                                                       LAYER_NORM::TYPE const type_layer_normalization_received,
                                                                       bool const reallocate_dimension_parameters_received,
                                                                       bool const organize_pointers_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"Layer received as argument is a nullptr.");
        return false;
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        ERR(L"Layer received as argument is the input layer.");
        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.");
        return false;
    }
    else if(this->type == MODEL::AUTOENCODER
             &&
             ptr_layer_received >= this->ptr_last_layer - (this->total_layers - 3_UZ) / 2_UZ + 1_UZ)
    {
        ERR(L"Layer received as argument is a decoded layer.");
        return false;
    }

    switch(type_layer_normalization_received)
    {
        case LAYER_NORM::NONE: return(this->Set__Normalization_None(ptr_layer_received, organize_pointers_received));
        case LAYER_NORM::BATCH_NORMALIZATION:
            return(this->Set__Batch_Normalization(ptr_layer_received,
                                                                     true,
                                                                     reallocate_dimension_parameters_received,
                                                                     organize_pointers_received));
        case LAYER_NORM::BATCH_RENORMALIZATION:
            return(this->Set__Batch_Renormalization(ptr_layer_received,
                                                                        true,
                                                                        reallocate_dimension_parameters_received,
                                                                        organize_pointers_received));
        case LAYER_NORM::GHOST_BATCH_NORMALIZATION:
            return(this->Set__Ghost_Batch_Normalization(ptr_layer_received,
                                                                               true,
                                                                               reallocate_dimension_parameters_received,
                                                                               organize_pointers_received));
        default:
            ERR(L"Type normalization layer (%d | %ls) is not managed "
                L"in the switch.",
                ptr_layer_received->type_normalization,
                LAYER_NORM_NAME[ptr_layer_received->type_normalization]
                    .c_str());
            return false;
    }
}

bool Model::Set__Normalization_None(Layer *const ptr_layer_received, bool const organize_pointers_received)
{
    switch(ptr_layer_received->type_normalization)
    {
        case LAYER_NORM::NONE:
            if(organize_pointers_received) { this->Order__Layer__Output(false, ptr_layer_received); }
                return true;
        case LAYER_NORM::BATCH_NORMALIZATION:
            return(this->Set__Batch_Normalization(ptr_layer_received,
                                                                   false,
                                                                   false,
                                                                   false));
        case LAYER_NORM::BATCH_RENORMALIZATION:
            return(this->Set__Batch_Renormalization(ptr_layer_received,
                                                                      false,
                                                                      false,
                                                                      false));
        case LAYER_NORM::GHOST_BATCH_NORMALIZATION:
            return(this->Set__Ghost_Batch_Normalization(ptr_layer_received,
                                                                                false,
                                                                                false,
                                                                                false));
        default:
            ERR(L"Type normalization layer (%d | %ls) is not managed "
                L"in the switch.",
                ptr_layer_received->type_normalization,
                LAYER_NORM_NAME[ptr_layer_received->type_normalization]
                    .c_str());
            return false;
    }
}

bool Model::Set__Batch_Normalization(Layer *const ptr_layer_received,
                                                                       bool const use_batch_normalization_received,
                                                                       bool const reallocate_dimension_parameters_received,
                                                                       bool const organize_pointers_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"Layer received as argument is a nullptr.");
        return false;
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        ERR(L"Layer received as argument is the input layer.");
        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.");
        return false;
    }
    
    if(ptr_layer_received->type_normalization != LAYER_NORM::BATCH_NORMALIZATION)
    {
        if(this->Set__Normalization_None(ptr_layer_received, organize_pointers_received) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Normalization_None(%ls)` function.",
                to_wstring(organize_pointers_received).c_str());
            return false;
        }
    }

    if(use_batch_normalization_received && ptr_layer_received->type_normalization == LAYER_NORM::NONE)
    {
        ptr_layer_received->type_normalization = LAYER_NORM::BATCH_NORMALIZATION;

        bool const tmp_normalization_initialized(this->Use__Normalization());

        if(++this->total_batch_normalization_layers == 1_UZ)
        {
            if(this->Allocate__Normalized_Unit(organize_pointers_received) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Allocate__Normalized_Unit(%ls)` function.",
                    to_wstring(organize_pointers_received).c_str());
                
                ptr_layer_received->type_normalization = LAYER_NORM::NONE;

                --this->total_batch_normalization_layers;

                return false;
            }
            else if(tmp_normalization_initialized == false
                     &&
                     reallocate_dimension_parameters_received
                     &&
                     this->Allocate__Parameter__Normalization() == false)
            {
                ERR(L"An error has been triggered from the "
                    L"`Allocate__Parameter__Normalization()` function.");
                
                ptr_layer_received->type_normalization = LAYER_NORM::NONE;

                --this->total_batch_normalization_layers;

                return false;
            }
            else if(this->Allocate__Normalized_Unit__Batch_Normalization() == false)
            {
                ERR(L"An error has been triggered from the "
                    L"`Allocate__Normalized_Unit__Batch_Normalization()` "
                    L"function.");
                
                ptr_layer_received->type_normalization = LAYER_NORM::NONE;

                --this->total_batch_normalization_layers;

                return false;
            }
        }
    }
    else if(use_batch_normalization_received == false && ptr_layer_received->type_normalization == LAYER_NORM::BATCH_NORMALIZATION)
    {
        ptr_layer_received->type_normalization = LAYER_NORM::NONE;
        
        if(this->total_batch_normalization_layers != 0_UZ
           &&
           --this->total_batch_normalization_layers == 0u
           &&
           this->Use__Normalization() == false)
        {
            this->Deallocate__Parameter__Batch_Normalization();
            
            this->Deallocate__Normalized_Unit();

            this->Deallocate__Normalized_Unit__Batch_Normalization();
        }
    }
    
    if(organize_pointers_received) { this->Order__Layer__Output(false, ptr_layer_received); }
    
    // Mirror layer.
    if(this->type == MODEL::AUTOENCODER
      &&
      ptr_layer_received < this->Get__End_Layer__Active() - 1 // Get last active layer.
      &&
      this->Set__Batch_Normalization(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1,
                                                      use_batch_normalization_received,
                                                      reallocate_dimension_parameters_received,
                                                      organize_pointers_received)) {
        ERR(L"An error has been triggered from the "
            L"`Set__Batch_Normalization(ptr, %ls, %ls, %ls)` function.",
            to_wstring(use_batch_normalization_received).c_str(),
            to_wstring(reallocate_dimension_parameters_received).c_str(),
            to_wstring(organize_pointers_received).c_str());
        return false;
    }
    // |END| Mirror layer. |END|

    return true;
}

bool Model::Set__Batch_Renormalization(Layer *const ptr_layer_received,
                                                                           bool const use_batch_renormalization_received,
                                                                           bool const reallocate_dimension_parameters_received,
                                                                           bool const organize_pointers_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"Layer received as argument is a nullptr.");
        return false;
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        ERR(L"Layer received as argument is the input layer.");
        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.");
        return false;
    }
    
    if(ptr_layer_received->type_normalization != LAYER_NORM::BATCH_RENORMALIZATION)
    {
        if(this->Set__Normalization_None(ptr_layer_received, organize_pointers_received) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Normalization_None(%ls)` function.",
                to_wstring(organize_pointers_received).c_str());
            return false;
        }
    }

    if(use_batch_renormalization_received && ptr_layer_received->type_normalization == LAYER_NORM::NONE)
    {
        ptr_layer_received->type_normalization = LAYER_NORM::BATCH_RENORMALIZATION;
        
        bool const tmp_normalization_initialized(this->Use__Normalization());

        if(++this->total_batch_renormalization_layers == 1_UZ)
        {
            if(this->Allocate__Normalized_Unit(organize_pointers_received) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Allocate__Normalized_Unit(%ls)` function.",
                    to_wstring(organize_pointers_received).c_str());
                
                ptr_layer_received->type_normalization = LAYER_NORM::NONE;

                --this->total_batch_renormalization_layers;

                return false;
            }
            else if(tmp_normalization_initialized == false
                     &&
                     reallocate_dimension_parameters_received
                     &&
                     this->Allocate__Parameter__Normalization() == false) {
                ERR(L"An error has been triggered from the "
                    L"`Allocate__Parameter__Normalization()` function.");
                
                ptr_layer_received->type_normalization = LAYER_NORM::NONE;

                --this->total_batch_renormalization_layers;

                return false;
            }
            else if(this->Allocate__Normalized_Unit__Batch_Normalization() == false) {
                ERR(L"An error has been triggered from the "
                    L"`Allocate__Normalized_Unit__Batch_Normalization()` "
                    L"function.");
                
                ptr_layer_received->type_normalization = LAYER_NORM::NONE;

                --this->total_batch_renormalization_layers;

                return false;
            }
            else if(this->Allocate__Normalized_Unit__Batch_Renormalization() == false) {
                ERR(L"An error has been triggered from the "
                    L"`Allocate__Normalized_Unit__Batch_Renormalization()` "
                    L"function.");
                
                ptr_layer_received->type_normalization = LAYER_NORM::NONE;

                --this->total_batch_renormalization_layers;

                return false;
            }
        }
    }
    else if(use_batch_renormalization_received == false && ptr_layer_received->type_normalization == LAYER_NORM::BATCH_RENORMALIZATION)
    {
        ptr_layer_received->type_normalization = LAYER_NORM::NONE;
        
        if(this->total_batch_renormalization_layers != 0_UZ && --this->total_batch_renormalization_layers == 0_UZ)
        {
            this->Deallocate__Normalized_Unit__Batch_Renormalization();
            
            if(this->Use__Normalization() == false)
            {
                this->Deallocate__Parameter__Batch_Normalization();

                this->Deallocate__Normalized_Unit();

                this->Deallocate__Normalized_Unit__Batch_Normalization();
            }
        }
    }
    
    if(organize_pointers_received) { this->Order__Layer__Output(false, ptr_layer_received); }
    
    // Mirror layer.
    if(this->type == MODEL::AUTOENCODER
      &&
      ptr_layer_received < this->Get__End_Layer__Active() - 1 // Get last active layer.
      &&
      this->Set__Batch_Renormalization(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1,
                                                         use_batch_renormalization_received,
                                                         reallocate_dimension_parameters_received,
                                                         organize_pointers_received)) {
        ERR(L"An error has been triggered from the "
            L"`Set__Batch_Renormalization(ptr, %ls, %ls, %ls)` function.",
            to_wstring(use_batch_renormalization_received).c_str(),
            to_wstring(reallocate_dimension_parameters_received).c_str(),
            to_wstring(organize_pointers_received).c_str());
        return false;
    }
    // |END| Mirror layer. |END|

    return true;
}

bool Model::Set__Ghost_Batch_Normalization(Layer *const ptr_layer_received,
                                                                                  bool const use_ghost_batch_normalization_received,
                                                                                  bool const reallocate_dimension_parameters_received,
                                                                                  bool const organize_pointers_received)
{
    if(ptr_layer_received == nullptr)
    {
        ERR(L"Layer received as argument is a nullptr.");
        return false;
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        ERR(L"Layer received as argument is the input layer.");
        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.");
        return false;
    }
    
    if(ptr_layer_received->type_normalization != LAYER_NORM::GHOST_BATCH_NORMALIZATION)
    {
        if(this->Set__Normalization_None(ptr_layer_received, organize_pointers_received) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Normalization_None(%ls)` function.",
                to_wstring(organize_pointers_received).c_str());
            return false;
        }
    }

    if(use_ghost_batch_normalization_received && ptr_layer_received->type_normalization == LAYER_NORM::NONE)
    {
        ptr_layer_received->type_normalization = LAYER_NORM::GHOST_BATCH_NORMALIZATION;
        
        bool const tmp_normalization_initialized(this->Use__Normalization());

        if(++this->total_ghost_batch_normalization_layers == 1_UZ)
        {
            if(this->Allocate__Normalized_Unit(organize_pointers_received) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Allocate__Normalized_Unit(%ls)` function.",
                    to_wstring(organize_pointers_received).c_str());
                
                ptr_layer_received->type_normalization = LAYER_NORM::NONE;

                --this->total_ghost_batch_normalization_layers;

                return false;
            }
            else if(tmp_normalization_initialized == false
                     &&
                     reallocate_dimension_parameters_received
                     &&
                     this->Allocate__Parameter__Normalization() == false) {
                ERR(L"An error has been triggered from the "
                    L"`Allocate__Parameter__Normalization()` function.");
                
                ptr_layer_received->type_normalization = LAYER_NORM::NONE;

                --this->total_ghost_batch_normalization_layers;

                return false;
            }
            else if(this->Allocate__Normalized_Unit__Batch_Normalization() == false) {
                ERR(L"An error has been triggered from the "
                    L"`Allocate__Normalized_Unit__Batch_Normalization()` "
                    L"function.");
                
                ptr_layer_received->type_normalization = LAYER_NORM::NONE;

                --this->total_ghost_batch_normalization_layers;

                return false;
            }
        }
    }
    else if(use_ghost_batch_normalization_received == false && ptr_layer_received->type_normalization == LAYER_NORM::GHOST_BATCH_NORMALIZATION)
    {
        ptr_layer_received->type_normalization = LAYER_NORM::NONE;
        
        if(this->total_ghost_batch_normalization_layers != 0_UZ
           &&
           --this->total_ghost_batch_normalization_layers == 0u
           &&
           this->Use__Normalization() == false)
        {
            this->Deallocate__Parameter__Batch_Normalization();
            
            this->Deallocate__Normalized_Unit();

            this->Deallocate__Normalized_Unit__Batch_Normalization();
        }
    }
    
    if(organize_pointers_received) { this->Order__Layer__Output(false, ptr_layer_received); }
    
    // Mirror layer.
    if(this->type == MODEL::AUTOENCODER
      &&
      ptr_layer_received < this->Get__End_Layer__Active() - 1 // Get last active layer.
      &&
      this->Set__Ghost_Batch_Normalization(this->ptr_last_layer - static_cast<size_t>(ptr_layer_received - this->ptr_array_layers) - 1,
                                                                use_ghost_batch_normalization_received,
                                                                reallocate_dimension_parameters_received,
                                                                organize_pointers_received)) {
        ERR(L"An error has been triggered from the "
            L"`Set__Ghost_Batch_Normalization(ptr, %ls, %ls, %ls)` function.",
            to_wstring(use_ghost_batch_normalization_received).c_str(),
            to_wstring(reallocate_dimension_parameters_received).c_str(),
            to_wstring(organize_pointers_received).c_str());
        return false;
    }
    // |END| Mirror layer. |END|

    return true;
}
}