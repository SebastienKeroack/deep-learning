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

#include <omp.h>

namespace DL::v1 {
bool Layer::Use__Tied_Parameter(void) const { return(this->use_tied_parameter); }

bool Model::Set__Tied_Parameter(size_t const index_layer_received,
                                                                bool const use_tied_parameter_received,
                                                                bool const transpose_received)
{
    if(index_layer_received >= this->total_layers)
    {
        ERR(L"Layer received (%zu) overflow the number of layers (%zu) in the neural network.",
                                 index_layer_received,
                                 this->total_layers);

        return false;
    }
    else if(this->ptr_array_layers == nullptr)
    {
        ERR(L"\"ptr_array_layers\" is a nullptr.",);

        return false;
    }

    return(this->Set__Tied_Parameter(this->ptr_array_layers + index_layer_received,
                                                      use_tied_parameter_received,
                                                      transpose_received));
}

bool Model::Set__Tied_Parameter(Layer *const layer,
                                                                bool const use_tied_parameter_received,
                                                                bool const transpose_received)
{
    auto valid_layer_fn([](Layer const *const layer) -> bool
    {
        if(layer->type_group == GROUP::RESIDUAL)
        {
            ERR(L"Group type (%d | %ls) is not managed in the function",
                                        layer->type_group,
                                        GROUP_NAME[layer->type_group].c_str());
            
            return false;
        }

        switch(layer->type_layer)
        {
            case LAYER::FULLY_CONNECTED:
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            case LAYER::FULLY_CONNECTED_RECURRENT: break;
            default:
                ERR(L"Layer type (%d | %ls) is not managed in",
                                         layer->type_layer,
                                         LAYER_NAME[layer->type_layer].c_str());
                    return false;
        }

        return true;
    });

    if(layer == nullptr)
    {
        ERR(L"\"layer\" is a nullptr.",);

        return false;
    }
    else if(layer == this->ptr_array_layers)
    {
        ERR(L"Layer received as argument is the input layer.",);

        return false;
    }
    else if(layer == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.",);

        return false;
    }
    else if(valid_layer_fn(layer) == false)
    {
        ERR(L"An error has been triggered from the \"valid_layer_fn(ptr)\" function.",);

        return false;
    }
    else if(layer->use_tied_parameter == use_tied_parameter_received) { return true; }

    // Mirror.
    if(layer < this->Get__End_Layer__Active() - 1) // Get last active layer.
    {
        Layer *const tmp_ptr_mirror_layer(this->ptr_last_layer - static_cast<size_t>(layer - this->ptr_array_layers) - 1);
        Layer const *const tmp_ptr_previous_layer_it(layer->previous_connected_layers[0]),
                                   *next_layer_it(tmp_ptr_mirror_layer->next_connected_layers[0]),
                                   *const next_layer_end(next_layer_it + tmp_ptr_mirror_layer->next_connected_layers.size());
        
        if(valid_layer_fn(tmp_ptr_previous_layer_it) == false)
        {
            ERR(L"An error has been triggered from the \"valid_layer_fn(ptr)\" function.",);

            return false;
        }
        else if(layer->type_layer != tmp_ptr_mirror_layer->type_layer)
        {
            ERR(L"The layer type (%d | %ls) differ from the mirror layer type (%d | %ls).",
                                     layer->type_layer,
                                     LAYER_NAME[layer->type_layer].c_str(),
                                     tmp_ptr_mirror_layer->type_layer,
                                     LAYER_NAME[tmp_ptr_mirror_layer->type_layer].c_str());

            return false;
        }
        else if(*layer->ptr_number_outputs != *tmp_ptr_mirror_layer->ptr_number_outputs)
        {
            ERR(L"The layer size (%zu) differ from the mirror layer size (%zu).",
                                     *layer->ptr_number_outputs,
                                     *tmp_ptr_mirror_layer->ptr_number_outputs);

            return false;
        }

        for(; next_layer_it != next_layer_end; ++next_layer_it)
        {
            if(valid_layer_fn(next_layer_it) == false)
            {
                ERR(L"An error has been triggered from the \"valid_layer_fn(ptr)\" function.",);

                return false;
            }
            else if(tmp_ptr_previous_layer_it->type_layer != next_layer_it->type_layer)
            {
                ERR(L"The previous connected layer type (%d | %ls) differ from the next connected layer type (%d | %ls).",
                                         tmp_ptr_previous_layer_it->type_layer,
                                         LAYER_NAME[tmp_ptr_previous_layer_it->type_layer].c_str(),
                                         next_layer_it->type_layer,
                                         LAYER_NAME[next_layer_it->type_layer].c_str());

                return false;
            }
            else if(*tmp_ptr_previous_layer_it->ptr_number_outputs != *next_layer_it->ptr_number_outputs)
            {
                ERR(L"The previous connected layer size (%zu) differ from the next connected layer size (%zu).",
                                         *tmp_ptr_previous_layer_it->ptr_number_outputs,
                                         *next_layer_it->ptr_number_outputs);

                return false;
            }
        }

        if(this->Set__Tied_Parameter(tmp_ptr_mirror_layer,
                                                   use_tied_parameter_received,
                                                   false))
        {
            ERR(L"An error has been triggered from the \"Set__Tied_Parameter(ptr, %ls, false)\" function.",
                                     use_tied_parameter_received ? "true" : "false");

            return false;
        }
    }
    // |END| Mirror. |END|
    
    if(layer->use_tied_parameter == false && use_tied_parameter_received)
    {
        ++this->total_tied_parameter_layers;

        if(transpose_received) { this->Tied__Transpose(layer); }
    }
    else if(layer->use_tied_parameter && use_tied_parameter_received == false) { --this->total_tied_parameter_layers; }

    layer->use_tied_parameter = use_tied_parameter_received;
    
    return true;
}

void Model::Tied__Transpose(void)
{
    Layer const *const tmp_ptr_end_layer(this->ptr_array_layers + (this->total_layers - 3_UZ) / 2_UZ + 1_UZ);
    Layer *layer_it(this->ptr_array_layers + 1);
    
    for(; layer_it != tmp_ptr_end_layer; ++layer_it)
    {
        if(layer_it->Use__Tied_Parameter())
        {
            this->Tied__Transpose(layer_it);
        }
    }
}

void Model::Tied__Transpose(Layer *const layer)
{
    this->Tied__Transpose__Weight(layer);

    if(layer->Use__Normalization()) { this->Tied__Transpose__Normalization(layer); }
}
}