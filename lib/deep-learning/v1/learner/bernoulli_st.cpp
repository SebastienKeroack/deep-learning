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
#include "deep-learning/v1/mem/reallocate.hpp"

#include <omp.h>

namespace DL::v1 {
void Model::Dropout_Bernoulli(void)
{
    if(this->use_mp && this->is_mp_initialized)
    {
        #pragma omp parallel
        this->Dropout_Bernoulli__OpenMP();
    }
    else
    { this->Dropout_Bernoulli__Loop(); }
}

void Model::Dropout_Bernoulli__Loop(void)
{
    size_t tmp_number_outputs;

    Layer const *const last_layer(this->ptr_last_layer);
    Layer *layer_it(this->ptr_array_layers);

    // Input layer.
    if(layer_it->type_dropout == LAYER_DROPOUT::BERNOULLI) { this->Dropout_Bernoulli__Layer__Loop(this->n_inp, layer_it); }

    for(++layer_it; layer_it != last_layer; ++layer_it)
    {
        if(layer_it->type_dropout == LAYER_DROPOUT::BERNOULLI)
        {
            switch(layer_it->type_layer)
            {
                case LAYER::FULLY_CONNECTED:
                case LAYER::FULLY_CONNECTED_RECURRENT: tmp_number_outputs = static_cast<size_t>(layer_it->ptr_last_AF_unit - layer_it->ptr_array_AF_units); break;
                case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT: tmp_number_outputs = static_cast<size_t>(layer_it->ptr_last_AF_Ind_recurrent_unit - layer_it->ptr_array_AF_Ind_recurrent_units); break;
                default:
                  ERR(L"Layer type (%d | %ls) is not managed in",
                      layer_it->type_layer,
                      LAYER_NAME[layer_it->type_layer].c_str());
                  return;
            }

            this->Dropout_Bernoulli__Layer__Loop(tmp_number_outputs, layer_it);
        }
    }
}

void Model::Dropout_Bernoulli__Layer__Loop(size_t const number_outputs_received, Layer *const layer_it)
{
    real const retained_probability(layer_it->dropout_values[0]);

    if(retained_probability != 0_r)
    {
        size_t tmp_unit_index,
                  tmp_time_step_index,
                  tmp_timed_mask_index;
        
        this->bernoulli->probability(retained_probability);
        
        for(tmp_time_step_index = 0_UZ; tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
        {
            tmp_timed_mask_index = tmp_time_step_index * number_outputs_received;

            for(tmp_unit_index = 0_UZ; tmp_unit_index != number_outputs_received; ++tmp_unit_index)
            {
                if((*this->bernoulli)()) // Keep unit.
                { layer_it->ptr_array__mask__dropout__bernoulli[tmp_timed_mask_index + tmp_unit_index] = true; }
                else // Drop unit.
                { layer_it->ptr_array__mask__dropout__bernoulli[tmp_timed_mask_index + tmp_unit_index] = false; }
            }
        }
    }
    else {
        Mem::fill(layer_it->ptr_array__mask__dropout__bernoulli,
                  layer_it->ptr_array__mask__dropout__bernoulli +
                      number_outputs_received * this->seq_w,
                  false);
    }
}
}
