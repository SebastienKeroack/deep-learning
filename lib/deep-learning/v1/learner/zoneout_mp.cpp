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

#include <omp.h>

namespace DL::v1 {
void Model::Dropout_Zoneout__OpenMP(void)
{
    Layer const *const last_layer(this->ptr_last_layer - 1);
    Layer *layer_it(this->ptr_array_layers + 1);
    
    for(; layer_it != last_layer; ++layer_it)
    {
        if(layer_it->type_dropout == LAYER_DROPOUT::ZONEOUT)
        {
            switch(layer_it->type_layer)
            {
                case LAYER::LSTM: this->Dropout_Zoneout__Block_Units__OpenMP(layer_it); break;
                default:
                    ERR(L"Layer type (%d | %ls) is not managed in",
                                             layer_it->type_layer,
                                             LAYER_NAME[layer_it->type_layer].c_str());
                        break;
            }
        }
    }
}

void Model::Dropout_Zoneout__Block_Units__OpenMP(Layer *const layer_it)
{
    int const tmp_number_recurrent_depth__int(static_cast<int>(this->seq_w));
    int tmp_time_step__int,
        tmp_thread_index__int;

    size_t const tmp_number_cell_units(static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units));
    size_t tmp_timed_mask_index;

    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;
    
    this->ptr_array_Class_Generator_Bernoulli_Zoneout_State[omp_get_thread_num()].probability(layer_it->dropout_values[0]);
    this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden[omp_get_thread_num()].probability(layer_it->dropout_values[1]);
    
    #pragma omp for schedule(static)
    for(tmp_time_step__int = 0; tmp_time_step__int < tmp_number_recurrent_depth__int; ++tmp_time_step__int)
    {
        tmp_thread_index__int = omp_get_thread_num();

        tmp_timed_mask_index = static_cast<size_t>(tmp_time_step__int) * tmp_number_cell_units;

        for(tmp_ptr_last_cell_unit = layer_it->ptr_last_cell_unit,
            tmp_ptr_cell_unit_it = layer_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
        {
            if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_State[tmp_thread_index__int]()) // Zoneout cell state.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state[tmp_timed_mask_index] = false; }
            else // Keep cell state.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state[tmp_timed_mask_index] = true; }

            if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden[tmp_thread_index__int]()) // Zoneout cell output.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output[tmp_timed_mask_index] = false; }
            else // Keep cell output.
            { tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output[tmp_timed_mask_index] = true; }
        }
    }
}
}
