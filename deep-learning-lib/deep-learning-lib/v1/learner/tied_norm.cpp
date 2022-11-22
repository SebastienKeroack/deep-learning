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

#include <omp.h>

namespace DL::v1 {
void Model::Tied__Transpose__Normalization(Layer *const layer)
{
    Layer *const mirror(this->ptr_last_layer - static_cast<size_t>(layer - this->ptr_array_layers) - 1);
    
    if(layer != mirror)
    {
        switch(layer->type_normalization)
        {
            case LAYER_NORM::BATCH_NORMALIZATION:
            case LAYER_NORM::BATCH_RENORMALIZATION: this->Tied__Transpose__Normalization__Batch_Normalization(layer, mirror); break;
            default:
                ERR(L"Layer normalization (%u | %ls) is not managed in",
                                            layer->type_normalization,
                                            LAYER_NORM_NAME[layer->type_normalization].c_str());
                    break;
        }
    }
}

void Model::Tied__Transpose__Normalization__Batch_Normalization(Layer const *const layer, Layer const *const mirror)
{
    Normalized_batch_unit const *const tmp_ptr_encoded_layer_it_first_normalized_batch_unit(&layer->ptr_array_normalized_units->normalized_batch_units),
                                                        *const tmp_ptr_mirror_layer_it_first_normalized_batch_unit(&mirror->ptr_array_normalized_units->normalized_batch_units);
    
    size_t const tmp_number_units(static_cast<size_t>(layer->ptr_last_normalized_unit - layer->ptr_array_normalized_units));
    
    VARCOPY(tmp_ptr_mirror_layer_it_first_normalized_batch_unit->ptr_scale,
                   tmp_ptr_encoded_layer_it_first_normalized_batch_unit->ptr_scale,
                   tmp_number_units * sizeof(var));
    
    VARCOPY(tmp_ptr_mirror_layer_it_first_normalized_batch_unit->ptr_shift,
                   tmp_ptr_encoded_layer_it_first_normalized_batch_unit->ptr_shift,
                   tmp_number_units * sizeof(var));
}
}
