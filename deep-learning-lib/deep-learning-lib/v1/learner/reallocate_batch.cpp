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

namespace DL::v1 {
bool Model::Reallocate__Batch(size_t const batch_size)
{
    if(this->Reallocate__Batch__Basic_Unit(batch_size) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Batch__Basic_Unit(%zu)\" function.",
                                 batch_size);

        return false;
    }
    else if(this->Reallocate__Batch__Basic_Indice_Unit(batch_size) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Batch__Basic_Indice_Unit(%zu)\" function.",
                                 batch_size);

        return false;
    }
    else if(this->Reallocate__Batch__Neuron_Unit(batch_size) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Batch__Neuron_Unit(%zu)\" function.",
                                 batch_size);

        return false;
    }
    else if(this->Reallocate__Batch__AF_Unit(batch_size) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Batch__AF_Unit(%zu)\" function.",
                                 batch_size);

        return false;
    }
    else if(this->Reallocate__Batch__AF_Ind_Recurrent_Unit(batch_size) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Batch__AF_Ind_Recurrent_Unit(%zu)\" function.",
                                 batch_size);

        return false;
    }
    else if(this->Reallocate__Normalized_Unit__Batch_Normalization(batch_size) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Normalized_Unit__Batch_Normalization(%zu)\" function.",
                                 batch_size);

        return false;
    }
    else if(this->Reallocate__Batch__LSTM(batch_size) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Batch__LSTM(%zu)\" function.",
                                 batch_size);

        return false;
    }
    else if(this->Use__Dropout__ShakeDrop() && this->Reallocate__Batch__Dropout__ShakeDrop(batch_size) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Batch__Dropout__ShakeDrop(%zu)\" function.",
                                 batch_size);

        return false;
    }
    
    this->Order__Layers__Output();

    return true;
}

bool Model::Reallocate__Batch__Basic_Unit(size_t const batch_size)
{
    if(this->total_basic_units_allocated != 0_UZ)
    {
        size_t tmp_number_basic_units;

        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);

        Basic_unit const *tmp_ptr_last_basic_unit;
        Basic_unit *tmp_ptr_basic_unit_it;

        // Allocating basic unit(s) value.
        var *tmp_ptr_array_basic_units_values(Mem::reallocate(this->ptr_array_basic_units_values,
                                                                                                           batch_size * this->total_basic_units_allocated * this->seq_w,
                                                                                                           this->batch_size * this->total_basic_units_allocated * this->seq_w));
        this->ptr_array_basic_units_values = tmp_ptr_array_basic_units_values;
        // |END| Allocating basic unit(s) value. |END|

        // Allocating basic unit(s) error.
        real *tmp_ptr_array_basic_units_errors(Mem::reallocate(this->ptr_array_basic_units_errors,
                                                                                                          batch_size * this->total_basic_units_allocated * this->seq_w,
                                                                                                          this->batch_size * this->total_basic_units_allocated * this->seq_w));
        this->ptr_array_basic_units_errors = tmp_ptr_array_basic_units_errors;
        // |END| Allocating basic unit(s) error. |END|

        for(; layer_it != last_layer; ++layer_it)
        {
            tmp_number_basic_units = static_cast<size_t>(layer_it->ptr_last_basic_unit - layer_it->ptr_array_basic_units);

            if(tmp_number_basic_units != 0_UZ)
            {
                for(tmp_ptr_last_basic_unit = layer_it->ptr_last_basic_unit,
                    tmp_ptr_basic_unit_it = layer_it->ptr_array_basic_units; tmp_ptr_basic_unit_it != tmp_ptr_last_basic_unit; ++tmp_ptr_basic_unit_it,
                                                                                                                                                                                           ++tmp_ptr_array_basic_units_values,
                                                                                                                                                                                           ++tmp_ptr_array_basic_units_errors)
                {
                    tmp_ptr_basic_unit_it->ptr_array_values = tmp_ptr_array_basic_units_values;
                    tmp_ptr_basic_unit_it->ptr_array_errors = tmp_ptr_array_basic_units_errors;
                }
                
                tmp_ptr_array_basic_units_values += (batch_size - 1_UZ) * tmp_number_basic_units * this->seq_w + tmp_number_basic_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_basic_units_errors += (batch_size - 1_UZ) * tmp_number_basic_units * this->seq_w + tmp_number_basic_units * (this->seq_w - 1_UZ);
            }
        }
    }

    return true;
}

bool Model::Reallocate__Batch__Basic_Indice_Unit(size_t const batch_size)
{
    if(this->total_basic_indice_units_allocated != 0_UZ)
    {
        size_t tmp_number_basic_indice_units;

        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);

        Basic_indice_unit const *tmp_ptr_last_basic_indice_unit;
        Basic_indice_unit *tmp_ptr_basic_indice_unit_it;
        
        // Allocating basic unit(s) indice.
        size_t *tmp_ptr_array_basic_indice_units_indices(Mem::reallocate<size_t, false>(this->ptr_array_basic_indice_units_indices,
                                                                                                                                 batch_size * this->total_basic_indice_units_allocated * this->seq_w,
                                                                                                                                 this->batch_size * this->total_basic_indice_units_allocated * this->seq_w));
        if(tmp_ptr_array_basic_indice_units_indices == nullptr)
        {
            ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function.",
                                     sizeof(size_t),
                                     batch_size * this->total_basic_indice_units_allocated * this->seq_w,
                                     this->batch_size * this->total_basic_indice_units_allocated * this->seq_w);

            return false;
        }
        this->ptr_array_basic_indice_units_indices = tmp_ptr_array_basic_indice_units_indices;
        // |END| Allocating basic unit(s) indice. |END|

        // Allocating basic unit(s) value.
        var *tmp_ptr_array_basic_indice_units_values(Mem::reallocate(this->ptr_array_basic_indice_units_values,
                                                                                                                     batch_size * this->total_basic_indice_units_allocated * this->seq_w,
                                                                                                                     this->batch_size * this->total_basic_indice_units_allocated * this->seq_w));
        this->ptr_array_basic_indice_units_values = tmp_ptr_array_basic_indice_units_values;
        // |END| Allocating basic unit(s) value. |END|

        // Allocating basic unit(s) error.
        real *tmp_ptr_array_basic_indice_units_errors(Mem::reallocate(
            this->ptr_array_basic_indice_units_errors,
                                                                                                                    batch_size * this->total_basic_indice_units_allocated * this->seq_w,
                                                                                                                    this->batch_size * this->total_basic_indice_units_allocated * this->seq_w));
        this->ptr_array_basic_indice_units_errors = tmp_ptr_array_basic_indice_units_errors;
        // |END| Allocating basic unit(s) error. |END|

        for(; layer_it != last_layer; ++layer_it)
        {
            tmp_number_basic_indice_units = static_cast<size_t>(layer_it->ptr_last_basic_indice_unit - layer_it->ptr_array_basic_indice_units);

            if(tmp_number_basic_indice_units != 0_UZ)
            {
                for(tmp_ptr_last_basic_indice_unit = layer_it->ptr_last_basic_indice_unit,
                    tmp_ptr_basic_indice_unit_it = layer_it->ptr_array_basic_indice_units; tmp_ptr_basic_indice_unit_it != tmp_ptr_last_basic_indice_unit; ++tmp_ptr_basic_indice_unit_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_basic_indice_units_indices,
                                                                                                                                                                                                                                   ++tmp_ptr_array_basic_indice_units_values,
                                                                                                                                                                                                                                   ++tmp_ptr_array_basic_indice_units_errors)
                {
                    tmp_ptr_basic_indice_unit_it->ptr_array_indices = tmp_ptr_array_basic_indice_units_indices;

                    tmp_ptr_basic_indice_unit_it->ptr_array_values = tmp_ptr_array_basic_indice_units_values;
                    tmp_ptr_basic_indice_unit_it->ptr_array_errors = tmp_ptr_array_basic_indice_units_errors;
                }
                
                tmp_ptr_array_basic_indice_units_indices += (batch_size - 1_UZ) * tmp_number_basic_indice_units * this->seq_w + tmp_number_basic_indice_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_basic_indice_units_values += (batch_size - 1_UZ) * tmp_number_basic_indice_units * this->seq_w + tmp_number_basic_indice_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_basic_indice_units_errors += (batch_size - 1_UZ) * tmp_number_basic_indice_units * this->seq_w + tmp_number_basic_indice_units * (this->seq_w - 1_UZ);
            }
        }
    }

    return true;
}

bool Model::Reallocate__Batch__Neuron_Unit(size_t const batch_size)
{
    if(this->total_neuron_units_allocated != 0_UZ)
    {
        size_t tmp_number_neuron_units;

        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);

        Neuron_unit const *tmp_ptr_last_neuron_unit;
        Neuron_unit *tmp_ptr_neuron_unit_it;

        // Allocating neuron unit(s) summation(s).
        var *tmp_ptr_array_neuron_units_summations(Mem::reallocate(this->ptr_array_neuron_units_summations,
                                                                                                                      batch_size * this->total_neuron_units_allocated * this->seq_w,
                                                                                                                      this->batch_size * this->total_neuron_units_allocated * this->seq_w));
        this->ptr_array_neuron_units_summations = tmp_ptr_array_neuron_units_summations;
        // |END| Allocating neuron unit(s) summation(s). |END|

        // Allocating neuron unit(s) error(s).
        real *tmp_ptr_array_neuron_units_errors(Mem::reallocate(this->ptr_array_neuron_units_errors,
                                                                                                            batch_size * this->total_neuron_units_allocated * this->seq_w,
                                                                                                            this->batch_size * this->total_neuron_units_allocated * this->seq_w));
        this->ptr_array_neuron_units_errors = tmp_ptr_array_neuron_units_errors;
        // |END| Allocating neuron unit(s) error(s). |END|
        
        for(; layer_it != last_layer; ++layer_it)
        {
            tmp_number_neuron_units = static_cast<size_t>(layer_it->ptr_last_neuron_unit - layer_it->ptr_array_neuron_units);

            if(tmp_number_neuron_units != 0_UZ)
            {
                for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit,
                    tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                                                  ++tmp_ptr_array_neuron_units_summations,
                                                                                                                                                                                                  ++tmp_ptr_array_neuron_units_errors)
                {
                    tmp_ptr_neuron_unit_it->ptr_array_summations = tmp_ptr_array_neuron_units_summations;
                    tmp_ptr_neuron_unit_it->ptr_array_errors = tmp_ptr_array_neuron_units_errors;
                }
                
                tmp_ptr_array_neuron_units_summations += (batch_size - 1_UZ) * tmp_number_neuron_units * this->seq_w + tmp_number_neuron_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_neuron_units_errors += (batch_size - 1_UZ) * tmp_number_neuron_units * this->seq_w + tmp_number_neuron_units * (this->seq_w - 1_UZ);
            }
        }
    }

    return true;
}

bool Model::Reallocate__Batch__AF_Unit(size_t const batch_size)
{
    if(this->total_AF_units_allocated != 0_UZ)
    {
        size_t tmp_number_AF_units;

        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);

        AF_unit const *tmp_ptr_last_AF_unit;
        AF_unit *tmp_ptr_AF_unit_it;

        // Allocating AF unit(s) value(s).
        var *tmp_ptr_array_AF_units_values(Mem::reallocate(this->ptr_array_AF_units_values,
                                                                                                      batch_size * this->total_AF_units_allocated * this->seq_w,
                                                                                                      this->batch_size * this->total_AF_units_allocated * this->seq_w));
        this->ptr_array_AF_units_values = tmp_ptr_array_AF_units_values;
        // |END| Allocating AF unit(s) value(s). |END|

        // Allocating AF unit(s) error(s).
        real *tmp_ptr_array_AF_units_errors(Mem::reallocate(
            this->ptr_array_AF_units_errors,
                                                                                                     batch_size * this->total_AF_units_allocated * this->seq_w,
                                                                                                     this->batch_size * this->total_AF_units_allocated * this->seq_w));
        this->ptr_array_AF_units_errors = tmp_ptr_array_AF_units_errors;
        // |END| Allocating AF unit(s) error(s). |END|
        
        for(; layer_it != last_layer; ++layer_it)
        {
            tmp_number_AF_units = static_cast<size_t>(layer_it->ptr_last_AF_unit - layer_it->ptr_array_AF_units);

            if(tmp_number_AF_units != 0_UZ)
            {
                for(tmp_ptr_last_AF_unit = layer_it->ptr_last_AF_unit,
                    tmp_ptr_AF_unit_it = layer_it->ptr_array_AF_units; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it,
                                                                                                                                                                      ++tmp_ptr_array_AF_units_values,
                                                                                                                                                                      ++tmp_ptr_array_AF_units_errors)
                {
                    tmp_ptr_AF_unit_it->ptr_array_values = tmp_ptr_array_AF_units_values;
                    tmp_ptr_AF_unit_it->ptr_array_errors = tmp_ptr_array_AF_units_errors;
                }
                
                tmp_ptr_array_AF_units_values += (batch_size - 1_UZ) * tmp_number_AF_units * this->seq_w + tmp_number_AF_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_AF_units_errors += (batch_size - 1_UZ) * tmp_number_AF_units * this->seq_w + tmp_number_AF_units * (this->seq_w - 1_UZ);
            }
        }
    }

    return true;
}

bool Model::Reallocate__Batch__AF_Ind_Recurrent_Unit(size_t const batch_size)
{
    if(this->total_AF_Ind_recurrent_units_allocated != 0_UZ)
    {
        size_t tmp_number_AF_Ind_recurrent_units;

        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);

        AF_Ind_recurrent_unit const *tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit;
        AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it;

        // Allocating af_ind unit(s) value(s).
        var *tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs(Mem::reallocate(this->ptr_array_AF_Ind_recurrent_units_pre_AFs,
                                                                                                            batch_size * this->total_AF_Ind_recurrent_units_allocated * this->seq_w,
                                                                                                            this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->seq_w));
        this->ptr_array_AF_Ind_recurrent_units_pre_AFs = tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs;
        // |END| Allocating af_ind unit(s) value(s). |END|
        
        // Allocating af_ind unit(s) value(s).
        var *tmp_ptr_array_AF_Ind_recurrent_units_AFs(Mem::reallocate(this->ptr_array_AF_Ind_recurrent_units_AFs,
                                                                                                            batch_size * this->total_AF_Ind_recurrent_units_allocated * this->seq_w,
                                                                                                            this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->seq_w));
        this->ptr_array_AF_Ind_recurrent_units_AFs = tmp_ptr_array_AF_Ind_recurrent_units_AFs;
        // |END| Allocating af_ind unit(s) value(s). |END|

        // Allocating af_ind unit(s) error(s).
        real *tmp_ptr_array_AF_Ind_recurrent_units_errors(Mem::reallocate(this->ptr_array_AF_Ind_recurrent_units_errors,
                                                                                                           batch_size * this->total_AF_Ind_recurrent_units_allocated * this->seq_w,
                                                                                                           this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->seq_w));
        this->ptr_array_AF_Ind_recurrent_units_errors = tmp_ptr_array_AF_Ind_recurrent_units_errors;
        // |END| Allocating af_ind unit(s) error(s). |END|
        
        // Allocating af_ind unit(s) dAF_Ind_Recurrent(s).
        real *tmp_ptr_array_AF_Ind_recurrent_units_dAFs(Mem::reallocate(
            this->ptr_array_AF_Ind_recurrent_units_dAFs,
                                                                                                                 batch_size * this->total_AF_Ind_recurrent_units_allocated * this->seq_w,
                                                                                                                 this->batch_size * this->total_AF_Ind_recurrent_units_allocated * this->seq_w));
        this->ptr_array_AF_Ind_recurrent_units_dAFs = tmp_ptr_array_AF_Ind_recurrent_units_dAFs;
        // |END| Allocating af_ind unit(s) dAF_Ind_Recurrent(s). |END|
        
        for(; layer_it != last_layer; ++layer_it)
        {
            tmp_number_AF_Ind_recurrent_units = static_cast<size_t>(layer_it->ptr_last_AF_Ind_recurrent_unit - layer_it->ptr_array_AF_Ind_recurrent_units);

            if(tmp_number_AF_Ind_recurrent_units != 0_UZ)
            {
                for(tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit = layer_it->ptr_last_AF_Ind_recurrent_unit,
                    tmp_ptr_AF_Ind_recurrent_unit_it = layer_it->ptr_array_AF_Ind_recurrent_units; tmp_ptr_AF_Ind_recurrent_unit_it != tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit; ++tmp_ptr_AF_Ind_recurrent_unit_it,
                                                                                                                                                                                              ++tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs,
                                                                                                                                                                                              ++tmp_ptr_array_AF_Ind_recurrent_units_AFs,
                                                                                                                                                                                              ++tmp_ptr_array_AF_Ind_recurrent_units_errors,
                                                                                                                                                                                              ++tmp_ptr_array_AF_Ind_recurrent_units_dAFs)
                {
                    tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_pre_AFs = tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs;
                    tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_AFs = tmp_ptr_array_AF_Ind_recurrent_units_AFs;
                    tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_errors = tmp_ptr_array_AF_Ind_recurrent_units_errors;
                    tmp_ptr_AF_Ind_recurrent_unit_it->ptr_array_dAFs = tmp_ptr_array_AF_Ind_recurrent_units_dAFs;
                }
                
                tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs += (batch_size - 1_UZ) * tmp_number_AF_Ind_recurrent_units * this->seq_w + tmp_number_AF_Ind_recurrent_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_AF_Ind_recurrent_units_AFs += (batch_size - 1_UZ) * tmp_number_AF_Ind_recurrent_units * this->seq_w + tmp_number_AF_Ind_recurrent_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_AF_Ind_recurrent_units_errors += (batch_size - 1_UZ) * tmp_number_AF_Ind_recurrent_units * this->seq_w + tmp_number_AF_Ind_recurrent_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_AF_Ind_recurrent_units_dAFs += (batch_size - 1_UZ) * tmp_number_AF_Ind_recurrent_units * this->seq_w + tmp_number_AF_Ind_recurrent_units * (this->seq_w - 1_UZ);
            }
        }
    }

    return true;
}

bool Model::Reallocate__Normalized_Unit__Batch_Normalization(size_t const batch_size)
{
    if(this->Use__Normalization()
       &&
       this->ptr_array_normalized_batch_units_values_hats != nullptr
       &&
       this->ptr_array_normalized_batch_units_values_normalizes != nullptr
       &&
       this->ptr_array_normalized_batch_units_errors != nullptr)
    {
        size_t tmp_number_units,
                  tmp_index;

        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);
        
        BlockUnit const *tmp_ptr_last_block_unit;
        BlockUnit *tmp_ptr_block_unit_it;
        
        CellUnit const *tmp_ptr_last_cell_unit;
        CellUnit *tmp_ptr_cell_unit_it;
        
        union Normalized_unit const *tmp_ptr_last_normalized_unit;
        union Normalized_unit *tmp_ptr_normalized_unit_it;
        
        // Allocating normalized unit(s) value(s) hat.
        var *tmp_ptr_array_normalized_units_values_hat(Mem::reallocate(this->ptr_array_normalized_batch_units_values_hats,
                                                                                                                         batch_size * this->total_normalized_units_allocated * this->seq_w,
                                                                                                                         this->batch_size * this->total_normalized_units_allocated * this->seq_w));
        this->ptr_array_normalized_batch_units_values_hats = tmp_ptr_array_normalized_units_values_hat;
        // |END| Allocating normalized unit(s) value(s) hat. |END|

        // Allocating normalized unit(s) value(s) normalize.
        var *tmp_ptr_array_normalized_units_values_normalize(Mem::reallocate(this->ptr_array_normalized_batch_units_values_normalizes,
                                                                                                                                   batch_size * this->total_normalized_units_allocated * this->seq_w,
                                                                                                                                   this->batch_size * this->total_normalized_units_allocated * this->seq_w));
        this->ptr_array_normalized_batch_units_values_normalizes = tmp_ptr_array_normalized_units_values_normalize;
        // |END| Allocating normalized unit(s) value(s) normalize. |END|
        
        // Allocating normalized unit(s) error(s).
        real *tmp_ptr_array_normalized_units_errors(Mem::reallocate(this->ptr_array_normalized_batch_units_errors,
                                                                                                                  batch_size * this->total_normalized_units_allocated * this->seq_w,
                                                                                                                  this->batch_size * this->total_normalized_units_allocated * this->seq_w));
        this->ptr_array_normalized_batch_units_errors = tmp_ptr_array_normalized_units_errors;
        // |END| Allocating normalized unit(s) error(s). |END|

        for(; layer_it != last_layer; ++layer_it)
        {
            if((tmp_number_units = static_cast<size_t>(layer_it->ptr_last_normalized_unit - layer_it->ptr_array_normalized_units)) != 0_UZ)
            {
                switch(layer_it->type_layer)
                {
                    case LAYER::FULLY_CONNECTED:
                    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case LAYER::FULLY_CONNECTED_RECURRENT:
                    case LAYER::RESIDUAL:
                        for(tmp_ptr_last_normalized_unit = layer_it->ptr_last_normalized_unit,
                            tmp_ptr_normalized_unit_it = layer_it->ptr_array_normalized_units; tmp_ptr_normalized_unit_it != tmp_ptr_last_normalized_unit; ++tmp_ptr_normalized_unit_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_values_hat,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_values_normalize,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_errors)
                        {
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_values_hats = tmp_ptr_array_normalized_units_values_hat;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_values_normalizes = tmp_ptr_array_normalized_units_values_normalize;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors;
                        }
                        
                        tmp_ptr_array_normalized_units_values_hat += (batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                        tmp_ptr_array_normalized_units_values_normalize += (batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                        tmp_ptr_array_normalized_units_errors += (batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            break;
                    case LAYER::LSTM:
                        if(static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units) != 0_UZ)
                        {
                            // [0]: Block input, input.
                            // [1]: Block input, recurrent.
                            // [2]: Cell state activate.

                            tmp_ptr_last_cell_unit = layer_it->ptr_last_cell_unit;
                        
                            tmp_number_units = static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units);

                            for(tmp_index = 0_UZ; tmp_index != 3_UZ; ++tmp_index)
                            {
                                for(tmp_ptr_cell_unit_it = layer_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it)
                                {
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_values_hats = tmp_ptr_array_normalized_units_values_hat++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_values_normalizes = tmp_ptr_array_normalized_units_values_normalize++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors++;
                                }
                            
                                tmp_ptr_array_normalized_units_values_hat += (batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_values_normalize += (batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_errors += (batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            }
                        
                            // [3]: Input gate, input.
                            // [4]: Input gate, recurrent.
                            // [5]: Forget gate, input.
                            // [6]: Forget gate, recurrent.
                            // [7]: Output gate, input.
                            // [8]: Output gate, recurrent.

                            tmp_ptr_last_block_unit = layer_it->ptr_last_block_unit;
                        
                            tmp_number_units = static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units);

                            for(tmp_index = 0_UZ; tmp_index != 6_UZ; ++tmp_index)
                            {
                                for(tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
                                {
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_values_hats = tmp_ptr_array_normalized_units_values_hat++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_values_normalizes = tmp_ptr_array_normalized_units_values_normalize++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors++;
                                }
                            
                                tmp_ptr_array_normalized_units_values_hat += (batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_values_normalize += (batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_errors += (batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            }
                        } 
                            break;
                    default:
                        ERR(L"Type layer (%u | %ls) is not managed in the switch.",
                                                 layer_it->type_layer,
                                                 LAYER_NAME[layer_it->type_layer].c_str());
                            return false;
                }
            }
        }
    }

    return true;
}

bool Model::Reallocate__Batch__LSTM(size_t const batch_size)
{
    if(this->total_block_units_allocated * this->total_cell_units_allocated != 0_UZ)
    {
        size_t tmp_number_block_units,
                  tmp_number_cell_units;
        
        Layer const *const last_layer(this->ptr_last_layer - 1); // Subtract output layer.
        Layer *layer_it;

        BlockUnit const *tmp_ptr_last_block_unit;
        BlockUnit *tmp_ptr_block_unit_it;

        CellUnit const *tmp_ptr_last_cell_unit;
        CellUnit *tmp_ptr_cell_unit_it;
        
        // Allocating summation cell input.
        var *tmp_ptr_array_summation_cells_inputs(Mem::reallocate(this->ptr_array_cells_summations_cells_inputs,
                                                                                                                   batch_size * this->total_cell_units * this->seq_w,
                                                                                                                   this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_summations_cells_inputs = tmp_ptr_array_summation_cells_inputs;
        // |END| Allocating summation cell input. |END|
        
        // Allocating summation input cell input.
        var *tmp_ptr_array_summation_input_cells_inputs(Mem::reallocate(this->ptr_array_cells_summations_input_cells_inputs,
                                                                                                                           batch_size * this->total_cell_units * this->seq_w,
                                                                                                                           this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_summations_input_cells_inputs = tmp_ptr_array_summation_input_cells_inputs;
        // |END| Allocating summation input cell input. |END|
        
        // Allocating summation recurrent cell input.
        var *tmp_ptr_array_summation_recurrent_cells_inputs(Mem::reallocate(this->ptr_array_cells_summations_recurrent_cells_inputs,
                                                                                                                                 batch_size * this->total_cell_units * this->seq_w,
                                                                                                                                 this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_summations_recurrent_cells_inputs = tmp_ptr_array_summation_recurrent_cells_inputs;
        // |END| Allocating summation recurrent cell input. |END|
        
        // Allocating LSTM summation input gate.
        var *tmp_ptr_array_summation_inputs_gates(Mem::reallocate(this->ptr_array_blocks_summations_inputs_gates,
                                                                                                                    batch_size * this->total_block_units * this->seq_w,
                                                                                                                    this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_summations_inputs_gates = tmp_ptr_array_summation_inputs_gates;
        // |END| Allocating LSTM summation input gate. |END|
        
        // Allocating LSTM summation input input gate.
        var *tmp_ptr_array_summation_input_inputs_gates(Mem::reallocate(this->ptr_array_blocks_summations_input_inputs_gates,
                                                                                                                             batch_size * this->total_block_units * this->seq_w,
                                                                                                                             this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_summations_input_inputs_gates = tmp_ptr_array_summation_input_inputs_gates;
        // |END| Allocating LSTM summation input input gate. |END|
        
        // Allocating LSTM summation recurrent input gate.
        var *tmp_ptr_array_summation_recurrent_inputs_gates(Mem::reallocate(this->ptr_array_blocks_summations_recurrent_inputs_gates,
                                                                                                                                   batch_size * this->total_block_units * this->seq_w,
                                                                                                                                   this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_summations_recurrent_inputs_gates = tmp_ptr_array_summation_recurrent_inputs_gates;
        // |END| Allocating LSTM summation recurrent input gate. |END|
        
        // Allocating LSTM summation forget gate.
        var *tmp_ptr_array_summation_forgets_gates(Mem::reallocate(this->ptr_array_blocks_summations_forgets_gates,
                                                                                                                     batch_size * this->total_block_units * this->seq_w,
                                                                                                                     this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_summations_forgets_gates = tmp_ptr_array_summation_forgets_gates;
        // |END| Allocating LSTM summation forget gate. |END|
        
        // Allocating LSTM summation input forget gate.
        var *tmp_ptr_array_summation_input_forgets_gates(Mem::reallocate(this->ptr_array_blocks_summations_input_forgets_gates,
                                                                                                                     batch_size * this->total_block_units * this->seq_w,
                                                                                                                     this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_summations_input_forgets_gates = tmp_ptr_array_summation_input_forgets_gates;
        // |END| Allocating LSTM summation input forget gate. |END|
        
        // Allocating LSTM summation recurrent forget gate.
        var *tmp_ptr_array_summation_recurrent_forgets_gates(Mem::reallocate(this->ptr_array_blocks_summations_recurrent_forgets_gates,
                                                                                                                     batch_size * this->total_block_units * this->seq_w,
                                                                                                                     this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_summations_recurrent_forgets_gates = tmp_ptr_array_summation_recurrent_forgets_gates;
        // |END| Allocating LSTM summation recurrent forget gate. |END|
        
        // Allocating LSTM summation outputs gate.
        var *tmp_ptr_array_summation_outputs_gates(Mem::reallocate(this->ptr_array_blocks_summations_outputs_gates,
                                                                                                                      batch_size * this->total_block_units * this->seq_w,
                                                                                                                      this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_summations_outputs_gates = tmp_ptr_array_summation_outputs_gates;
        // |END| Allocating LSTM summation outputs gate. |END|
        
        // Allocating LSTM summation input outputs gate.
        var *tmp_ptr_array_summation_input_outputs_gates(Mem::reallocate(this->ptr_array_blocks_summations_input_outputs_gates,
                                                                                                                      batch_size * this->total_block_units * this->seq_w,
                                                                                                                      this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_summations_input_outputs_gates = tmp_ptr_array_summation_input_outputs_gates;
        // |END| Allocating LSTM summation input outputs gate. |END|
        
        // Allocating LSTM summation recurrent outputs gate.
        var *tmp_ptr_array_summation_recurrent_outputs_gates(Mem::reallocate(this->ptr_array_blocks_summations_recurrent_outputs_gates,
                                                                                                                      batch_size * this->total_block_units * this->seq_w,
                                                                                                                      this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_summations_recurrent_outputs_gates = tmp_ptr_array_summation_recurrent_outputs_gates;
        // |END| Allocating LSTM summation recurrent outputs gate. |END|
        
        // Allocating cell input.
        var *tmp_ptr_array_cells_inputs(Mem::reallocate(this->ptr_array_cells_inputs,
                                                                                                 batch_size * this->total_cell_units * this->seq_w,
                                                                                                 this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_inputs = tmp_ptr_array_cells_inputs;
        // |END| Allocating cell input. |END|
        
        // Allocating LSTM cell state.
        var *tmp_ptr_array_cells_states(Mem::reallocate(this->ptr_array_cells_states,
                                                                                                 batch_size * this->total_cell_units * this->seq_w,
                                                                                                 this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_states = tmp_ptr_array_cells_states;
        // |END| Allocating LSTM cell state. |END|
        
        // Allocating LSTM cell state activate.
        var *tmp_ptr_array_cells_states_activates(Mem::reallocate(this->ptr_array_cells_states_activates,
                                                                                                 batch_size * this->total_cell_units * this->seq_w,
                                                                                                 this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_states_activates = tmp_ptr_array_cells_states_activates;
        // |END| Allocating LSTM cell state activate. |END|
        
        // Allocating LSTM cell outputs.
        var *tmp_ptr_array_cells_outputs(Mem::reallocate(this->ptr_array_cells_outputs,
                                                                                                   batch_size * this->total_cell_units * this->seq_w,
                                                                                                   this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_outputs = tmp_ptr_array_cells_outputs;
        // |END| Allocating LSTM cell outputs. |END|
        
        // Allocating LSTM input gate.
        var *tmp_ptr_array_inputs_gates(Mem::reallocate(this->ptr_array_blocks_inputs_gates,
                                                                                                  batch_size * this->total_block_units * this->seq_w,
                                                                                                  this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_inputs_gates = tmp_ptr_array_inputs_gates;
        // |END| Allocating LSTM input gate. |END|
        
        // Allocating LSTM forget gate.
        var *tmp_ptr_array_forgets_gates(Mem::reallocate(this->ptr_array_blocks_forgets_gates,
                                                                                                   batch_size * this->total_block_units * this->seq_w,
                                                                                                   this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_forgets_gates = tmp_ptr_array_forgets_gates;
        // |END| Allocating LSTM forget gate. |END|
        
        // Allocating LSTM outputs gate.
        var *tmp_ptr_array_outputs_gates(Mem::reallocate(this->ptr_array_blocks_outputs_gates,
                                                                                                    batch_size * this->total_block_units * this->seq_w,
                                                                                                    this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_outputs_gates = tmp_ptr_array_outputs_gates;
        // |END| Allocating LSTM outputs gate. |END|
        
        // Allocating LSTM delta cell inputs.
        real *tmp_ptr_array_delta_cells_inputs(Mem::reallocate(this->ptr_array_cells_delta_inputs,
                                                                                                          batch_size * this->total_cell_units * this->seq_w,
                                                                                                          this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_delta_inputs = tmp_ptr_array_delta_cells_inputs;
        // |END| Allocating LSTM delta cell inputs. |END|
        
        // Allocating LSTM delta cell input inputs.
        real *tmp_ptr_array_delta_cells_input_inputs(Mem::reallocate(this->ptr_array_cells_delta_input_inputs,
                                                                                                                  batch_size * this->total_cell_units * this->seq_w,
                                                                                                                  this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_delta_input_inputs = tmp_ptr_array_delta_cells_input_inputs;
        // |END| Allocating LSTM delta cell input inputs. |END|
        
        // Allocating LSTM delta cell recurrent inputs.
        real *tmp_ptr_array_delta_cells_recurrent_inputs(Mem::reallocate(this->ptr_array_cells_delta_recurrent_inputs,
                                                                                                                      batch_size * this->total_cell_units * this->seq_w,
                                                                                                                      this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_delta_recurrent_inputs = tmp_ptr_array_delta_cells_recurrent_inputs;
        // |END| Allocating LSTM delta cell recurrent inputs. |END|
        
        // Allocating LSTM delta cell state.
        real *tmp_ptr_array_delta_cells_states(Mem::reallocate(this->ptr_array_cells_delta_states,
                                                                                                          batch_size * this->total_cell_units * this->seq_w,
                                                                                                          this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_delta_states = tmp_ptr_array_delta_cells_states;
        // |END| Allocating LSTM delta cell state. |END|
        
        // Allocating LSTM delta cell outputs.
        real *tmp_ptr_array_delta_cells_outputs(Mem::reallocate(this->ptr_array_cells_delta_outputs,
                                                                                                            batch_size * this->total_cell_units * this->seq_w,
                                                                                                            this->batch_size * this->total_cell_units * this->seq_w));
        this->ptr_array_cells_delta_outputs = tmp_ptr_array_delta_cells_outputs;
        // |END| Allocating LSTM delta cell outputs. |END|
        
        // Allocating LSTM delta input gate.
        real *tmp_ptr_array_delta_inputs_gates(Mem::reallocate(this->ptr_array_blocks_delta_inputs_gates,
                                                                                                           batch_size * this->total_block_units * this->seq_w,
                                                                                                           this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_delta_inputs_gates = tmp_ptr_array_delta_inputs_gates;
        // |END| Allocating LSTM delta input gate. |END|
        
        // Allocating LSTM delta input input gate.
        real *tmp_ptr_array_delta_input_inputs_gates(Mem::reallocate(this->ptr_array_blocks_delta_input_inputs_gates,
                                                                                                                    batch_size * this->total_block_units * this->seq_w,
                                                                                                                    this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_delta_input_inputs_gates = tmp_ptr_array_delta_input_inputs_gates;
        // |END| Allocating LSTM delta input input gate. |END|
        
        // Allocating LSTM delta recurrent input gate.
        real *tmp_ptr_array_delta_recurrent_inputs_gates(Mem::reallocate(this->ptr_array_blocks_delta_recurrent_inputs_gates,
                                                                                                                        batch_size * this->total_block_units * this->seq_w,
                                                                                                                        this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_delta_recurrent_inputs_gates = tmp_ptr_array_delta_recurrent_inputs_gates;
        // |END| Allocating LSTM delta recurrent input gate. |END|
        
        // Allocating LSTM delta forget gate.
        real *tmp_ptr_array_delta_forgets_gates(Mem::reallocate(this->ptr_array_blocks_delta_forgets_gates,
                                                                                                            batch_size * this->total_block_units * this->seq_w,
                                                                                                            this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_delta_forgets_gates = tmp_ptr_array_delta_forgets_gates;
        // |END| Allocating LSTM delta forget gate. |END|
        
        // Allocating LSTM delta input forget gate.
        real *tmp_ptr_array_delta_input_forgets_gates(Mem::reallocate(this->ptr_array_blocks_delta_input_forgets_gates,
                                                                                                                    batch_size * this->total_block_units * this->seq_w,
                                                                                                                    this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_delta_input_forgets_gates = tmp_ptr_array_delta_input_forgets_gates;
        // |END| Allocating LSTM delta input forget gate. |END|
        
        // Allocating LSTM delta recurrent forget gate.
        real *tmp_ptr_array_delta_recurrent_forgets_gates(Mem::reallocate(this->ptr_array_blocks_delta_recurrent_forgets_gates,
                                                                                                                            batch_size * this->total_block_units * this->seq_w,
                                                                                                                            this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_delta_recurrent_forgets_gates = tmp_ptr_array_delta_recurrent_forgets_gates;
        // |END| Allocating LSTM delta recurrent forget gate. |END|
        
        // Allocating LSTM delta outputs gate.
        real *tmp_ptr_array_delta_outputs_gates(Mem::reallocate(this->ptr_array_blocks_delta_outputs_gates,
                                                                                                             batch_size * this->total_block_units * this->seq_w,
                                                                                                             this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_delta_outputs_gates = tmp_ptr_array_delta_outputs_gates;
        // |END| Allocating LSTM delta outputs gate. |END|
        
        // Allocating LSTM delta input outputs gate.
        real *tmp_ptr_array_delta_input_outputs_gates(Mem::reallocate(this->ptr_array_blocks_delta_input_outputs_gates,
                                                                                                                     batch_size * this->total_block_units * this->seq_w,
                                                                                                                     this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_delta_input_outputs_gates = tmp_ptr_array_delta_input_outputs_gates;
        // |END| Allocating LSTM delta input outputs gate. |END|
        
        // Allocating LSTM delta recurrent outputs gate.
        real *tmp_ptr_array_delta_recurrent_outputs_gates(Mem::reallocate(this->ptr_array_blocks_delta_recurrent_outputs_gates,
                                                                                                                            batch_size * this->total_block_units * this->seq_w,
                                                                                                                            this->batch_size * this->total_block_units * this->seq_w));
        this->ptr_array_blocks_delta_recurrent_outputs_gates = tmp_ptr_array_delta_recurrent_outputs_gates;
        // |END| Allocating LSTM delta recurrent outputs gate. |END|
        
        for(layer_it = this->ptr_array_layers + 1; layer_it != last_layer; ++layer_it)
        {
            tmp_number_block_units = static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units);
            tmp_number_cell_units = static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units);
            
            if(tmp_number_block_units * tmp_number_cell_units != 0_UZ)
            {
                for(tmp_ptr_last_block_unit = layer_it->ptr_last_block_unit,
                    tmp_ptr_block_unit_it = layer_it->ptr_array_block_units; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it,
                                                                                                                                                                                          ++tmp_ptr_array_summation_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_input_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_recurrent_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_input_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_recurrent_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_input_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_summation_recurrent_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_input_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_recurrent_inputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_input_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_recurrent_forgets_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_input_outputs_gates,
                                                                                                                                                                                          ++tmp_ptr_array_delta_recurrent_outputs_gates)
                {
                    tmp_ptr_block_unit_it->ptr_array_summation_cells_inputs = tmp_ptr_array_summation_cells_inputs;
                    tmp_ptr_block_unit_it->ptr_array_summation_input_cells_inputs = tmp_ptr_array_summation_input_cells_inputs;
                    tmp_ptr_block_unit_it->ptr_array_summation_recurrent_cells_inputs = tmp_ptr_array_summation_recurrent_cells_inputs;
                    tmp_ptr_block_unit_it->ptr_summation_inputs_gates = tmp_ptr_array_summation_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_summation_input_inputs_gates = tmp_ptr_array_summation_input_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_summation_recurrent_inputs_gates = tmp_ptr_array_summation_recurrent_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_summation_forgets_gates = tmp_ptr_array_summation_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_summation_input_forgets_gates = tmp_ptr_array_summation_input_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_summation_recurrent_forgets_gates = tmp_ptr_array_summation_recurrent_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_summation_outputs_gates = tmp_ptr_array_summation_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_summation_input_outputs_gates = tmp_ptr_array_summation_input_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_summation_recurrent_outputs_gates = tmp_ptr_array_summation_recurrent_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_array_cells_inputs = tmp_ptr_array_cells_inputs;
                    tmp_ptr_block_unit_it->ptr_array_cells_states = tmp_ptr_array_cells_states;
                    tmp_ptr_block_unit_it->ptr_array_cells_states_activates = tmp_ptr_array_cells_states_activates;
                    tmp_ptr_block_unit_it->ptr_array_cells_outputs = tmp_ptr_array_cells_outputs;
                    tmp_ptr_block_unit_it->ptr_inputs_gates = tmp_ptr_array_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_forgets_gates = tmp_ptr_array_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_outputs_gates = tmp_ptr_array_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_array_delta_cells_inputs = tmp_ptr_array_delta_cells_inputs;
                    tmp_ptr_block_unit_it->ptr_array_delta_cells_input_inputs = tmp_ptr_array_delta_cells_input_inputs;
                    tmp_ptr_block_unit_it->ptr_array_delta_cells_recurrent_inputs = tmp_ptr_array_delta_cells_recurrent_inputs;
                    tmp_ptr_block_unit_it->ptr_array_delta_cells_states = tmp_ptr_array_delta_cells_states;
                    tmp_ptr_block_unit_it->ptr_array_delta_cells_outputs = tmp_ptr_array_delta_cells_outputs;
                    tmp_ptr_block_unit_it->ptr_delta_inputs_gates = tmp_ptr_array_delta_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_delta_input_inputs_gates = tmp_ptr_array_delta_input_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_delta_recurrent_inputs_gates = tmp_ptr_array_delta_recurrent_inputs_gates;
                    tmp_ptr_block_unit_it->ptr_delta_forgets_gates = tmp_ptr_array_delta_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_delta_input_forgets_gates = tmp_ptr_array_delta_input_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_delta_recurrent_forgets_gates = tmp_ptr_array_delta_recurrent_forgets_gates;
                    tmp_ptr_block_unit_it->ptr_delta_outputs_gates = tmp_ptr_array_delta_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_delta_input_outputs_gates = tmp_ptr_array_delta_input_outputs_gates;
                    tmp_ptr_block_unit_it->ptr_delta_recurrent_outputs_gates = tmp_ptr_array_delta_recurrent_outputs_gates;
                    
                    for(tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
                        tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units; tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit; ++tmp_ptr_cell_unit_it,
                                                                                                                                                                                           ++tmp_ptr_array_summation_cells_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_summation_input_cells_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_summation_recurrent_cells_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_cells_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_cells_states,
                                                                                                                                                                                           ++tmp_ptr_array_cells_states_activates,
                                                                                                                                                                                           ++tmp_ptr_array_cells_outputs,
                                                                                                                                                                                           ++tmp_ptr_array_delta_cells_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_delta_cells_input_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_delta_cells_recurrent_inputs,
                                                                                                                                                                                           ++tmp_ptr_array_delta_cells_states,
                                                                                                                                                                                           ++tmp_ptr_array_delta_cells_outputs)
                    {
                            tmp_ptr_cell_unit_it->ptr_summation_cell_input = tmp_ptr_array_summation_cells_inputs;
                            tmp_ptr_cell_unit_it->ptr_summation_input_cell_input = tmp_ptr_array_summation_input_cells_inputs;
                            tmp_ptr_cell_unit_it->ptr_summation_recurrent_cell_input = tmp_ptr_array_summation_recurrent_cells_inputs;
                            tmp_ptr_cell_unit_it->ptr_cell_input = tmp_ptr_array_cells_inputs;
                            tmp_ptr_cell_unit_it->ptr_cell_state = tmp_ptr_array_cells_states;
                            tmp_ptr_cell_unit_it->ptr_cell_state_activate = tmp_ptr_array_cells_states_activates;
                            tmp_ptr_cell_unit_it->ptr_cell_output = tmp_ptr_array_cells_outputs;
                            tmp_ptr_cell_unit_it->ptr_delta_cell_input = tmp_ptr_array_delta_cells_inputs;
                            tmp_ptr_cell_unit_it->ptr_delta_cell_input_input = tmp_ptr_array_delta_cells_input_inputs;
                            tmp_ptr_cell_unit_it->ptr_delta_cell_recurrent_input = tmp_ptr_array_delta_cells_recurrent_inputs;
                            tmp_ptr_cell_unit_it->ptr_delta_cell_state = tmp_ptr_array_delta_cells_states;
                            tmp_ptr_cell_unit_it->ptr_delta_cell_output = tmp_ptr_array_delta_cells_outputs;
                    }
                }

                tmp_ptr_array_summation_cells_inputs += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_input_cells_inputs += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_recurrent_cells_inputs += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_inputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_input_inputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_recurrent_inputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_forgets_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_input_forgets_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_recurrent_forgets_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_outputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_input_outputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_recurrent_outputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_cells_inputs += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_cells_states += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_cells_states_activates += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_cells_outputs += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_inputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_forgets_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_outputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_cells_inputs += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_cells_input_inputs += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_cells_recurrent_inputs += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_cells_states += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_cells_outputs += (batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_inputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_input_inputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_recurrent_inputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_forgets_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_input_forgets_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_recurrent_forgets_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_outputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_input_outputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_recurrent_outputs_gates += (batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
            }
        }
    }

    return true;
}

bool Model::Reallocate__Batch__Dropout__ShakeDrop(size_t const batch_size)
{
    if(this->total_layers != 0_UZ)
    {
        bool *tmp_ptr_array_layers_mask_dropout_shakedrop(Mem::reallocate<bool, false>(
          this->ptr_array_layers_mask_dropout_shakedrop,
           this->total_layers * this->seq_w * batch_size,
                         this->total_layers * this->seq_w * this->batch_size));
        if(tmp_ptr_array_layers_mask_dropout_shakedrop == nullptr)
        {
            ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu)\" function.",
                                     sizeof(bool),
                                     this->total_layers * this->seq_w * batch_size,
                                     this->total_layers * this->seq_w * this->batch_size);

            return false;
        }
        this->ptr_array_layers_mask_dropout_shakedrop = tmp_ptr_array_layers_mask_dropout_shakedrop;
        
        for(Layer *layer_it(this->ptr_array_layers); layer_it != this->ptr_last_layer; ++layer_it)
        {
            layer_it->ptr_array__mask__dropout__shakedrop = tmp_ptr_array_layers_mask_dropout_shakedrop;
            
            tmp_ptr_array_layers_mask_dropout_shakedrop += this->seq_w * batch_size;
        }
    }

    return true;
}
}
