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

#include <chrono>

namespace DL::v1 {
bool Model::Allocate__Structure(size_t const n_layers, size_t const allowable_memory)
{
    this->maximum_allowable_memory_bytes = allowable_memory;

    // allocate layers.
    Layer *layer_it(new Layer[n_layers]);
    this->ptr_array_layers = layer_it;
    this->ptr_last_layer = this->ptr_array_layers + n_layers;
    this->total_layers = n_layers;
    
    size_t *tmp_ptr_array_layers_number_outputs_it(new size_t[n_layers]());
    this->ptr_array_layers_number_outputs = tmp_ptr_array_layers_number_outputs_it;
    
    size_t *tmp_ptr_array_layers_first_connection_index_it(new size_t[n_layers]());
    this->ptr_array_layers_first_connection_index = tmp_ptr_array_layers_first_connection_index_it;
    
    size_t *tmp_ptr_array_layers_last_connection_index_it(new size_t[n_layers]());
    this->ptr_array_layers_last_connection_index = tmp_ptr_array_layers_last_connection_index_it;
    
    for(; layer_it != this->ptr_last_layer; ++layer_it,
                                                                    ++tmp_ptr_array_layers_number_outputs_it,
                                                                    ++tmp_ptr_array_layers_first_connection_index_it,
                                                                    ++tmp_ptr_array_layers_last_connection_index_it)
    {
        layer_it->ptr_number_outputs = tmp_ptr_array_layers_number_outputs_it;

        layer_it->ptr_first_connection_index = tmp_ptr_array_layers_first_connection_index_it;
        layer_it->ptr_last_connection_index = tmp_ptr_array_layers_last_connection_index_it;
    }
    // |END| allocate layers. |END|

    this->ptr_array_number_loss = new size_t[1]();
    this->ptr_array_number_bit_fail = new size_t[1]();
    this->ptr_array_loss_values = new double[1]();
    
    for (int i(0); i != 5; ++i)
      this->ptr_array_accuracy_values[i] = new double[1];
    
    this->real_gen.seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    this->real_gen.range(0_r, 1_r);

    this->gaussian.seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    this->gaussian.range(0_r, 1_r);

    return true;
}

bool Model::Allocate__Sparse_K_Filter(void) {
  this->ptr_array_k_sparse_activities =
      new std::pair<size_t, var>[this->number_threads *
                                 (this->total_AF_units +
                                  this->total_AF_Ind_recurrent_units +
                                  this->total_block_units)];

  this->Assign__Sparsity_Activities(this->number_threads);

  return true;
}

bool Model::Allocate__Generator__Dropout_Bernoulli(void)
{
    if(this->bernoulli == nullptr)
    {
    this->bernoulli = new class Dist::Bernoulli[this->number_threads];

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        { this->bernoulli[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    return true;
}

bool Model::Allocate__Generator__Dropout_Gaussian(void)
{
    if(this->ptr_array_Class_Generator_Real_Gaussian == nullptr)
    {
    this->ptr_array_Class_Generator_Real_Gaussian =
        new class Dist::Gaussian[this->number_threads];

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        {
            this->ptr_array_Class_Generator_Real_Gaussian[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            this->ptr_array_Class_Generator_Real_Gaussian[tmp_generator_index].range(0_r, 1_r);
        }
    }

    return true;
}

bool Model::Allocate__Generator__Dropout_ShakeDrop(void)
{
    if(this->ptr_array_Class_Generator_Bernoulli_ShakeDrop == nullptr)
    {
    this->ptr_array_Class_Generator_Bernoulli_ShakeDrop =
        new class Dist::Bernoulli[this->number_threads];

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        { this->ptr_array_Class_Generator_Bernoulli_ShakeDrop[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    if(this->ptr_array_Class_Generator_Real_ShakeDrop == nullptr)
    {
      this->ptr_array_Class_Generator_Real_ShakeDrop =
          new class Dist::Real[this->number_threads];

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        {
            this->ptr_array_Class_Generator_Real_ShakeDrop[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            this->ptr_array_Class_Generator_Real_ShakeDrop[tmp_generator_index].range(0_r, 1_r);
        }
    }

    return true;
}

bool Model::Allocate__Generator__Dropout_Uout(void)
{
    if(this->ptr_array_Class_Generator_Real_Uout == nullptr)
    {
    this->ptr_array_Class_Generator_Real_Uout =
        new class Dist::Real[this->number_threads];

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        {
            this->ptr_array_Class_Generator_Real_Uout[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            this->ptr_array_Class_Generator_Real_Uout[tmp_generator_index].range(0_r, 1_r);
        }
    }

    return true;
}

bool Model::Allocate__Generator__Dropout_Zoneout(void)
{
    if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_State == nullptr)
    {
    this->ptr_array_Class_Generator_Bernoulli_Zoneout_State =
        new class Dist::Bernoulli[this->number_threads];

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        { this->ptr_array_Class_Generator_Bernoulli_Zoneout_State[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }
    
    if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden == nullptr)
    {
      this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden =
          new class Dist::Bernoulli[this->number_threads];

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != this->number_threads; ++tmp_generator_index)
        { this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    return true;
}

bool Model::Allocate__Neuron__Mask_Dropout_Bernoulli(void)
{
    if(this->ptr_array_units_mask_dropout_bernoulli == nullptr)
    {
        bool *tmp_ptr_array_units_mask_dropout_bernoulli(new bool[(this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated) * this->seq_w]);
        Mem::fill(tmp_ptr_array_units_mask_dropout_bernoulli,
                                     tmp_ptr_array_units_mask_dropout_bernoulli + (this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated) * this->seq_w,
                                     true);
        
        this->Reset__Parameter__Mask_Dropout(tmp_ptr_array_units_mask_dropout_bernoulli);
    }

    return true;
}

bool Model::Allocate__Layer__Mask__Dropout__ShakeDrop(void)
{
    if(this->ptr_array_layers_mask_dropout_shakedrop == nullptr)
    {
        bool *tmp_ptr_array_layers_mask_dropout_shakedrop(new bool[this->total_layers * this->seq_w * this->batch_size]);
        this->ptr_array_layers_mask_dropout_shakedrop = tmp_ptr_array_layers_mask_dropout_shakedrop;
        Mem::fill(tmp_ptr_array_layers_mask_dropout_shakedrop,
                                     tmp_ptr_array_layers_mask_dropout_shakedrop + this->total_layers * this->seq_w * this->batch_size,
                                     true);
        
        for(Layer *layer_it(this->ptr_array_layers); layer_it != this->ptr_last_layer; ++layer_it)
        {
            layer_it->ptr_array__mask__dropout__shakedrop = tmp_ptr_array_layers_mask_dropout_shakedrop;
            
            tmp_ptr_array_layers_mask_dropout_shakedrop += this->seq_w * this->batch_size;
        }
    }

    return true;
}

bool Model::Allocate__Basic_Units(void)
{
    size_t tmp_number_basic_units,
              tmp_basic_unit_index;
    
    if(this->ptr_array_basic_units == nullptr && this->total_basic_units != 0_UZ)
    {
        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);

        Basic_unit *tmp_ptr_array_basic_units(new Basic_unit[this->total_basic_units]);
        this->ptr_array_basic_units = tmp_ptr_array_basic_units;
        
        var *tmp_ptr_array_basic_units_values(new var[this->batch_size * this->total_basic_units * this->seq_w]());
        this->ptr_array_basic_units_values = tmp_ptr_array_basic_units_values;
        
        real *tmp_ptr_array_basic_units_errors(
            new real[this->batch_size * this->total_basic_units * this->seq_w]());
        this->ptr_array_basic_units_errors = tmp_ptr_array_basic_units_errors;

        for(; layer_it != last_layer; ++layer_it)
        {
            tmp_number_basic_units = static_cast<size_t>(layer_it->ptr_last_basic_unit - layer_it->ptr_array_basic_units);

            layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
            
            if(tmp_number_basic_units != 0_UZ)
            {
                // Assign basic unit variables.
                for(tmp_basic_unit_index = 0_UZ; tmp_basic_unit_index != tmp_number_basic_units; ++tmp_basic_unit_index,
                                                                                                                                            ++tmp_ptr_array_basic_units_values,
                                                                                                                                            ++tmp_ptr_array_basic_units_errors)
                {
                    tmp_ptr_array_basic_units[tmp_basic_unit_index].ptr_array_values = tmp_ptr_array_basic_units_values;
                    tmp_ptr_array_basic_units[tmp_basic_unit_index].ptr_array_errors = tmp_ptr_array_basic_units_errors;
                }

                tmp_ptr_array_basic_units_values += (this->batch_size - 1_UZ) * tmp_number_basic_units * this->seq_w + tmp_number_basic_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_basic_units_errors += (this->batch_size - 1_UZ) * tmp_number_basic_units * this->seq_w + tmp_number_basic_units * (this->seq_w - 1_UZ);
                // |END| Assign basic unit variables. |END|
                
                tmp_ptr_array_basic_units += tmp_number_basic_units;
            }

            layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
        }
        
        this->ptr_last_basic_unit = tmp_ptr_array_basic_units;

        this->total_basic_units_allocated = this->total_basic_units;
    }

    return true;
}

bool Model::Allocate__Basic_Indice_Units(void)
{
    size_t tmp_number_basic_indice_units,
              tmp_basic_indice_unit_index;
    
    if(this->ptr_array_basic_indice_units == nullptr && this->total_basic_indice_units != 0_UZ)
    {
        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);

        Basic_indice_unit *tmp_ptr_array_basic_indice_units(new Basic_indice_unit[this->total_basic_indice_units]);
        this->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
        
        size_t *tmp_ptr_array_basic_indice_units_indices(new size_t[this->batch_size * this->total_basic_indice_units * this->seq_w]());
        this->ptr_array_basic_indice_units_indices = tmp_ptr_array_basic_indice_units_indices;
        
        var *tmp_ptr_array_basic_indice_units_values(new var[this->batch_size * this->total_basic_indice_units * this->seq_w]());
        this->ptr_array_basic_indice_units_values = tmp_ptr_array_basic_indice_units_values;
        
        real *tmp_ptr_array_basic_indice_units_errors(
            new real[this->batch_size * this->total_basic_indice_units *
                     this->seq_w]());
        this->ptr_array_basic_indice_units_errors = tmp_ptr_array_basic_indice_units_errors;

        for(; layer_it != last_layer; ++layer_it)
        {
            tmp_number_basic_indice_units = static_cast<size_t>(layer_it->ptr_last_basic_indice_unit - layer_it->ptr_array_basic_indice_units);

            layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
            
            if(tmp_number_basic_indice_units != 0_UZ)
            {
                // Assign basic unit variables.
                for(tmp_basic_indice_unit_index = 0_UZ; tmp_basic_indice_unit_index != tmp_number_basic_indice_units; ++tmp_basic_indice_unit_index,
                                                                                                                                                                          ++tmp_ptr_array_basic_indice_units_indices,
                                                                                                                                                                          ++tmp_ptr_array_basic_indice_units_values,
                                                                                                                                                                          ++tmp_ptr_array_basic_indice_units_errors)
                {
                    tmp_ptr_array_basic_indice_units[tmp_basic_indice_unit_index].ptr_array_indices = tmp_ptr_array_basic_indice_units_indices;

                    tmp_ptr_array_basic_indice_units[tmp_basic_indice_unit_index].ptr_array_values = tmp_ptr_array_basic_indice_units_values;
                    tmp_ptr_array_basic_indice_units[tmp_basic_indice_unit_index].ptr_array_errors = tmp_ptr_array_basic_indice_units_errors;
                }

                tmp_ptr_array_basic_indice_units_indices += (this->batch_size - 1_UZ) * tmp_number_basic_indice_units * this->seq_w + tmp_number_basic_indice_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_basic_indice_units_values += (this->batch_size - 1_UZ) * tmp_number_basic_indice_units * this->seq_w + tmp_number_basic_indice_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_basic_indice_units_errors += (this->batch_size - 1_UZ) * tmp_number_basic_indice_units * this->seq_w + tmp_number_basic_indice_units * (this->seq_w - 1_UZ);
                // |END| Assign basic unit variables. |END|
                
                tmp_ptr_array_basic_indice_units += tmp_number_basic_indice_units;
            }

            layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
        }
        
        this->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;

        this->total_basic_indice_units_allocated = this->total_basic_indice_units;
    }

    return true;
}

bool Model::Allocate__Neuron_Units(void)
{
    size_t tmp_number_neuron_units,
              tmp_neuron_index;
    
    if(this->ptr_array_neuron_units == nullptr && this->total_neuron_units != 0_UZ)
    {
        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);

        Neuron_unit *tmp_ptr_array_neuron_units(new Neuron_unit[this->total_neuron_units]);
        this->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
        
        size_t *tmp_ptr_array_neuron_units_first_forward_connection_index(new size_t[this->total_neuron_units]());
        this->ptr_array_neuron_units_first_forward_connection_index = tmp_ptr_array_neuron_units_first_forward_connection_index;
        
        size_t *tmp_ptr_array_neuron_units_last_forward_connection_index(new size_t[this->total_neuron_units]());
        this->ptr_array_neuron_units_last_forward_connection_index = tmp_ptr_array_neuron_units_last_forward_connection_index;
        
        size_t *tmp_ptr_array_neuron_units_number_forward_connections(new size_t[this->total_neuron_units]());
        this->ptr_array_neuron_units_number_forward_connections = tmp_ptr_array_neuron_units_number_forward_connections;
        
        var *tmp_ptr_array_neuron_units_summations(
            new var[this->batch_size * this->total_neuron_units *
                    this->seq_w]());
        this->ptr_array_neuron_units_summations = tmp_ptr_array_neuron_units_summations;
        
        real *tmp_ptr_array_neuron_units_errors(
            new real[this->batch_size * this->total_neuron_units *
                     this->seq_w]());
        this->ptr_array_neuron_units_errors = tmp_ptr_array_neuron_units_errors;
        
        for(; layer_it != last_layer; ++layer_it)
        {
            tmp_number_neuron_units = static_cast<size_t>(layer_it->ptr_last_neuron_unit - layer_it->ptr_array_neuron_units);

            layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
            
            if(tmp_number_neuron_units != 0_UZ)
            {
                // Assign neurons variable.
                for(tmp_neuron_index = 0_UZ; tmp_neuron_index != tmp_number_neuron_units; ++tmp_neuron_index,
                                                                                                                                    ++tmp_ptr_array_neuron_units_first_forward_connection_index,
                                                                                                                                    ++tmp_ptr_array_neuron_units_last_forward_connection_index,
                                                                                                                                    ++tmp_ptr_array_neuron_units_number_forward_connections,
                                                                                                                                    ++tmp_ptr_array_neuron_units_summations,
                                                                                                                                    ++tmp_ptr_array_neuron_units_errors)
                {
                    tmp_ptr_array_neuron_units[tmp_neuron_index].ptr_first_connection_index = tmp_ptr_array_neuron_units_first_forward_connection_index;
                    tmp_ptr_array_neuron_units[tmp_neuron_index].ptr_last_connection_index = tmp_ptr_array_neuron_units_last_forward_connection_index;
                    tmp_ptr_array_neuron_units[tmp_neuron_index].ptr_number_connections = tmp_ptr_array_neuron_units_number_forward_connections;

                    tmp_ptr_array_neuron_units[tmp_neuron_index].ptr_array_summations = tmp_ptr_array_neuron_units_summations;
                    tmp_ptr_array_neuron_units[tmp_neuron_index].ptr_array_errors = tmp_ptr_array_neuron_units_errors;
                }

                tmp_ptr_array_neuron_units_summations += (this->batch_size - 1_UZ) * tmp_number_neuron_units * this->seq_w + tmp_number_neuron_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_neuron_units_errors += (this->batch_size - 1_UZ) * tmp_number_neuron_units * this->seq_w + tmp_number_neuron_units * (this->seq_w - 1_UZ);
                // |END| Assign neurons variable. |END|
                
                tmp_ptr_array_neuron_units += tmp_number_neuron_units;
            }

            layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
        }
        
        this->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;

        this->total_neuron_units_allocated = this->total_neuron_units;
    }

    return true;
}

bool Model::Allocate__AF_Units(void)
{
    size_t tmp_number_AF_units,
              tmp_AF_index;
    
    if(this->ptr_array_AF_units == nullptr && this->total_AF_units != 0_UZ)
    {
        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);

        AF_unit *tmp_ptr_array_AF_units(new AF_unit[this->total_AF_units]);
        this->ptr_array_AF_units = tmp_ptr_array_AF_units;
        
        var *tmp_ptr_array_AF_units_values(new var[this->batch_size * this->total_AF_units * this->seq_w]());
        this->ptr_array_AF_units_values = tmp_ptr_array_AF_units_values;
        
        real *tmp_ptr_array_AF_units_errors(
            new real[this->batch_size * this->total_AF_units * this->seq_w]());
        this->ptr_array_AF_units_errors = tmp_ptr_array_AF_units_errors;
        
        ACTIVATION::TYPE *tmp_ptr_array_AF_units_type_activations_functions(new ACTIVATION::TYPE[this->total_AF_units]);
        this->ptr_array_AF_units_type_activation_function = tmp_ptr_array_AF_units_type_activations_functions;
        Mem::fill(tmp_ptr_array_AF_units_type_activations_functions,
                                                                                                                              tmp_ptr_array_AF_units_type_activations_functions + this->total_AF_units,
                                                                                                                              ACTIVATION::NONE);
        
        for(; layer_it != last_layer; ++layer_it)
        {
            tmp_number_AF_units = static_cast<size_t>(layer_it->ptr_last_AF_unit - layer_it->ptr_array_AF_units);

            layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
            
            if(tmp_number_AF_units != 0_UZ)
            {
                // Assign AF unit(s) variable.
                for(tmp_AF_index = 0_UZ; tmp_AF_index != tmp_number_AF_units; ++tmp_AF_index,
                                                                                                              ++tmp_ptr_array_AF_units_values,
                                                                                                              ++tmp_ptr_array_AF_units_errors,
                                                                                                              ++tmp_ptr_array_AF_units_type_activations_functions)
                {
                    tmp_ptr_array_AF_units[tmp_AF_index].ptr_array_values = tmp_ptr_array_AF_units_values;
                    tmp_ptr_array_AF_units[tmp_AF_index].ptr_array_errors = tmp_ptr_array_AF_units_errors;

                    tmp_ptr_array_AF_units[tmp_AF_index].ptr_type_activation_function = tmp_ptr_array_AF_units_type_activations_functions;
                }

                tmp_ptr_array_AF_units_values += (this->batch_size - 1_UZ) * tmp_number_AF_units * this->seq_w + tmp_number_AF_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_AF_units_errors += (this->batch_size - 1_UZ) * tmp_number_AF_units * this->seq_w + tmp_number_AF_units * (this->seq_w - 1_UZ);
                // |END| Assign AF unit(s) variable. |END|
                
                tmp_ptr_array_AF_units += tmp_number_AF_units;
            }

            layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
        }
        
        this->ptr_last_AF_unit = tmp_ptr_array_AF_units;

        this->total_AF_units_allocated = this->total_AF_units;
    }

    return true;
}

bool Model::Allocate__AF_Ind_Recurrent_Units(void)
{
    size_t tmp_number_AF_Ind_recurrent_units,
              tmp_AF_Ind_index;
    
    if(this->ptr_array_AF_Ind_recurrent_units == nullptr && this->total_AF_Ind_recurrent_units != 0_UZ)
    {
        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);

        AF_Ind_recurrent_unit *tmp_ptr_array_AF_Ind_recurrent_units(new AF_Ind_recurrent_unit[this->total_AF_Ind_recurrent_units]);
        this->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
        
        size_t *tmp_ptr_array_AF_Ind_recurrent_units_first_recurrent_connection_index(new size_t[this->total_AF_Ind_recurrent_units]());
        this->ptr_array_AF_Ind_recurrent_units_recurrent_connection_index = tmp_ptr_array_AF_Ind_recurrent_units_first_recurrent_connection_index;
        
        var *tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs(
            new var[this->batch_size * this->total_AF_Ind_recurrent_units *
                    this->seq_w]());
        this->ptr_array_AF_Ind_recurrent_units_pre_AFs = tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs;
        
        var *tmp_ptr_array_AF_Ind_recurrent_units_AFs(
            new var[this->batch_size * this->total_AF_Ind_recurrent_units *
                    this->seq_w]());
        this->ptr_array_AF_Ind_recurrent_units_AFs = tmp_ptr_array_AF_Ind_recurrent_units_AFs;
        
        real *tmp_ptr_array_AF_Ind_recurrent_units_errors(
            new real[this->batch_size * this->total_AF_Ind_recurrent_units *
                     this->seq_w]());
        this->ptr_array_AF_Ind_recurrent_units_errors = tmp_ptr_array_AF_Ind_recurrent_units_errors;
        
        real *tmp_ptr_array_AF_Ind_recurrent_units_dAFs(
            new real[this->batch_size * this->total_AF_Ind_recurrent_units *
                    this->seq_w]());
        this->ptr_array_AF_Ind_recurrent_units_dAFs = tmp_ptr_array_AF_Ind_recurrent_units_dAFs;
        
        ACTIVATION::TYPE *tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions(new ACTIVATION::TYPE[this->total_AF_Ind_recurrent_units]);
        this->ptr_array_AF_Ind_recurrent_units_type_activation_function = tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions;
        Mem::fill<ACTIVATION::TYPE>(tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions,
                                                                                                                              tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions + this->total_AF_Ind_recurrent_units,
                                                                                                                              ACTIVATION::NONE);
        
        for(; layer_it != last_layer; ++layer_it)
        {
            tmp_number_AF_Ind_recurrent_units = static_cast<size_t>(layer_it->ptr_last_AF_Ind_recurrent_unit - layer_it->ptr_array_AF_Ind_recurrent_units);

            layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
            
            if(tmp_number_AF_Ind_recurrent_units != 0_UZ)
            {
                // Assign AF Ind recurrent unit(s) variable.
                for(tmp_AF_Ind_index = 0_UZ; tmp_AF_Ind_index != tmp_number_AF_Ind_recurrent_units; ++tmp_AF_Ind_index,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_first_recurrent_connection_index,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_AFs,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_errors,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_dAFs,
                                                                                                                                ++tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions)
                {
                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_recurrent_connection_index = tmp_ptr_array_AF_Ind_recurrent_units_first_recurrent_connection_index;

                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_array_pre_AFs = tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs;
                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_array_AFs = tmp_ptr_array_AF_Ind_recurrent_units_AFs;
                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_array_errors = tmp_ptr_array_AF_Ind_recurrent_units_errors;
                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_array_dAFs = tmp_ptr_array_AF_Ind_recurrent_units_dAFs;

                    tmp_ptr_array_AF_Ind_recurrent_units[tmp_AF_Ind_index].ptr_type_activation_function = tmp_ptr_array_AF_Ind_recurrent_units_type_activations_functions;
                }

                tmp_ptr_array_AF_Ind_recurrent_units_pre_AFs += (this->batch_size - 1_UZ) * tmp_number_AF_Ind_recurrent_units * this->seq_w + tmp_number_AF_Ind_recurrent_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_AF_Ind_recurrent_units_AFs += (this->batch_size - 1_UZ) * tmp_number_AF_Ind_recurrent_units * this->seq_w + tmp_number_AF_Ind_recurrent_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_AF_Ind_recurrent_units_errors += (this->batch_size - 1_UZ) * tmp_number_AF_Ind_recurrent_units * this->seq_w + tmp_number_AF_Ind_recurrent_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_AF_Ind_recurrent_units_dAFs += (this->batch_size - 1_UZ) * tmp_number_AF_Ind_recurrent_units * this->seq_w + tmp_number_AF_Ind_recurrent_units * (this->seq_w - 1_UZ);
                // |END| Assign AF Ind recurrent unit(s) variable. |END|
                
                tmp_ptr_array_AF_Ind_recurrent_units += tmp_number_AF_Ind_recurrent_units;
            }

            layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
        }
        
        this->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;

        this->total_AF_Ind_recurrent_units_allocated = this->total_AF_Ind_recurrent_units;
    }

    return true;
}

bool Model::Allocate__Normalized_Unit(bool const organize_pointers_received)
{
    if(this->ptr_array_normalized_units == nullptr)
    {
        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it;
        
        // TODO: allocate the exact number of necessary normalized unit(s) per layer.
        if(organize_pointers_received) { this->Prepare__Normalized__Layers(); }

        if(this->total_normalized_units != 0_UZ)
        {
            union Normalized_unit *tmp_ptr_array_normalized_units(new union Normalized_unit[this->total_normalized_units]);
            this->ptr_array_normalized_units = tmp_ptr_array_normalized_units;

            if(organize_pointers_received)
            {
                size_t tmp_number_normalized_units;

                for(layer_it = this->ptr_array_layers; layer_it != last_layer; ++layer_it)
                {
                    tmp_number_normalized_units = static_cast<size_t>(layer_it->ptr_last_normalized_unit - layer_it->ptr_array_normalized_units);

                    layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                    tmp_ptr_array_normalized_units += tmp_number_normalized_units;
                    layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                }
            }
            
            this->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;

            this->total_normalized_units_allocated = this->total_normalized_units;
        }
    }

    return true;
}

bool Model::Allocate__Block_Unit__Mask_Dropout_Zoneout(void)
{
    if(this->ptr_array_cell_units_mask_dropout_zoneout == nullptr)
    {
        bool *tmp_ptr_array_cell_units_mask_dropout_zoneout(new bool[2_UZ * this->total_cell_units_allocated * this->seq_w]);
        Mem::fill(tmp_ptr_array_cell_units_mask_dropout_zoneout,
                                     tmp_ptr_array_cell_units_mask_dropout_zoneout + 2_UZ * this->total_cell_units_allocated * this->seq_w,
                                     true);
        
        this->Reset__Parameters__Cell_Unit__Mask_Dropout(tmp_ptr_array_cell_units_mask_dropout_zoneout);
    }

    return true;
}

bool Model::Allocate__LSTM_Layers(void)
{
    size_t tmp_number_block_units,
              tmp_number_cell_units,
              tmp_number_cell_units_per_block,
              tmp_block_index,
              tmp_block_index_cell_index;
    
    if(this->total_block_units * this->total_cell_units != 0_UZ)
    {
        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it;

        BlockUnit *tmp_ptr_array_block_units(new BlockUnit[this->total_block_units]);
        
        CellUnit *tmp_ptr_array_cell_units(new CellUnit[this->total_cell_units]);
        
        var *tmp_ptr_array_summation_cells_inputs(new var[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_summations_cells_inputs = tmp_ptr_array_summation_cells_inputs;
        
        var *tmp_ptr_array_summation_input_cells_inputs(
            new var[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_summations_input_cells_inputs = tmp_ptr_array_summation_input_cells_inputs;
        
        var *tmp_ptr_array_summation_recurrent_cells_inputs(
            new var[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_summations_recurrent_cells_inputs = tmp_ptr_array_summation_recurrent_cells_inputs;
        
        var *tmp_ptr_array_summation_inputs_gates(
            new var[this->batch_size * this->total_block_units *
                    this->seq_w]());
        this->ptr_array_blocks_summations_inputs_gates = tmp_ptr_array_summation_inputs_gates;
        
        var *tmp_ptr_array_summation_input_inputs_gates(
            new var[this->batch_size * this->total_block_units *
                    this->seq_w]());
        this->ptr_array_blocks_summations_input_inputs_gates = tmp_ptr_array_summation_input_inputs_gates;
        
        var *tmp_ptr_array_summation_recurrent_inputs_gates(
            new var[this->batch_size * this->total_block_units *
                    this->seq_w]());
        this->ptr_array_blocks_summations_recurrent_inputs_gates = tmp_ptr_array_summation_recurrent_inputs_gates;
        
        var *tmp_ptr_array_summation_forgets_gates(
            new var[this->batch_size * this->total_block_units *
                    this->seq_w]());
        this->ptr_array_blocks_summations_forgets_gates = tmp_ptr_array_summation_forgets_gates;
        
        var *tmp_ptr_array_summation_input_forgets_gates(
            new var[this->batch_size * this->total_block_units *
                    this->seq_w]());
        this->ptr_array_blocks_summations_input_forgets_gates = tmp_ptr_array_summation_input_forgets_gates;
        
        var *tmp_ptr_array_summation_recurrent_forgets_gates(new var[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_summations_recurrent_forgets_gates = tmp_ptr_array_summation_recurrent_forgets_gates;
        
        var *tmp_ptr_array_summation_outputs_gates(new var[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_summations_outputs_gates = tmp_ptr_array_summation_outputs_gates;
        
        var *tmp_ptr_array_summation_input_outputs_gates(new var[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_summations_input_outputs_gates = tmp_ptr_array_summation_input_outputs_gates;
        
        var *tmp_ptr_array_summation_recurrent_outputs_gates(new var[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_summations_recurrent_outputs_gates = tmp_ptr_array_summation_recurrent_outputs_gates;
        
        var *tmp_ptr_array_cells_inputs(new var[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_inputs = tmp_ptr_array_cells_inputs;
        
        var *tmp_ptr_array_cells_states(new var[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_states = tmp_ptr_array_cells_states;
        
        var *tmp_ptr_array_cells_states_activates(new var[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_states_activates = tmp_ptr_array_cells_states_activates;
        
        var *tmp_ptr_array_cells_outputs(new var[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_outputs = tmp_ptr_array_cells_outputs;
        
        var *tmp_ptr_array_inputs_gates(new var[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_inputs_gates = tmp_ptr_array_inputs_gates;
        
        var *tmp_ptr_array_forgets_gates(new var[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_forgets_gates = tmp_ptr_array_forgets_gates;
        
        var *tmp_ptr_array_outputs_gates(new var[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_outputs_gates = tmp_ptr_array_outputs_gates;
        
        real *tmp_ptr_array_delta_cells_inputs(
            new real[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_delta_inputs = tmp_ptr_array_delta_cells_inputs;
        
        real *tmp_ptr_array_delta_cells_input_inputs(
            new real[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_delta_input_inputs = tmp_ptr_array_delta_cells_input_inputs;
        
        real *tmp_ptr_array_delta_cells_recurrent_inputs(
            new real[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_delta_recurrent_inputs = tmp_ptr_array_delta_cells_recurrent_inputs;
        
        real *tmp_ptr_array_delta_cells_states(
            new real[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_delta_states = tmp_ptr_array_delta_cells_states;
        
        real *tmp_ptr_array_delta_cells_outputs(
            new real[this->batch_size * this->total_cell_units * this->seq_w]());
        this->ptr_array_cells_delta_outputs = tmp_ptr_array_delta_cells_outputs;
        
        real *tmp_ptr_array_delta_inputs_gates(
            new real[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_delta_inputs_gates = tmp_ptr_array_delta_inputs_gates;
        
        real *tmp_ptr_array_delta_input_inputs_gates(
            new real[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_delta_input_inputs_gates = tmp_ptr_array_delta_input_inputs_gates;
        
        real *tmp_ptr_array_delta_recurrent_inputs_gates(
            new real[this->batch_size * this->total_block_units *
                    this->seq_w]());
        this->ptr_array_blocks_delta_recurrent_inputs_gates = tmp_ptr_array_delta_recurrent_inputs_gates;
        
        real *tmp_ptr_array_delta_forgets_gates(
            new real[this->batch_size * this->total_block_units *
                     this->seq_w]());
        this->ptr_array_blocks_delta_forgets_gates = tmp_ptr_array_delta_forgets_gates;
        
        real *tmp_ptr_array_delta_input_forgets_gates(
            new real[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_delta_input_forgets_gates = tmp_ptr_array_delta_input_forgets_gates;

        real *tmp_ptr_array_delta_recurrent_forgets_gates(
            new real[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_delta_recurrent_forgets_gates = tmp_ptr_array_delta_recurrent_forgets_gates;

        real *tmp_ptr_array_delta_outputs_gates(
            new real[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_delta_outputs_gates = tmp_ptr_array_delta_outputs_gates;

        real *tmp_ptr_array_delta_input_outputs_gates(
            new real[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_delta_input_outputs_gates = tmp_ptr_array_delta_input_outputs_gates;

        real *tmp_ptr_array_delta_recurrent_outputs_gates(
            new real[this->batch_size * this->total_block_units * this->seq_w]());
        this->ptr_array_blocks_delta_recurrent_outputs_gates = tmp_ptr_array_delta_recurrent_outputs_gates;
        
        this->ptr_array_block_units = tmp_ptr_array_block_units;
        this->ptr_array_cell_units = tmp_ptr_array_cell_units;

        for(layer_it = this->ptr_array_layers; layer_it != last_layer; ++layer_it)
        {
            // [0] Assign block units.
            tmp_number_block_units = static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units);
            tmp_number_cell_units = static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units);
            
            layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
            layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
            
            if(tmp_number_block_units * tmp_number_cell_units != 0_UZ)
            {
                tmp_number_cell_units_per_block = tmp_number_cell_units / tmp_number_block_units;

                //    [1] Assign time step.
                for(tmp_block_index = 0_UZ; tmp_block_index != tmp_number_block_units; ++tmp_block_index,
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
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_summation_cells_inputs = tmp_ptr_array_summation_cells_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_summation_input_cells_inputs = tmp_ptr_array_summation_input_cells_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_summation_recurrent_cells_inputs = tmp_ptr_array_summation_recurrent_cells_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_inputs_gates = tmp_ptr_array_summation_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_input_inputs_gates = tmp_ptr_array_summation_input_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_recurrent_inputs_gates = tmp_ptr_array_summation_recurrent_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_forgets_gates = tmp_ptr_array_summation_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_input_forgets_gates = tmp_ptr_array_summation_input_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_recurrent_forgets_gates = tmp_ptr_array_summation_recurrent_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_outputs_gates = tmp_ptr_array_summation_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_input_outputs_gates = tmp_ptr_array_summation_input_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_summation_recurrent_outputs_gates = tmp_ptr_array_summation_recurrent_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_cells_inputs = tmp_ptr_array_cells_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_cells_states = tmp_ptr_array_cells_states;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_cells_states_activates = tmp_ptr_array_cells_states_activates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_cells_outputs = tmp_ptr_array_cells_outputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_inputs_gates = tmp_ptr_array_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_forgets_gates = tmp_ptr_array_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_outputs_gates = tmp_ptr_array_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_delta_cells_inputs = tmp_ptr_array_delta_cells_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_delta_cells_input_inputs = tmp_ptr_array_delta_cells_input_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_delta_cells_recurrent_inputs = tmp_ptr_array_delta_cells_recurrent_inputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_delta_cells_states = tmp_ptr_array_delta_cells_states;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_delta_cells_outputs = tmp_ptr_array_delta_cells_outputs;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_inputs_gates = tmp_ptr_array_delta_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_input_inputs_gates = tmp_ptr_array_delta_input_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_recurrent_inputs_gates = tmp_ptr_array_delta_recurrent_inputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_forgets_gates = tmp_ptr_array_delta_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_input_forgets_gates = tmp_ptr_array_delta_input_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_recurrent_forgets_gates = tmp_ptr_array_delta_recurrent_forgets_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_outputs_gates = tmp_ptr_array_delta_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_input_outputs_gates = tmp_ptr_array_delta_input_outputs_gates;
                    tmp_ptr_array_block_units[tmp_block_index].ptr_delta_recurrent_outputs_gates = tmp_ptr_array_delta_recurrent_outputs_gates;
                    
                    //        [2] Assign LSTM cells.
                    tmp_ptr_array_block_units[tmp_block_index].ptr_array_cell_units = tmp_ptr_array_cell_units;
                    
                    for(tmp_block_index_cell_index = 0_UZ; tmp_block_index_cell_index != tmp_number_cell_units_per_block; ++tmp_block_index_cell_index,
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
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_summation_cell_input = tmp_ptr_array_summation_cells_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_summation_input_cell_input = tmp_ptr_array_summation_input_cells_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_summation_recurrent_cell_input = tmp_ptr_array_summation_recurrent_cells_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_cell_input = tmp_ptr_array_cells_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_cell_state = tmp_ptr_array_cells_states;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_cell_state_activate = tmp_ptr_array_cells_states_activates;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_cell_output = tmp_ptr_array_cells_outputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_delta_cell_input = tmp_ptr_array_delta_cells_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_delta_cell_input_input = tmp_ptr_array_delta_cells_input_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_delta_cell_recurrent_input = tmp_ptr_array_delta_cells_recurrent_inputs;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_delta_cell_state = tmp_ptr_array_delta_cells_states;
                        tmp_ptr_array_cell_units[tmp_block_index_cell_index].ptr_delta_cell_output = tmp_ptr_array_delta_cells_outputs;
                    }

                    tmp_ptr_array_cell_units += tmp_number_cell_units_per_block;

                    tmp_ptr_array_block_units[tmp_block_index].ptr_last_cell_unit = tmp_ptr_array_cell_units;
                    //        [2] |END| Assign LSTM cells. |END|
                }
                
                tmp_ptr_array_summation_cells_inputs += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_input_cells_inputs += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_recurrent_cells_inputs += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_inputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_input_inputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_recurrent_inputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_forgets_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_input_forgets_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_recurrent_forgets_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_outputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_input_outputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_summation_recurrent_outputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_cells_inputs += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_cells_states += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_cells_states_activates += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_cells_outputs += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_inputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_forgets_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_outputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_cells_inputs += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_cells_input_inputs += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_cells_recurrent_inputs += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_cells_states += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_cells_outputs += (this->batch_size - 1_UZ) * tmp_number_cell_units * this->seq_w + tmp_number_cell_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_inputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_input_inputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_recurrent_inputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_forgets_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_input_forgets_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_recurrent_forgets_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_outputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_input_outputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                tmp_ptr_array_delta_recurrent_outputs_gates += (this->batch_size - 1_UZ) * tmp_number_block_units * this->seq_w + tmp_number_block_units * (this->seq_w - 1_UZ);
                //    [1] |END| Assign time step. |END|

                tmp_ptr_array_block_units += tmp_number_block_units;
            }

            layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;
            layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
            // [0] |END| Assign block units. |END|
        }
        
        this->ptr_last_block_unit = tmp_ptr_array_block_units;
        this->ptr_last_cell_unit = tmp_ptr_array_cell_units;

        this->total_block_units_allocated = this->total_block_units;
        this->total_cell_units_allocated = this->total_cell_units;
    }

    return true;
}

bool Model::Allocate__Bidirectional__Layers(void)
{
    size_t tmp_number_block_units_per_layer,
              tmp_number_cell_units_per_layer,
              tmp_number_bidirectional_layers(0_UZ);
    
    Layer const *const last_layer(this->ptr_last_layer);
    Layer *layer_it;
        
    BlockUnit *tmp_ptr_array_block_units;

    CellUnit *tmp_ptr_array_cell_units;

    for(layer_it = this->ptr_array_layers; layer_it != last_layer; ++layer_it)
    { if(layer_it->Use__Bidirectional()) { ++tmp_number_bidirectional_layers; } }

    if(tmp_number_bidirectional_layers != 0_UZ)
    {
        Bidirectional_Layer *tmp_ptr_array_bidirectional_layers(new Bidirectional_Layer[tmp_number_bidirectional_layers]);
        this->ptr_array_bidirectional_layers = tmp_ptr_array_bidirectional_layers;
            
        for(layer_it = this->ptr_array_layers; layer_it != last_layer; ++layer_it)
        {
            layer_it->ptr_Bidirectional_Layer = tmp_ptr_array_bidirectional_layers;

            if(layer_it->Use__Bidirectional())
            {
                //    [1] Forward layer.
                //        [2] Assign parameters.
                tmp_ptr_array_bidirectional_layers->forward_layer.type_layer = layer_it->type_layer;
                tmp_ptr_array_bidirectional_layers->forward_layer.type_activation = layer_it->type_activation;
                tmp_ptr_array_bidirectional_layers->forward_layer.type_dropout = layer_it->type_dropout;
                tmp_ptr_array_bidirectional_layers->forward_layer.type_normalization = layer_it->type_normalization;

                tmp_ptr_array_bidirectional_layers->forward_layer.ptr_number_outputs = layer_it->ptr_number_outputs;
                tmp_ptr_array_bidirectional_layers->forward_layer.ptr_first_connection_index = layer_it->ptr_first_connection_index;
                tmp_ptr_array_bidirectional_layers->forward_layer.ptr_last_connection_index = layer_it->ptr_last_connection_index;
                tmp_ptr_array_bidirectional_layers->forward_layer.first_bias_connection_index = layer_it->first_bias_connection_index;
                tmp_ptr_array_bidirectional_layers->forward_layer.last_bias_connection_index = layer_it->first_bias_connection_index + (layer_it->last_bias_connection_index - layer_it->first_bias_connection_index) / 2_UZ;

                tmp_ptr_array_bidirectional_layers->forward_layer.dropout_values[0] = layer_it->dropout_values[0];
                tmp_ptr_array_bidirectional_layers->forward_layer.dropout_values[1] = layer_it->dropout_values[1];
                tmp_ptr_array_bidirectional_layers->forward_layer.dropout_values[2] = layer_it->dropout_values[2];
                //        [2] |END| Assign parameters. |END|
                //    [1] |END| Forward layer. |END|
                    
                //    [1] Backward layer.
                //        [2] Assign parameters.
                tmp_ptr_array_bidirectional_layers->backward_layer.type_layer = layer_it->type_layer;
                tmp_ptr_array_bidirectional_layers->backward_layer.type_activation = layer_it->type_activation;
                tmp_ptr_array_bidirectional_layers->backward_layer.type_dropout = layer_it->type_dropout;
                tmp_ptr_array_bidirectional_layers->backward_layer.type_normalization = layer_it->type_normalization;

                tmp_ptr_array_bidirectional_layers->backward_layer.ptr_number_outputs = layer_it->ptr_number_outputs;
                tmp_ptr_array_bidirectional_layers->backward_layer.ptr_first_connection_index = layer_it->ptr_first_connection_index;
                tmp_ptr_array_bidirectional_layers->backward_layer.ptr_last_connection_index = layer_it->ptr_last_connection_index;
                tmp_ptr_array_bidirectional_layers->backward_layer.first_bias_connection_index = layer_it->first_bias_connection_index + (layer_it->last_bias_connection_index - layer_it->first_bias_connection_index) / 2_UZ;
                tmp_ptr_array_bidirectional_layers->backward_layer.last_bias_connection_index = layer_it->last_bias_connection_index;

                tmp_ptr_array_bidirectional_layers->backward_layer.dropout_values[0] = layer_it->dropout_values[0];
                tmp_ptr_array_bidirectional_layers->backward_layer.dropout_values[1] = layer_it->dropout_values[1];
                tmp_ptr_array_bidirectional_layers->backward_layer.dropout_values[2] = layer_it->dropout_values[2];
                //        [2] |END| Assign parameters. |END|
                //    [1] |END| Backward layer. |END|

                switch(layer_it->type_layer)
                {
                    case LAYER::LSTM:
                        //    [1] Forward layer.
                        //        [2] Assign block units.
                        tmp_number_block_units_per_layer = static_cast<size_t>(layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units) >> 1;
                        tmp_number_cell_units_per_layer = static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units) >> 1;

                        tmp_ptr_array_block_units = layer_it->ptr_array_block_units;
                        tmp_ptr_array_bidirectional_layers->forward_layer.ptr_array_block_units = tmp_ptr_array_block_units;
                        tmp_ptr_array_block_units += tmp_number_block_units_per_layer;
                        tmp_ptr_array_bidirectional_layers->forward_layer.ptr_last_block_unit = tmp_ptr_array_block_units;

                        tmp_ptr_array_cell_units = layer_it->ptr_array_cell_units;
                        tmp_ptr_array_bidirectional_layers->forward_layer.ptr_array_cell_units = tmp_ptr_array_cell_units;
                        tmp_ptr_array_cell_units += tmp_number_cell_units_per_layer;
                        tmp_ptr_array_bidirectional_layers->forward_layer.ptr_last_cell_unit = tmp_ptr_array_cell_units;
                        //        [2] |END| Assign block units. |END|
                        //    [1] |END| Forward layer. |END|

                        //    [1] Backward layer.
                        //        [2] Assign block units.
                        tmp_ptr_array_bidirectional_layers->backward_layer.ptr_array_block_units = tmp_ptr_array_block_units;
                        tmp_ptr_array_block_units += tmp_number_block_units_per_layer;
                        tmp_ptr_array_bidirectional_layers->backward_layer.ptr_last_block_unit = tmp_ptr_array_block_units;

                        tmp_ptr_array_bidirectional_layers->backward_layer.ptr_array_cell_units = tmp_ptr_array_cell_units;
                        tmp_ptr_array_cell_units += tmp_number_cell_units_per_layer;
                        tmp_ptr_array_bidirectional_layers->backward_layer.ptr_last_cell_unit = tmp_ptr_array_cell_units;
                        //        [2] |END| Assign block units. |END|
                        //    [1] |END| Backward layer. |END|
                            break;
                    default:
                        ERR(L"Layer type (%d | %ls) is not managed in the switch.",
                                                 layer_it->type_layer,
                                                 LAYER_NAME[layer_it->type_layer].c_str());
                        return false;
                }
                    
                ++tmp_ptr_array_bidirectional_layers;
            }
        }
            
        this->ptr_last_bidirectional_layer = tmp_ptr_array_bidirectional_layers;
    }

    return true;
}

bool Model::Allocate__Parameter(void)
{
  this->ptr_array_parameters = new var[this->total_parameters]();
  this->ptr_array_ptr_connections = new void *[this->total_parameters]();

    this->total_parameters_allocated = this->total_parameters;

    this->total_weights_allocated = this->total_weights;

    this->total_bias_allocated = this->total_bias;

    return true;
}

bool Model::Allocate__Parameter__Optimizer(void)
{
    switch(this->type_optimizer_function)
    {
        case OPTIMIZER::GD: return(this->Allocate__Parameter__Gradient_Descent());
        case OPTIMIZER::IRPROP_MINUS: return(this->Allocate__Parameter__iRPROP_minus());
        case OPTIMIZER::IRPROP_PLUS: return(this->Allocate__Parameter__iRPROP_plus());
        case OPTIMIZER::ADABOUND:
        case OPTIMIZER::ADAM:
        case OPTIMIZER::ADAMAX:
        case OPTIMIZER::NOSADAM: return(this->Allocate__Parameter__Adam());
        case OPTIMIZER::AMSBOUND:
        case OPTIMIZER::AMSGRAD: return(this->Allocate__Parameter__AMSGrad());
        default:
            ERR(L"Can not allocate parameters of the optimizer (%u | %ls).",
                                     this->type_optimizer_function,
                                     OPTIMIZER_NAME[this->type_optimizer_function].c_str());
                return false;
    }
}

bool Model::Allocate__Parameter__Gradient_Descent(void)
{
    if(this->learning_momentum != 0_r
      &&
      this->ptr_array_previous_delta_parameters == nullptr)
    {
    this->ptr_array_previous_delta_parameters =
        new real[this->total_parameters]();
    }

    return true;
}

bool Model::Allocate__Parameter__iRPROP_minus(void)
{
    if(this->ptr_array_previous_steps == nullptr)
    {
    this->ptr_array_previous_steps = new real[this->total_parameters];
        Mem::fill(this->ptr_array_previous_steps,
                                   this->ptr_array_previous_steps + this->total_parameters,
                                   this->rprop_delta_zero);
    }
    
    if(this->ptr_array_previous_derivatives_parameters == nullptr)
    {
      this->ptr_array_previous_derivatives_parameters =
          new real[this->total_parameters]();
    }

    return true;
}

bool Model::Allocate__Parameter__iRPROP_plus(void)
{
    if(this->Allocate__Parameter__iRPROP_minus() == false)
    {
        ERR(L"An error has been triggered from the \"Allocate__Parameter__iRPROP_minus()\" function.",);

        return false;
    }

    if(this->ptr_array_previous_delta_parameters == nullptr)
    {
      this->ptr_array_previous_delta_parameters =
          new real[this->total_parameters]();
    }

    return true;
}

bool Model::Allocate__Parameter__Adam(void)
{
    if(this->ptr_array_previous_biased_first_moment == nullptr)
    {
    this->ptr_array_previous_biased_first_moment =
          new real[this->total_parameters]();
    }
    
    if(this->ptr_array_previous_biased_second_moment == nullptr)
    {
      this->ptr_array_previous_biased_second_moment =
          new real[this->total_parameters]();
    }

    return true;
}

bool Model::Allocate__Parameter__AMSGrad(void)
{
    if(this->Allocate__Parameter__Adam() == false)
    {
        ERR(L"An error has been triggered from the \"Allocate__Parameter__Adam()\" function.",);

        return false;
    }

    if(this->ptr_array_previous_biased_second_moment_hat == nullptr)
    {
      this->ptr_array_previous_biased_second_moment_hat =
          new real[this->total_parameters]();
    }

    return true;
}

bool Model::Allocate__Parameter__Normalization(void)
{
    // TODO: Reorganasition of the array. [------Weights-----][----Bias----][----Normalized unit----]. Allocating with the size of each layer. No waste of memory.
    if(this->ptr_array_parameters != nullptr)
    {
        // Parameters + ((Scale + shift)=2) * NormUnits.
        size_t const tmp_new_dimension_parameters(this->total_parameters_allocated + 2_UZ * this->total_normalized_units);
        
        if(this->Reallocate__Parameter(tmp_new_dimension_parameters) == false)
        {
            ERR(L"An error has been triggered from the \"Reallocate__Parameter(%zu)\" function.",
                                     tmp_new_dimension_parameters);

            return false;
        }

        // clear shift array.
        VARZERO(this->ptr_array_parameters + this->total_weights_allocated + this->total_bias_allocated + this->total_normalized_units,
                       this->total_normalized_units * sizeof(var));
    }
    else { return false; }

    return true;
}

bool Model::Allocate__Parameter__Regularization(void) {
  if (this->ptr_array_mask_regularized_parameters == nullptr) {
    this->ptr_array_mask_regularized_parameters =
        new real[this->total_parameters_allocated]();
    Mem::fill(this->ptr_array_mask_regularized_parameters,
              this->ptr_array_mask_regularized_parameters +
                  this->total_weights_allocated + this->total_bias_allocated,
              1_r);

    if (this->total_weights_allocated + this->total_bias_allocated <
        this->total_parameters_allocated) {
      memset(this->ptr_array_mask_regularized_parameters +
                 this->total_weights_allocated + this->total_bias_allocated,
             0,
             (this->total_parameters_allocated - this->total_weights_allocated -
              this->total_bias_allocated) *
                 sizeof(real));
    }
  }

  return true;
}

bool Model::Allocate__Normalized_Unit__Batch_Normalization(void)
{
    if(this->ptr_array_normalized_batch_units_values_hats == nullptr
      &&
      this->ptr_array_normalized_batch_units_values_normalizes == nullptr
      &&
      this->ptr_array_normalized_batch_units_means == nullptr
      &&
      this->ptr_array_normalized_batch_units_variances == nullptr
      &&
      this->ptr_array_normalized_batch_units_derivatives_means == nullptr
      &&
      this->ptr_array_normalized_batch_units_derivatives_variances == nullptr
      &&
      this->ptr_array_normalized_batch_units_means_averages == nullptr
      &&
      this->ptr_array_normalized_batch_units_variances_averages == nullptr
      &&
      this->ptr_array_normalized_batch_units_errors == nullptr)
    {
        size_t tmp_number_units,
                  tmp_index;
        
        void **tmp_ptr_array_ptr_connections(this->ptr_array_ptr_connections + this->total_weights_allocated + this->total_bias_allocated);

        var *tmp_ptr_array_parameters_scale_it(this->ptr_array_parameters +
                                               this->total_weights_allocated +
                                               this->total_bias_allocated),
            *tmp_ptr_array_parameters_shift_it(
                this->ptr_array_parameters + this->total_weights_allocated +
                this->total_bias_allocated +
                this->total_normalized_units_allocated);
        real *tmp_ptr_array_derivatives_parameters_scale_it(
            this->ptr_array_derivatives_parameters +
            this->total_weights_allocated + this->total_bias_allocated),
            *tmp_ptr_array_derivatives_parameters_shift_it(
                this->ptr_array_derivatives_parameters +
                this->total_weights_allocated + this->total_bias_allocated +
                this->total_normalized_units_allocated);
        
        Layer const *const last_layer(this->ptr_last_layer);
        Layer *layer_it(this->ptr_array_layers);
        
        BlockUnit const *tmp_ptr_last_block_unit;
        BlockUnit *tmp_ptr_block_unit_it;
        
        CellUnit const *tmp_ptr_last_cell_unit;
        CellUnit *tmp_ptr_cell_unit_it;
        
        union Normalized_unit const *tmp_ptr_last_normalized_unit;
        union Normalized_unit *tmp_ptr_normalized_unit_it;
        
        var *tmp_ptr_array_normalized_units_values_hat(new var[this->batch_size * this->seq_w * this->total_normalized_units_allocated]());
        this->ptr_array_normalized_batch_units_values_hats = tmp_ptr_array_normalized_units_values_hat;
        
        var *tmp_ptr_array_normalized_units_values_normalize(new var[this->batch_size * this->seq_w * this->total_normalized_units_allocated]());
        this->ptr_array_normalized_batch_units_values_normalizes = tmp_ptr_array_normalized_units_values_normalize;
        
        var *tmp_ptr_array_normalized_units_mean_it(new var[this->number_threads * this->total_normalized_units_allocated * this->seq_w]());
        this->ptr_array_normalized_batch_units_means = tmp_ptr_array_normalized_units_mean_it;
        
        var *tmp_ptr_array_normalized_units_variance_it(new var[this->number_threads * this->total_normalized_units_allocated * this->seq_w]());
        this->ptr_array_normalized_batch_units_variances = tmp_ptr_array_normalized_units_variance_it;
        
        real *tmp_ptr_array_normalized_units_derivative_mean_it(
            new real[this->number_threads *
                     this->total_normalized_units_allocated * this->seq_w]());
        this->ptr_array_normalized_batch_units_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it;

        real *tmp_ptr_array_normalized_units_derivative_variance_it(
            new real[this->number_threads *
                    this->total_normalized_units_allocated * this->seq_w]());
        this->ptr_array_normalized_batch_units_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it;

        var *tmp_ptr_array_normalized_units_mean_average_it(
            new var[this->total_normalized_units_allocated * this->seq_w]());
        this->ptr_array_normalized_batch_units_means_averages = tmp_ptr_array_normalized_units_mean_average_it;

        var *tmp_ptr_array_normalized_units_variance_average_it(
            new var[this->total_normalized_units_allocated * this->seq_w]);
        Mem::fill<var>(tmp_ptr_array_normalized_units_variance_average_it,
                                   tmp_ptr_array_normalized_units_variance_average_it + this->total_normalized_units_allocated * this->seq_w,
                                   1_r);
        this->ptr_array_normalized_batch_units_variances_averages = tmp_ptr_array_normalized_units_variance_average_it;
        
        real *tmp_ptr_array_normalized_units_errors(
            new real[this->batch_size * this->seq_w *
                     this->total_normalized_units_allocated]());
        this->ptr_array_normalized_batch_units_errors = tmp_ptr_array_normalized_units_errors;
        
        this->ptr_array_normalized_batch_units_scales = tmp_ptr_array_parameters_scale_it;
        this->ptr_array_normalized_batch_units_shifts = tmp_ptr_array_parameters_shift_it;
        
        this->ptr_array_normalized_batch_units_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it;
        this->ptr_array_normalized_batch_units_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it;
        
        for(; layer_it != last_layer; ++layer_it)
        {
            if((tmp_number_units = static_cast<size_t>(layer_it->ptr_last_normalized_unit - layer_it->ptr_array_normalized_units)) != 0_UZ)
            {
                // Initialize values.
                switch(layer_it->type_layer)
                {
                    case LAYER::FULLY_CONNECTED:
                    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case LAYER::FULLY_CONNECTED_RECURRENT:
                    case LAYER::RESIDUAL:
                        for(tmp_ptr_last_normalized_unit = layer_it->ptr_last_normalized_unit,
                            tmp_ptr_normalized_unit_it = layer_it->ptr_array_normalized_units; tmp_ptr_normalized_unit_it != tmp_ptr_last_normalized_unit; ++tmp_ptr_normalized_unit_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_parameters_scale_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_parameters_shift_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_derivatives_parameters_scale_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_derivatives_parameters_shift_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_values_hat,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_values_normalize,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_mean_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_variance_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_derivative_mean_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_derivative_variance_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_mean_average_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_variance_average_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_errors,
                                                                                                                                                                                                                                   ++tmp_ptr_array_ptr_connections)
                        {
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_scale = tmp_ptr_array_parameters_scale_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_shift = tmp_ptr_array_parameters_shift_it;

                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it;
                            
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_values_hats = tmp_ptr_array_normalized_units_values_hat;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_values_normalizes = tmp_ptr_array_normalized_units_values_normalize;
                            
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it;
                            
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it;
                            
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_mean_average = tmp_ptr_array_normalized_units_mean_average_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_variance_average = tmp_ptr_array_normalized_units_variance_average_it;

                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors;

                            tmp_ptr_array_ptr_connections[0] = tmp_ptr_normalized_unit_it;
                            tmp_ptr_array_ptr_connections[this->total_normalized_units_allocated] = tmp_ptr_normalized_unit_it;
                        }
                        
                        tmp_ptr_array_normalized_units_values_hat += (this->batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                        tmp_ptr_array_normalized_units_values_normalize += (this->batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                        
                        tmp_ptr_array_normalized_units_mean_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                        tmp_ptr_array_normalized_units_variance_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                        
                        tmp_ptr_array_normalized_units_derivative_mean_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                        tmp_ptr_array_normalized_units_derivative_variance_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                        
                        tmp_ptr_array_normalized_units_mean_average_it += tmp_number_units * (this->seq_w - 1_UZ);
                        tmp_ptr_array_normalized_units_variance_average_it += tmp_number_units * (this->seq_w - 1_UZ);

                        tmp_ptr_array_normalized_units_errors += (this->batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
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
                                
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_scale = tmp_ptr_array_parameters_scale_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_shift = tmp_ptr_array_parameters_shift_it++;

                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it++;
                                
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it++;
                                
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it++;
                                
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_mean_average = tmp_ptr_array_normalized_units_mean_average_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_variance_average = tmp_ptr_array_normalized_units_variance_average_it++;
                                    
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors++;
                                
                                    tmp_ptr_array_ptr_connections[0] = tmp_ptr_cell_unit_it;
                                    tmp_ptr_array_ptr_connections[this->total_normalized_units_allocated] = tmp_ptr_cell_unit_it;
                                    ++tmp_ptr_array_ptr_connections;
                                }
                            
                                tmp_ptr_array_normalized_units_values_hat += (this->batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_values_normalize += (this->batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            
                                tmp_ptr_array_normalized_units_mean_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_variance_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            
                                tmp_ptr_array_normalized_units_derivative_mean_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_derivative_variance_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            
                                tmp_ptr_array_normalized_units_mean_average_it += tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_variance_average_it += tmp_number_units * (this->seq_w - 1_UZ);

                                tmp_ptr_array_normalized_units_errors += (this->batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
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
                                
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_scale = tmp_ptr_array_parameters_scale_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_shift = tmp_ptr_array_parameters_shift_it++;

                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it++;
                                
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it++;
                                
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it++;
                                
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_mean_average = tmp_ptr_array_normalized_units_mean_average_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_variance_average = tmp_ptr_array_normalized_units_variance_average_it++;
                                    
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_errors = tmp_ptr_array_normalized_units_errors++;

                                    tmp_ptr_array_ptr_connections[0] = tmp_ptr_block_unit_it;
                                    tmp_ptr_array_ptr_connections[this->total_normalized_units_allocated] = tmp_ptr_block_unit_it;
                                    ++tmp_ptr_array_ptr_connections;
                                }
                            
                                tmp_ptr_array_normalized_units_values_hat += (this->batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_values_normalize += (this->batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            
                                tmp_ptr_array_normalized_units_mean_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_variance_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            
                                tmp_ptr_array_normalized_units_derivative_mean_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_derivative_variance_it += (this->number_threads - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            
                                tmp_ptr_array_normalized_units_mean_average_it += tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_variance_average_it += tmp_number_units * (this->seq_w - 1_UZ);
                                
                                tmp_ptr_array_normalized_units_errors += (this->batch_size - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            }
                        } 
                            break;
                    default:
                        ERR(L"Type layer (%u | %ls) is not managed in the switch.",
                                                 layer_it->type_layer,
                                                 LAYER_NAME[layer_it->type_layer].c_str());
                            return false;
                }
                
                tmp_number_units = static_cast<size_t>(layer_it->ptr_last_normalized_unit - layer_it->ptr_array_normalized_units);

                // Initialize scale.
                switch(layer_it->type_layer)
                {
                    case LAYER::FULLY_CONNECTED:
                        Mem::fill<var>(layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale,
                                                  layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale + tmp_number_units,
                                                  1_r);
                            break;
                    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    case LAYER::FULLY_CONNECTED_RECURRENT:
                    case LAYER::LSTM:
                      Mem::fill<var>(
                          layer_it->ptr_array_normalized_units
                              ->normalized_batch_units.ptr_scale,
                                                  layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale + tmp_number_units,
                                                  0.1_r);
                            break;
                    case LAYER::RESIDUAL:
                      Mem::fill<var>(
                          layer_it->ptr_array_normalized_units
                              ->normalized_batch_units.ptr_scale,
                                                  layer_it->ptr_array_normalized_units->normalized_batch_units.ptr_scale + tmp_number_units,
                                                  this->seq_w == 1_UZ ? 1_r : 0.1_r);
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
    else { return false; }

    return true;
}

bool Model::Allocate__Normalized_Unit__Batch_Renormalization(void)
{
    if(this->ptr_array_normalized_batch_units_r_corrections == nullptr && this->ptr_array_normalized_batch_units_d_corrections == nullptr)
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
        
        var *tmp_ptr_array_normalized_units_r_correction_it(new var[this->seq_w * this->total_normalized_units_allocated]());
        var *tmp_ptr_array_normalized_units_d_correction_it(new var[this->seq_w * this->total_normalized_units_allocated]());
        
        this->ptr_array_normalized_batch_units_r_corrections = tmp_ptr_array_normalized_units_r_correction_it;
        this->ptr_array_normalized_batch_units_d_corrections = tmp_ptr_array_normalized_units_d_correction_it;
        
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
                                                                                                                                                                                                                                  ++tmp_ptr_array_normalized_units_r_correction_it,
                                                                                                                                                                                                                                  ++tmp_ptr_array_normalized_units_d_correction_it)
                        {
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_r_correction = tmp_ptr_array_normalized_units_r_correction_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_d_correction = tmp_ptr_array_normalized_units_d_correction_it;
                        }

                        tmp_ptr_array_normalized_units_r_correction_it += tmp_number_units * (this->seq_w - 1_UZ);
                        tmp_ptr_array_normalized_units_d_correction_it += tmp_number_units * (this->seq_w - 1_UZ);
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
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_r_correction = tmp_ptr_array_normalized_units_r_correction_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_d_correction = tmp_ptr_array_normalized_units_d_correction_it++;
                                }

                                tmp_ptr_array_normalized_units_r_correction_it += tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_d_correction_it += tmp_number_units * (this->seq_w - 1_UZ);
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
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_r_correction = tmp_ptr_array_normalized_units_r_correction_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_d_correction = tmp_ptr_array_normalized_units_d_correction_it++;
                                }

                                tmp_ptr_array_normalized_units_r_correction_it += tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_d_correction_it += tmp_number_units * (this->seq_w - 1_UZ);
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
    else { return false; }

    return true;
}
}
