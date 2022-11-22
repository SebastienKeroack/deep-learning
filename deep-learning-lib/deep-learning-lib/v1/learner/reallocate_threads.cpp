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
bool Model::Reallocate__Thread(size_t const number_threads_received)
{
    if(this->Reallocate__Thread__Cost(number_threads_received) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Thread__Cost(%zu)\" function.",
                                 number_threads_received);

        return false;
    }
    else if(this->Reallocate__Thread__Normalized_Unit__Batch_Normalization(number_threads_received) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Thread__Normalized_Unit__Batch_Normalization(%zu)\" function.",
                                 number_threads_received);

        return false;
    }
    else if(this->Reallocate__Thread__Parameter(number_threads_received) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Thread__Parameter(%zu)\" function.",
                                 number_threads_received);

        return false;
    }
    else if(this->Use__K_Sparse() && this->Reallocate__Thread__Sparse_K_Filter(number_threads_received) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Thread__Sparse_K_Filter(%zu)\" function.",
                                 number_threads_received);

        return false;
    }
    else if((this->Use__Dropout__Bernoulli() || this->Use__Dropout__Bernoulli__Inverted()) && this->Reallocate__Thread__Generator__Dropout__Bernoulli(number_threads_received) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Thread__Generator__Dropout__Bernoulli(%zu)\" function.",
                                 number_threads_received);

        return false;
    }
    else if(this->Use__Dropout__Gaussian() && this->Reallocate__Thread__Generator__Dropout__Gaussian(number_threads_received) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Thread__Generator__Dropout__Gaussian(%zu)\" function.",
                                 number_threads_received);

        return false;
    }
    else if(this->Use__Dropout__ShakeDrop() && this->Reallocate__Thread__Generator__Dropout__ShakeDrop(number_threads_received) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Thread__Generator__Dropout__ShakeDrop(%zu)\" function.",
                                 number_threads_received);

        return false;
    }
    else if(this->Use__Dropout__Uout() && this->Reallocate__Thread__Generator__Dropout__Uout(number_threads_received) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Thread__Generator__Dropout__Uout(%zu)\" function.",
                                 number_threads_received);

        return false;
    }
    else if(this->Use__Dropout__Zoneout() && this->Reallocate__Thread__Generator__Dropout__Zoneout(number_threads_received) == false)
    {
        ERR(L"An error has been triggered from the \"Reallocate__Thread__Generator__Dropout__Zoneout(%zu)\" function.",
                                 number_threads_received);

        return false;
    }

    return true;
}

bool Model::Reallocate__Thread__Sparse_K_Filter(size_t const number_threads_received)
{
    this->ptr_array_k_sparse_activities = Mem::reallocate_obj<std::pair<size_t, var>, false>(this->ptr_array_k_sparse_activities,
                                                                                                                                            number_threads_received * (this->total_basic_units_allocated + this->total_basic_indice_units_allocated + this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated + this->total_cell_units_allocated),
                                                                                                                                            this->number_threads * (this->total_basic_units_allocated + this->total_basic_indice_units_allocated + this->total_AF_units_allocated + this->total_AF_Ind_recurrent_units_allocated + this->total_cell_units_allocated));
    this->Assign__Sparsity_Activities(number_threads_received);

    return true;
}

bool Model::Reallocate__Thread__Cost(size_t const number_threads_received)
{        
    // reallocate number loss.
    size_t *tmp_ptr_array_number_loss(Mem::reallocate<size_t, false>(this->ptr_array_number_loss,
                                                                                                         number_threads_received,
                                                                                                         this->number_threads));
    this->ptr_array_number_loss = tmp_ptr_array_number_loss;
    // |END| reallocate number loss. |END|
        
    // reallocate number loss.
    size_t *tmp_ptr_array_bit_fail_values(Mem::reallocate<size_t, false>(this->ptr_array_number_bit_fail,
                                                                                                          number_threads_received,
                                                                                                          this->number_threads));
    this->ptr_array_number_bit_fail = tmp_ptr_array_bit_fail_values;
    // |END| reallocate number loss. |END|
    
    // reallocate loss values.
    double *tmp_ptr_array_loss_values(Mem::reallocate(this->ptr_array_loss_values,
                                                                                             number_threads_received,
                                                                                             this->number_threads));
    this->ptr_array_loss_values = tmp_ptr_array_loss_values;
    // |END| reallocate loss values. |END|

    for (int i(0); i != 5; ++i)
      this->ptr_array_accuracy_values[i] =
          Mem::reallocate(this->ptr_array_accuracy_values[i],
                          number_threads_received, this->number_threads);
    
    return true;
}

bool Model::Reallocate__Thread__Normalized_Unit__Batch_Normalization(size_t const number_threads_received)
{
    if(this->Use__Normalization()
      &&
      this->ptr_array_normalized_batch_units_means != nullptr
      &&
      this->ptr_array_normalized_batch_units_variances != nullptr
      &&
      this->ptr_array_normalized_batch_units_derivatives_means != nullptr
      &&
      this->ptr_array_normalized_batch_units_derivatives_variances != nullptr)
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
        
        var *tmp_ptr_array_normalized_units_mean_it(Mem::reallocate(this->ptr_array_normalized_batch_units_means,
                                                                                                                     number_threads_received * this->seq_w * this->total_normalized_units_allocated,
                                                                                                                     this->number_threads * this->seq_w * this->total_normalized_units_allocated));
        var *tmp_ptr_array_normalized_units_variance_it(Mem::reallocate(this->ptr_array_normalized_batch_units_variances,
                                                                                                                         number_threads_received * this->seq_w * this->total_normalized_units_allocated,
                                                                                                                         this->number_threads * this->seq_w * this->total_normalized_units_allocated));
        real *tmp_ptr_array_normalized_units_derivative_mean_it(Mem::reallocate(this->ptr_array_normalized_batch_units_derivatives_means,
                                                                                                                                    number_threads_received * this->seq_w * this->total_normalized_units_allocated,
                                                                                                                                    this->number_threads * this->seq_w * this->total_normalized_units_allocated));
        real *tmp_ptr_array_normalized_units_derivative_variance_it(
            Mem::reallocate(
                this->ptr_array_normalized_batch_units_derivatives_variances,
                                                                                                                                        number_threads_received * this->seq_w * this->total_normalized_units_allocated,
                                                                                                                                        this->number_threads * this->seq_w * this->total_normalized_units_allocated));
        
        this->ptr_array_normalized_batch_units_means = tmp_ptr_array_normalized_units_mean_it;
        this->ptr_array_normalized_batch_units_variances = tmp_ptr_array_normalized_units_variance_it;
        this->ptr_array_normalized_batch_units_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it;
        this->ptr_array_normalized_batch_units_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it;
        
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
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_mean_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_variance_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_derivative_mean_it,
                                                                                                                                                                                                                                   ++tmp_ptr_array_normalized_units_derivative_variance_it)
                        {
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it;
                            
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it;
                            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it;
                        }

                        tmp_ptr_array_normalized_units_mean_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                        tmp_ptr_array_normalized_units_variance_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                        
                        tmp_ptr_array_normalized_units_derivative_mean_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                        tmp_ptr_array_normalized_units_derivative_variance_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
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
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it++;
                                
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it++;
                                    tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it++;
                                }
                            
                                tmp_ptr_array_normalized_units_mean_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_variance_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            
                                tmp_ptr_array_normalized_units_derivative_mean_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_derivative_variance_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
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
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_means = tmp_ptr_array_normalized_units_mean_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_variances = tmp_ptr_array_normalized_units_variance_it++;
                                    
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_means = tmp_ptr_array_normalized_units_derivative_mean_it++;
                                    tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index].normalized_batch_units.ptr_array_derivatives_variances = tmp_ptr_array_normalized_units_derivative_variance_it++;
                                }
                            
                                tmp_ptr_array_normalized_units_mean_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_variance_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                            
                                tmp_ptr_array_normalized_units_derivative_mean_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
                                tmp_ptr_array_normalized_units_derivative_variance_it += (number_threads_received - 1_UZ) * tmp_number_units * this->seq_w + tmp_number_units * (this->seq_w - 1_UZ);
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

bool Model::Reallocate__Thread__Parameter(size_t const number_threads_received)
{
    if(this->total_parameters_allocated != 0_UZ)
    {
        // Derivates parameters.
        if(this->ptr_array_derivatives_parameters != nullptr)
        {
          this->ptr_array_derivatives_parameters = Mem::reallocate(
              this->ptr_array_derivatives_parameters,
              number_threads_received * this->total_parameters_allocated,
              this->number_threads * this->total_parameters_allocated);

            if(this->Use__Normalization()) { this->Reset__Derivative_Parameter__Normalized_Unit(); }
        }
        // |END| Derivates parameters. |END|
    }

    return true;
}

bool Model::Reallocate__Thread__Generator__Dropout__Bernoulli(size_t const number_threads_received)
{
    if(this->bernoulli != nullptr)
    {
        Dist::Bernoulli *tmp_ptr_array_Class_Generator_Random_Bernoulli(Mem::reallocate_obj<Dist::Bernoulli>(this->bernoulli,
                                                                                                                                                                                                                                                                                                                                      number_threads_received,
                                                                                                                                                                                                                                                                                                                                      this->number_threads));
        this->bernoulli = tmp_ptr_array_Class_Generator_Random_Bernoulli;

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        { tmp_ptr_array_Class_Generator_Random_Bernoulli[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    return true;
}

bool Model::Reallocate__Thread__Generator__Dropout__Zoneout(size_t const number_threads_received)
{
    if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_State != nullptr)
    {
        Dist::Bernoulli *tmp_ptr_array_Class_Generator_Random_Zoneout(Mem::reallocate_obj<Dist::Bernoulli>(this->ptr_array_Class_Generator_Bernoulli_Zoneout_State,
                                                                                                                                                                                                                                                                                                                                     number_threads_received,
                                                                                                                                                                                                                                                                                                                                     this->number_threads));
        this->ptr_array_Class_Generator_Bernoulli_Zoneout_State = tmp_ptr_array_Class_Generator_Random_Zoneout;

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        { tmp_ptr_array_Class_Generator_Random_Zoneout[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }
    
    if(this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden != nullptr)
    {
        Dist::Bernoulli *tmp_ptr_array_Class_Generator_Random_Hidden(Mem::reallocate_obj<Dist::Bernoulli>(this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden,
                                                                                                                                                                                                                                                                                                                                    number_threads_received,
                                                                                                                                                                                                                                                                                                                                    this->number_threads));
        this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden = tmp_ptr_array_Class_Generator_Random_Hidden;

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        { tmp_ptr_array_Class_Generator_Random_Hidden[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    return true;
}

bool Model::Reallocate__Thread__Generator__Dropout__Gaussian(size_t const number_threads_received)
{
    if(this->ptr_array_Class_Generator_Real_Gaussian != nullptr)
    {
        Dist::Gaussian *tmp_ptr_array_Class_Generator_Random_Gaussian(Mem::reallocate_obj<Dist::Gaussian>(this->ptr_array_Class_Generator_Real_Gaussian,
                                                                                                                                                                                                                                                                                                                                          number_threads_received,
                                                                                                                                                                                                                                                                                                                                          this->number_threads));
        this->ptr_array_Class_Generator_Real_Gaussian = tmp_ptr_array_Class_Generator_Random_Gaussian;

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        {
            tmp_ptr_array_Class_Generator_Random_Gaussian[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            tmp_ptr_array_Class_Generator_Random_Gaussian[tmp_generator_index].range(0_r, 1_r);
        }
    }

    return true;
}

bool Model::Reallocate__Thread__Generator__Dropout__ShakeDrop(size_t const number_threads_received)
{
    if(this->ptr_array_Class_Generator_Bernoulli_ShakeDrop != nullptr)
    {
        Dist::Bernoulli *tmp_ptr_array_Class_Generator_Random_Bernoulli_ShakeDrop(Mem::reallocate_obj<Dist::Bernoulli>(this->ptr_array_Class_Generator_Bernoulli_ShakeDrop,
                                                                                                                                                                                                                                                                                                                                                        number_threads_received,
                                                                                                                                                                                                                                                                                                                                                        this->number_threads));
        this->ptr_array_Class_Generator_Bernoulli_ShakeDrop = tmp_ptr_array_Class_Generator_Random_Bernoulli_ShakeDrop;

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        { tmp_ptr_array_Class_Generator_Random_Bernoulli_ShakeDrop[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index)); }
    }

    if(this->ptr_array_Class_Generator_Real_ShakeDrop != nullptr)
    {
        Dist::Real *tmp_ptr_array_Class_Generator_Random_ShakeDrop(Mem::reallocate_obj<Dist::Real>(this->ptr_array_Class_Generator_Real_ShakeDrop,
                                                                                                                                                                                                                                                                                                                              number_threads_received,
                                                                                                                                                                                                                                                                                                                              this->number_threads));
        this->ptr_array_Class_Generator_Real_ShakeDrop = tmp_ptr_array_Class_Generator_Random_ShakeDrop;

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        {
            tmp_ptr_array_Class_Generator_Random_ShakeDrop[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            tmp_ptr_array_Class_Generator_Random_ShakeDrop[tmp_generator_index].range(0_r, 1_r);
        }
    }

    return true;
}

bool Model::Reallocate__Thread__Generator__Dropout__Uout(size_t const number_threads_received)
{
    if(this->ptr_array_Class_Generator_Real_Uout != nullptr)
    {
        Dist::Real *tmp_ptr_array_Class_Generator_Random_Uout(Mem::reallocate_obj<Dist::Real>(this->ptr_array_Class_Generator_Real_Uout,
                                                                                                                                                                                                                                                                                                                    number_threads_received,
                                                                                                                                                                                                                                                                                                                    this->number_threads));
        this->ptr_array_Class_Generator_Real_Uout = tmp_ptr_array_Class_Generator_Random_Uout;

        for(size_t tmp_generator_index(0_UZ); tmp_generator_index != number_threads_received; ++tmp_generator_index)
        {
            tmp_ptr_array_Class_Generator_Random_Uout[tmp_generator_index].seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + static_cast<unsigned int>(tmp_generator_index));
            tmp_ptr_array_Class_Generator_Random_Uout[tmp_generator_index].range(0_r, 1_r);
        }
    }

    return true;
}
}
