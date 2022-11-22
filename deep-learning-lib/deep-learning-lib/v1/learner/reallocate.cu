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

#include "deep-learning-lib/v1/learner/model.cuh"
#include "deep-learning-lib/ops/fill.cuh"

__device__ bool cuModel::Reallocate__Thread(size_t const number_threads_received)
{
    if(this->Reallocate__Thread__Cost(number_threads_received) == false)
    {
        ERR(L"From \"Reallocate__Thread__Cost\"",);

        return false;
    }
    else if(this->Reallocate_Reduce_Threads(number_threads_received) == false)
    {
        ERR(L"From \"Reallocate_Reduce_Threads\"",);

        return false;
    }
    else if(this->Reallocate__Thread__Parameter(number_threads_received) == false)
    {
        ERR(L"From \"Reallocate__Thread__Parameter\"",);

        return false;
    }

    this->Prepare__Threads__Grids_Blocks_Dimensions(number_threads_received);

    return true;
}

__device__ bool cuModel::Reallocate__Batch(size_t const batch_size)
{
    if(this->Reallocate__Batch__Neuron_Unit(batch_size) == false)
    {
        ERR(L"From \"Reallocate__Batch__Neuron_Unit\"",);

        return false;
    }
    else if(this->Reallocate__Batch__Neuron_Reduce_Summation(batch_size) == false)
    {
        ERR(L"From \"Reallocate__Batch__Neuron_Reduce_Summation\"",);

        return false;
    }
    else if(this->Reallocate__Batch__Neuron_Reduce_Error(batch_size) == false)
    {
        ERR(L"From \"Reallocate__Batch__Neuron_Reduce_Error\"",);

        return false;
    }
    else if(this->Reallocate__Normalized_Unit__Batch_Normalization(batch_size) == false)
    {
        ERR(L"From \"Reallocate__Normalized_Unit__Batch_Normalization\"",);

        return false;
    }
    else if(this->Reallocate__Batch__Neuron_Batch_Normalization_Transpose(batch_size) == false)
    {
        ERR(L"From \"Reallocate__Normalized_Unit__Batch_Normalization\"",);

        return false;
    }
    else if(this->Reallocate__Batch__Neuron_Batch_Normalization_Reduce(batch_size) == false)
    {
        ERR(L"From \"Reallocate__Batch__Neuron_Batch_Normalization_Reduce\"",);

        return false;
    }

    this->Prepare__Batch__Grids_Blocks_Dimensions(batch_size);

    return true;
}

__device__ bool cuModel::Reallocate__Thread__Cost(size_t const number_threads_received)
{
    // reallocate loss values.
    var *tmp_ptr_array_loss_values(Memory::reallocate_cpp<var>(this->ptr_array_loss_values,
                                                                                             number_threads_received,
                                                                                             this->number_threads,
                                                                                             false));
    if(tmp_ptr_array_loss_values == nullptr)
    {
        ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                 sizeof(var),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return false;
    }
    this->ptr_array_loss_values = tmp_ptr_array_loss_values;
    // |END| reallocate loss values. |END|
    
    // reallocate number loss.
    size_t *tmp_ptr_array_number_loss(Memory::reallocate_cpp<size_t>(this->ptr_array_number_loss,
                                                                                                         number_threads_received,
                                                                                                         this->number_threads,
                                                                                                         false));
    if(tmp_ptr_array_number_loss == nullptr)
    {
        ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                 sizeof(size_t),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return false;
    }
    this->ptr_array_number_loss = tmp_ptr_array_number_loss;
    // |END| reallocate number loss. |END|
    
    // reallocate number loss.
    size_t *tmp_ptr_array_bit_fail_values(Memory::reallocate_cpp<size_t>(this->ptr_array_number_bit_fail,
                                                                                                          number_threads_received,
                                                                                                          this->number_threads,
                                                                                                          false));
    if(tmp_ptr_array_bit_fail_values == nullptr)
    {
        ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                 sizeof(size_t),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return false;
    }
    this->ptr_array_number_bit_fail = tmp_ptr_array_bit_fail_values;
    // |END| reallocate number loss. |END|
    
    // reallocate number accuracy value.
    var *tmp_ptr_array_number_accuracy_value(Memory::reallocate_cpp<var>(this->ptr_array_accuracy_values[0],
                                                                                                               number_threads_received,
                                                                                                               this->number_threads,
                                                                                                               false));
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                 sizeof(var),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return false;
    }
    this->ptr_array_accuracy_values[0] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<var>(this->ptr_array_accuracy_values[1],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                 sizeof(var),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return false;
    }
    this->ptr_array_accuracy_values[1] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<var>(this->ptr_array_accuracy_values[2],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                 sizeof(var),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return false;
    }
    this->ptr_array_accuracy_values[2] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<var>(this->ptr_array_accuracy_values[3],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                 sizeof(var),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return false;
    }
    this->ptr_array_accuracy_values[3] = tmp_ptr_array_number_accuracy_value;

    tmp_ptr_array_number_accuracy_value = Memory::reallocate_cpp<var>(this->ptr_array_accuracy_values[4],
                                                                                                            number_threads_received,
                                                                                                            this->number_threads,
                                                                                                            false);
    if(tmp_ptr_array_number_accuracy_value == nullptr)
    {
        ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                 sizeof(var),
                                 number_threads_received,
                                 this->number_threads,
                                 __LINE__);

        return false;
    }
    this->ptr_array_accuracy_values[4] = tmp_ptr_array_number_accuracy_value;
    // |END| reallocate number accuracy value. |END|
    
    return true;
}

__device__ bool cuModel::Reallocate_Reduce_Threads(size_t const number_threads_received)
{
    if(this->total_reduce_batch_size != 0u)
    {
        size_t tmp_total_elements_to_reduce;

        class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;

        // Compute dimension reduce data batch.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                              0,
                                                                                              tmp_dim3_grid,
                                                                                              tmp_dim3_block);
        
        // Get remaining elements to reduce.
        tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            ERR(L"No elements to reduce.",);

            return false;
        }
        // |END| Compute dimension reduce data batch. |END|

        if(this->Reallocate_Reduce_Cost(tmp_total_elements_to_reduce) == false)
        {
            ERR(L"From \"Reallocate_Reduce_Cost\"",);

            return false;
        }
        else if(this->Reallocate_Reduce_Threads_Dim(number_threads_received) == false)
        {
            ERR(L"From \"Reallocate_Reduce_Threads_Dim\"",);

            return false;
        }

        this->total_reduce_batch_size = tmp_total_elements_to_reduce;
        
        // Compute dimension reduce data batch dynamic parallelisme.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_Dynamic_Parallelisme(tmp_total_elements_to_reduce,
                                                                                                            0,
                                                                                                            tmp_dim3_grid,
                                                                                                            tmp_dim3_block);
        
        // Get remaining elements to reduce.
        tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            ERR(L"No elements to reduce.",);

            return false;
        }
        // |END| Compute dimension reduce data batch dynamic parallelisme. |END|

        if(this->Reallocate_Reduce_Threads_Dim_DP(number_threads_received) == false)
        {
            ERR(L"From \"Reallocate_Reduce_Threads_Dim_DP\"",);

            return false;
        }

        this->total_reduce_batch_DP_size = tmp_total_elements_to_reduce;
    }

    return true;
}

__device__ bool cuModel::Reallocate_Reduce_Threads_Dim(size_t const number_threads_received)
{
    size_t tmp_total_elements_to_reduce,
                      tmp_index_dim3(0u);
    
    if(this->ptr_array_dim3_grid_reduce_threads != nullptr && number_threads_received != 0u)
    {
        class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;

        // Compute dimension reduce data batch.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                0,
                                                                                                tmp_dim3_grid,
                                                                                                tmp_dim3_block);
        
        // Get remaining elements to reduce.
        tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            ERR(L"No elements to reduce.",);

            return false;
        }
        // |END| Compute dimension reduce data batch. |END|
        
        // Allocating neurons reduce summation dim3 grid.
        struct dim3 *tmp_ptr_array_dim3_grid_reduce_threads(Memory::reallocate<struct dim3>(this->ptr_array_dim3_grid_reduce_threads,
                                                                                                                                      tmp_total_elements_to_reduce * sizeof(struct dim3),
                                                                                                                                      this->total_reduce_batch_size * sizeof(struct dim3),
                                                                                                                                      false));
        if(tmp_ptr_array_dim3_grid_reduce_threads == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        this->ptr_array_dim3_grid_reduce_threads = tmp_ptr_array_dim3_grid_reduce_threads;
        // |END| Allocating neurons reduce summation dim3 grid. |END|
            
        // Allocating neurons reduce summation dim3 block.
        struct dim3 *tmp_ptr_array_dim3_block_reduce_threads(Memory::reallocate<struct dim3>(this->ptr_array_dim3_block_reduce_threads,
                                                                                                                                        tmp_total_elements_to_reduce * sizeof(struct dim3),
                                                                                                                                        this->total_reduce_batch_size * sizeof(struct dim3),
                                                                                                                                        false));
        if(tmp_ptr_array_dim3_block_reduce_threads == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        this->ptr_array_dim3_block_reduce_threads = tmp_ptr_array_dim3_block_reduce_threads;
        // |END| Allocating neurons reduce summation dim3 block. |END|
            
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;

        // Loop to reduce "number of elements" to one at the end.
        do
        {
            // Compute remaining results to reduce.
            tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                    0,
                                                                                                    tmp_ptr_array_dim3_grid_reduce_threads[tmp_index_dim3],
                                                                                                    tmp_ptr_array_dim3_block_reduce_threads[tmp_index_dim3]);

            // Get the remaining results to reduce.
            tmp_total_elements_to_reduce = tmp_ptr_array_dim3_grid_reduce_threads[tmp_index_dim3].x;

            // Increment index to dim3.
            ++tmp_index_dim3;
        } while(tmp_total_elements_to_reduce != 1u);
    }

    return true;
}

__device__ bool cuModel::Reallocate_Reduce_Threads_Dim_DP(size_t const number_threads_received)
{
    size_t tmp_total_elements_to_reduce,
                      tmp_index_dim3(0u);
    
    if(this->ptr_array_dim3_grid_reduce_threads_DP != nullptr && number_threads_received != 0u)
    {
        class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;

        // Compute dimension reduce data batch.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_Dynamic_Parallelisme(tmp_total_elements_to_reduce,
                                                                                                            tmp_ptr_CUDA_Device->Get__Maximum_Blocks_Per_Multiprocessor(),
                                                                                                            tmp_dim3_grid,
                                                                                                            tmp_dim3_block);
        
        // Get remaining elements to reduce.
        tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            ERR(L"No elements to reduce.",);

            return false;
        }
        // |END| Compute dimension reduce data batch. |END|
        
        // Allocating neurons reduce summation dim3 grid.
        struct dim3 *tmp_ptr_array_dim3_grid_threads_DP(Memory::reallocate<struct dim3>(this->ptr_array_dim3_grid_reduce_threads_DP,
                                                                                                                                   tmp_total_elements_to_reduce * sizeof(struct dim3),
                                                                                                                                   this->total_reduce_batch_DP_size * sizeof(struct dim3),
                                                                                                                                   false));
        if(tmp_ptr_array_dim3_grid_threads_DP == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        this->ptr_array_dim3_grid_reduce_threads_DP = tmp_ptr_array_dim3_grid_threads_DP;
        // |END| Allocating neurons reduce summation dim3 grid. |END|
            
        // Allocating neurons reduce summation dim3 block.
        struct dim3 *tmp_ptr_array_dim3_block_threads_DP(Memory::reallocate<struct dim3>(this->ptr_array_dim3_block_reduce_threads_DP,
                                                                                                                                      tmp_total_elements_to_reduce * sizeof(struct dim3),
                                                                                                                                      this->total_reduce_batch_DP_size * sizeof(struct dim3),
                                                                                                                                      false));
        if(tmp_ptr_array_dim3_block_threads_DP == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        this->ptr_array_dim3_block_reduce_threads_DP = tmp_ptr_array_dim3_block_threads_DP;
        // |END| Allocating neurons reduce summation dim3 block. |END|
            
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = number_threads_received;

        // Loop to reduce "number of elements" to one at the end.
        do
        {
            // Compute remaining results to reduce.
            tmp_ptr_CUDA_Device->Grid_Block_Reduce_Dynamic_Parallelisme(tmp_total_elements_to_reduce,
                                                                                                                tmp_ptr_CUDA_Device->Get__Maximum_Blocks_Per_Multiprocessor(),
                                                                                                                tmp_ptr_array_dim3_grid_threads_DP[tmp_index_dim3],
                                                                                                                tmp_ptr_array_dim3_block_threads_DP[tmp_index_dim3]);

            // Get the remaining results to reduce.
            tmp_total_elements_to_reduce = tmp_ptr_array_dim3_grid_threads_DP[tmp_index_dim3].x;

            // Increment index to dim3.
            ++tmp_index_dim3;
        } while(tmp_total_elements_to_reduce != 1u);
    }

    return true;
}

__device__ bool cuModel::Reallocate_Reduce_Cost(size_t const total_reduce_batch_size_received)
{
    if(this->ptr_array_reduce_number_loss != nullptr && total_reduce_batch_size_received != 0u)
    {
        // Allocating reduce number loss.
        size_t *tmp_ptr_array_reduce_number_loss(Memory::reallocate_cpp<size_t>(this->ptr_array_reduce_number_loss,
                                                                                                                        total_reduce_batch_size_received,
                                                                                                                        this->total_reduce_batch_size,
                                                                                                                        false));
        if(tmp_ptr_array_reduce_number_loss == nullptr)
        {
            ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                     sizeof(size_t),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return false;
        }
        this->ptr_array_reduce_number_loss = tmp_ptr_array_reduce_number_loss;
        // |END| Allocating reduce number loss. |END|
        
        // Allocating reduce bit fail values.
        size_t *tmp_ptr_array_reduce_bit_fail_values(Memory::reallocate_cpp<size_t>(this->ptr_array_reduce_bit_fail_values,
                                                                                                                          total_reduce_batch_size_received,
                                                                                                                          this->total_reduce_batch_size,
                                                                                                                          false));
        if(tmp_ptr_array_reduce_bit_fail_values == nullptr)
        {
            ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                     sizeof(size_t),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return false;
        }
        this->ptr_array_reduce_bit_fail_values = tmp_ptr_array_reduce_bit_fail_values;
        // |END| Allocating reduce bit fail values. |END|
        
        // Allocating reduce accuracy values.
        var *tmp_ptr_array_reduce_accuracy_values(Memory::reallocate_cpp<var>(this->ptr_array_reduce_accuracy_values[0],
                                                                                                                   total_reduce_batch_size_received,
                                                                                                                   this->total_reduce_batch_size,
                                                                                                                   false));
        if(tmp_ptr_array_reduce_accuracy_values == nullptr)
        {
            ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                     sizeof(var),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return false;
        }
        this->ptr_array_reduce_accuracy_values[0] = tmp_ptr_array_reduce_accuracy_values;

        tmp_ptr_array_reduce_accuracy_values = Memory::reallocate_cpp<var>(this->ptr_array_reduce_accuracy_values[1],
                                                                                                                 total_reduce_batch_size_received,
                                                                                                                 this->total_reduce_batch_size,
                                                                                                                 false);
        if(tmp_ptr_array_reduce_accuracy_values == nullptr)
        {
            ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                     sizeof(var),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return false;
        }
        this->ptr_array_reduce_accuracy_values[1] = tmp_ptr_array_reduce_accuracy_values;

        tmp_ptr_array_reduce_accuracy_values = Memory::reallocate_cpp<var>(this->ptr_array_reduce_accuracy_values[2],
                                                                                                                 total_reduce_batch_size_received,
                                                                                                                 this->total_reduce_batch_size,
                                                                                                                 false);
        if(tmp_ptr_array_reduce_accuracy_values == nullptr)
        {
            ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                     sizeof(var),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return false;
        }
        this->ptr_array_reduce_accuracy_values[2] = tmp_ptr_array_reduce_accuracy_values;

        tmp_ptr_array_reduce_accuracy_values = Memory::reallocate_cpp<var>(this->ptr_array_reduce_accuracy_values[3],
                                                                                                                 total_reduce_batch_size_received,
                                                                                                                 this->total_reduce_batch_size,
                                                                                                                 false);
        if(tmp_ptr_array_reduce_accuracy_values == nullptr)
        {
            ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                     sizeof(var),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return false;
        }
        this->ptr_array_reduce_accuracy_values[3] = tmp_ptr_array_reduce_accuracy_values;

        tmp_ptr_array_reduce_accuracy_values = Memory::reallocate_cpp<var>(this->ptr_array_reduce_accuracy_values[4],
                                                                                                                 total_reduce_batch_size_received,
                                                                                                                 this->total_reduce_batch_size,
                                                                                                                 false);
        if(tmp_ptr_array_reduce_accuracy_values == nullptr)
        {
            ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                     sizeof(var),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return false;
        }
        this->ptr_array_reduce_accuracy_values[4] = tmp_ptr_array_reduce_accuracy_values;
        // |END| Allocating reduce accuracy values.. |END|
        
        // Allocating reduce loss values.
        var *tmp_ptr_array_reduce_loss_values(Memory::reallocate_cpp<var>(this->ptr_array_reduce_loss_values,
                                                                                                            total_reduce_batch_size_received,
                                                                                                            this->total_reduce_batch_size,
                                                                                                            false));
        if(tmp_ptr_array_reduce_loss_values == nullptr)
        {
            ERR(L"An error has been triggered from the \"reallocate_cpp<%zu>(ptr, %zu, %zu, false)\" function. At line %d.",
                                     sizeof(var),
                                     total_reduce_batch_size_received,
                                     this->total_reduce_batch_size,
                                     __LINE__);

            return false;
        }
        this->ptr_array_reduce_loss_values = tmp_ptr_array_reduce_loss_values;
        // |END| Allocating reduce loss values.. |END|
    }

    return true;
}

__device__ bool cuModel::Reallocate__Batch__Neuron_Unit(size_t const batch_size)
{
    if(this->total_neuron_units_allocated != 0u)
    {
        size_t tmp_number_neuron_units;

        struct cuLayer const *const last_layer(this->ptr_last_layer);
        struct cuLayer *layer_it(this->ptr_array_layers);

        struct cuNeuron const *tmp_ptr_last_neuron_unit;
        struct cuNeuron *tmp_ptr_neuron_unit_it;

        // Allocating neuron unit(s) summation(s).
        var *tmp_ptr_array_neuron_units_summations(Memory::reallocate_cpp<var>(this->ptr_array_neuron_units_summations,
                                                                                                            batch_size * this->total_neuron_units_allocated,
                                                                                                            this->batch_size * this->total_neuron_units_allocated,
                                                                                                            false));
        if(tmp_ptr_array_neuron_units_summations == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        this->ptr_array_neuron_units_summations = tmp_ptr_array_neuron_units_summations;
        // |END| Allocating neuron unit(s) summation(s). |END|
        
        // Allocating neuron unit(s) value(s).
        var *tmp_ptr_array_neuron_units_values(Memory::reallocate_cpp<var>(this->ptr_array_neuron_units_values,
                                                                                                    batch_size * this->total_neuron_units_allocated,
                                                                                                    this->batch_size * this->total_neuron_units_allocated,
                                                                                                    false));
        if(tmp_ptr_array_neuron_units_values == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        this->ptr_array_neuron_units_values = tmp_ptr_array_neuron_units_values;
        // |END| Allocating neuron unit(s) value(s). |END|
        
        // Allocating neuron unit(s) error(s).
        var *tmp_ptr_array_neuron_units_errors(Memory::reallocate_cpp<var>(this->ptr_array_neuron_units_errors,
                                                                                                    batch_size * this->total_neuron_units_allocated,
                                                                                                    this->batch_size * this->total_neuron_units_allocated,
                                                                                                    false));
        if(tmp_ptr_array_neuron_units_errors == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        this->ptr_array_neuron_units_errors = tmp_ptr_array_neuron_units_errors;
        // |END| Allocating neuron unit(s) error(s). |END|
        
        for(; layer_it != last_layer; ++layer_it)
        {
            if((tmp_number_neuron_units = *layer_it->ptr_number_neurons) != 0u)
            {
                for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit,
                    tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
                {
                    tmp_ptr_neuron_unit_it->ptr_array_summations = tmp_ptr_array_neuron_units_summations++;
                    tmp_ptr_neuron_unit_it->ptr_array_values = tmp_ptr_array_neuron_units_values++;
                    tmp_ptr_neuron_unit_it->ptr_array_errors = tmp_ptr_array_neuron_units_errors++;
                }

                tmp_ptr_array_neuron_units_summations += (batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_values += (batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_errors += (batch_size - 1u) * tmp_number_neuron_units;
            }
        }
    }

    return true;
}

__device__ bool cuModel::Reallocate__Batch__Neuron_Reduce_Summation(size_t const batch_size)
{
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_2D_neurons_reduce_summation != nullptr)
    {
        struct cuNeuron *tmp_ptr_neuron_unit_it(this->ptr_array_layers->ptr_array_neuron_units);
        struct cuNeuron const *const tmp_ptr_last_neuron_unit(tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated);
        
        struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                         tmp_dim3_block_zero(1,1, 1u),
                         tmp_dim3_grid_copy(1,1, 1u),
                         tmp_dim3_block_copy(1,1, 1u);

        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(batch_size * this->neurons_total_reduce_summation_size,
                                                                                                  this->batch_size * this->neurons_total_reduce_summation_size,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        // Allocating neuron unit(s) value(s).
        var **tmp_ptr_array_2D_neurons_position_reduce_summation_array(this->ptr_array_2D_neurons_reduce_summation);
        
        var *tmp_ptr_array_neuron_units_reduce_summation_results(Memory::reallocate_cpp<var>(*this->ptr_array_2D_neurons_reduce_summation,
                                                                                                                                    batch_size * this->neurons_total_reduce_summation_size,
                                                                                                                                    this->batch_size * this->neurons_total_reduce_summation_size,
                                                                                                                                    &tmp_dim3_grid_zero,
                                                                                                                                    &tmp_dim3_block_zero,
                                                                                                                                    &tmp_dim3_grid_copy,
                                                                                                                                    &tmp_dim3_block_copy,
                                                                                                                                    false));
        if(tmp_ptr_array_neuron_units_reduce_summation_results == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neuron unit(s) value(s). |END|
        
        // Loop through each neurons in the network.
        for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                             ++tmp_ptr_array_2D_neurons_position_reduce_summation_array)
        {
            // Assign the position index of the begining results array from that array.
            *tmp_ptr_array_2D_neurons_position_reduce_summation_array = tmp_ptr_array_neuron_units_reduce_summation_results;

            // Assign the begining results array to that pointer.
            tmp_ptr_neuron_unit_it->ptr_array_reduce_summation = tmp_ptr_array_2D_neurons_position_reduce_summation_array;
            
            // If is not the bias. (The bias have no elements to reduce.)
            if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections != 0u)
            {
                // Increment the begining results by the reduce summation size of that neuron.
                tmp_ptr_array_neuron_units_reduce_summation_results += *tmp_ptr_neuron_unit_it->ptr_reduce_summation_size;
            }
        }
    }

    return true;
}

__device__ bool cuModel::Reallocate__Batch__Neuron_Reduce_Error(size_t const batch_size)
{
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_2D_neurons_reduce_error != nullptr)
    {
        struct cuNeuron *tmp_ptr_neuron_unit_it(this->ptr_array_layers->ptr_array_neuron_units);
        struct cuNeuron const *const tmp_ptr_last_neuron_unit(tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated);
        
        struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                         tmp_dim3_block_zero(1,1, 1u),
                         tmp_dim3_grid_copy(1,1, 1u),
                         tmp_dim3_block_copy(1,1, 1u);

        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(batch_size * this->neurons_total_reduce_error_size,
                                                                                                  this->batch_size * this->neurons_total_reduce_error_size,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        // Allocating neuron unit(s) value(s).
        var **tmp_ptr_array_2D_neurons_position_reduce_error_array(this->ptr_array_2D_neurons_reduce_error);
        
        var *tmp_ptr_array_neuron_units_reduce_error_results(Memory::reallocate_cpp<var>(*this->ptr_array_2D_neurons_reduce_error,
                                                                                                                          batch_size * this->neurons_total_reduce_error_size,
                                                                                                                          this->batch_size * this->neurons_total_reduce_error_size,
                                                                                                                          &tmp_dim3_grid_zero,
                                                                                                                          &tmp_dim3_block_zero,
                                                                                                                          &tmp_dim3_grid_copy,
                                                                                                                          &tmp_dim3_block_copy,
                                                                                                                          false));
        if(tmp_ptr_array_neuron_units_reduce_error_results == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neuron unit(s) value(s). |END|
        
        // Loop through each neurons in the network.
        for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                             ++tmp_ptr_array_2D_neurons_position_reduce_error_array)
        {
            // Assign the position index of the begining results array from that array.
            *tmp_ptr_array_2D_neurons_position_reduce_error_array = tmp_ptr_array_neuron_units_reduce_error_results;

            // Assign the begining results array to that pointer.
            tmp_ptr_neuron_unit_it->ptr_array_reduce_error = tmp_ptr_array_2D_neurons_position_reduce_error_array;
            
            // Increment the begining results by the reduce error size of that neuron.
            tmp_ptr_array_neuron_units_reduce_error_results += *tmp_ptr_neuron_unit_it->ptr_reduce_error_size;
        }
    }

    return true;
}

__device__ bool cuModel::Reallocate__Normalized_Unit__Batch_Normalization(size_t const batch_size)
{
    if(this->use_Batch_Renormalization && this->total_neuron_units_allocated != 0u)
    {
        size_t tmp_number_neuron_units;

        struct cuLayer const *const last_layer(this->ptr_last_layer);
        struct cuLayer *layer_it(this->ptr_array_layers);

        struct cuNeuron const *tmp_ptr_last_neuron_unit;
        struct cuNeuron *tmp_ptr_neuron_unit_it;
        
        struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                         tmp_dim3_block_zero(1,1, 1u),
                         tmp_dim3_grid_copy(1,1, 1u),
                         tmp_dim3_block_copy(1,1, 1u);

        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(batch_size * this->total_neuron_units_allocated,
                                                                                                  this->batch_size * this->total_neuron_units_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        // Allocating neuron unit(s) value(s) hat.
        var *tmp_ptr_array_neuron_units_values_hat(Memory::reallocate_cpp<var>(this->ptr_array_normalized_batch_units_values_hats,
                                                                                                            batch_size * this->total_neuron_units_allocated,
                                                                                                            this->batch_size * this->total_neuron_units_allocated,
                                                                                                            &tmp_dim3_grid_zero,
                                                                                                            &tmp_dim3_block_zero,
                                                                                                            &tmp_dim3_grid_copy,
                                                                                                            &tmp_dim3_block_copy,
                                                                                                            false));
        if(tmp_ptr_array_neuron_units_values_hat == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neuron unit(s) value(s) hat. |END|
            
        // Allocating neuron unit(s) value(s) normalize.
        var *tmp_ptr_array_neuron_units_values_normalize(Memory::reallocate_cpp<var>(this->ptr_array_normalized_batch_units_values_normalizes,
                                                                                                                      batch_size * this->total_neuron_units_allocated,
                                                                                                                      this->batch_size * this->total_neuron_units_allocated,
                                                                                                                      &tmp_dim3_grid_zero,
                                                                                                                      &tmp_dim3_block_zero,
                                                                                                                      &tmp_dim3_grid_copy,
                                                                                                                      &tmp_dim3_block_copy,
                                                                                                                      false));
        if(tmp_ptr_array_neuron_units_values_normalize == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neuron unit(s) value(s) normalize. |END|
        
        // Allocating neurons mean.
        var *tmp_ptr_array_neuron_units_mean_it(Memory::reallocate_cpp<var>(this->ptr_array_normalized_batch_units_means,
                                                                                                        batch_size * this->total_neuron_units_allocated,
                                                                                                        this->batch_size * this->total_neuron_units_allocated,
                                                                                                        &tmp_dim3_grid_zero,
                                                                                                        &tmp_dim3_block_zero,
                                                                                                        &tmp_dim3_grid_copy,
                                                                                                        &tmp_dim3_block_copy,
                                                                                                        false));
        if(tmp_ptr_array_neuron_units_mean_it == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neurons mean. |END|
        
        // Allocating neurons variance.
        var *tmp_ptr_array_neuron_units_variance_it(Memory::reallocate_cpp<var>(this->ptr_array_normalized_batch_units_variances,
                                                                                                            batch_size * this->total_neuron_units_allocated,
                                                                                                            this->batch_size * this->total_neuron_units_allocated,
                                                                                                            &tmp_dim3_grid_zero,
                                                                                                            &tmp_dim3_block_zero,
                                                                                                            &tmp_dim3_grid_copy,
                                                                                                            &tmp_dim3_block_copy,
                                                                                                            false));
        if(tmp_ptr_array_neuron_units_variance_it == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neurons variance. |END|
        
        // Allocating neurons derivative mean.
        var *tmp_ptr_array_neuron_units_derivative_mean_it(Memory::reallocate_cpp<var>(this->ptr_array_normalized_batch_units_derivatives_means,
                                                                                                                        batch_size * this->total_neuron_units_allocated,
                                                                                                                        this->batch_size * this->total_neuron_units_allocated,
                                                                                                                        &tmp_dim3_grid_zero,
                                                                                                                        &tmp_dim3_block_zero,
                                                                                                                        &tmp_dim3_grid_copy,
                                                                                                                        &tmp_dim3_block_copy,
                                                                                                                        false));
        if(tmp_ptr_array_neuron_units_derivative_mean_it == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neurons derivative mean. |END|
        
        // Allocating neurons derivative variance.
        var *tmp_ptr_array_neuron_units_derivative_variance_it(Memory::reallocate_cpp<var>(this->ptr_array_normalized_batch_units_derivatives_variances,
                                                                                                                           batch_size * this->total_neuron_units_allocated,
                                                                                                                           this->batch_size * this->total_neuron_units_allocated,
                                                                                                                           &tmp_dim3_grid_zero,
                                                                                                                           &tmp_dim3_block_zero,
                                                                                                                           &tmp_dim3_grid_copy,
                                                                                                                           &tmp_dim3_block_copy,
                                                                                                                           false));
        if(tmp_ptr_array_neuron_units_derivative_variance_it == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neurons derivative variance. |END|
        
        this->ptr_array_normalized_batch_units_values_hats = tmp_ptr_array_neuron_units_values_hat;
        this->ptr_array_normalized_batch_units_values_normalizes = tmp_ptr_array_neuron_units_values_normalize;
        this->ptr_array_normalized_batch_units_means = tmp_ptr_array_neuron_units_mean_it;
        this->ptr_array_normalized_batch_units_variances = tmp_ptr_array_neuron_units_variance_it;
        this->ptr_array_normalized_batch_units_derivatives_means = tmp_ptr_array_neuron_units_derivative_mean_it;
        this->ptr_array_normalized_batch_units_derivatives_variances = tmp_ptr_array_neuron_units_derivative_variance_it;
        
        for(; layer_it != last_layer; ++layer_it)
        {
            if((tmp_number_neuron_units = *layer_it->ptr_number_neurons) != 0u)
            {
                for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit,
                    tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_values_hat,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_values_normalize,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_mean_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_variance_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_derivative_mean_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_derivative_variance_it)
                {
                    tmp_ptr_neuron_unit_it->ptr_array_values_hats = tmp_ptr_array_neuron_units_values_hat;
                    tmp_ptr_neuron_unit_it->ptr_array_values_normalizes = tmp_ptr_array_neuron_units_values_normalize;
                    tmp_ptr_neuron_unit_it->ptr_array_means = tmp_ptr_array_neuron_units_mean_it;
                    tmp_ptr_neuron_unit_it->ptr_array_variances = tmp_ptr_array_neuron_units_variance_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_means = tmp_ptr_array_neuron_units_derivative_mean_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances = tmp_ptr_array_neuron_units_derivative_variance_it;
                }

                tmp_ptr_array_neuron_units_values_hat += (batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_values_normalize += (batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_mean_it += (batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_variance_it += (batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_derivative_mean_it += (batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_derivative_variance_it += (batch_size - 1u) * tmp_number_neuron_units;
            }
        }
    }

    return true;
}

__device__ bool cuModel::Reallocate__Batch__Neuron_Batch_Normalization_Transpose(size_t const batch_size)
{
    if(this->use_Batch_Renormalization && this->ptr_array_neuron_units_transposed_mean != nullptr)
    {
        struct cuLayer const *const last_layer(this->ptr_last_layer);
        struct cuLayer *layer_it(this->ptr_array_layers);

        struct cuNeuron const *tmp_ptr_last_neuron_unit;
        struct cuNeuron *tmp_ptr_neuron_unit_it;
        
        struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                         tmp_dim3_block_zero(1,1, 1u),
                         tmp_dim3_grid_copy(1,1, 1u),
                         tmp_dim3_block_copy(1,1, 1u);

        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(batch_size * this->total_neuron_units_allocated,
                                                                                                  this->batch_size * this->total_neuron_units_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        // Allocating neurons mean.
        var *tmp_ptr_array_neuron_units_transposed_mean_it(Memory::reallocate_cpp<var>(this->ptr_array_neuron_units_transposed_mean,
                                                                                                                          batch_size * this->total_neuron_units_allocated,
                                                                                                                          this->batch_size * this->total_neuron_units_allocated,
                                                                                                                          &tmp_dim3_grid_zero,
                                                                                                                          &tmp_dim3_block_zero,
                                                                                                                          &tmp_dim3_grid_copy,
                                                                                                                          &tmp_dim3_block_copy,
                                                                                                                          false));
        if(tmp_ptr_array_neuron_units_transposed_mean_it == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neurons mean. |END|
        
        // Allocating neurons variance.
        var *tmp_ptr_array_neuron_units_transposed_variance_it(Memory::reallocate_cpp<var>(this->ptr_array_neuron_units_transposed_variance,
                                                                                                                              batch_size * this->total_neuron_units_allocated,
                                                                                                                              this->batch_size * this->total_neuron_units_allocated,
                                                                                                                              &tmp_dim3_grid_zero,
                                                                                                                              &tmp_dim3_block_zero,
                                                                                                                              &tmp_dim3_grid_copy,
                                                                                                                              &tmp_dim3_block_copy,
                                                                                                                              false));
        if(tmp_ptr_array_neuron_units_transposed_variance_it == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neurons variance. |END|
        
        this->ptr_array_neuron_units_transposed_mean = tmp_ptr_array_neuron_units_transposed_mean_it;
        this->ptr_array_neuron_units_transposed_variance = tmp_ptr_array_neuron_units_transposed_variance_it;
        
        for(; layer_it != last_layer; ++layer_it)
        {
            for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit,
                tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
            {
                tmp_ptr_neuron_unit_it->ptr_array_transposed_mean = tmp_ptr_array_neuron_units_transposed_mean_it;
                tmp_ptr_neuron_unit_it->ptr_array_transposed_variance = tmp_ptr_array_neuron_units_transposed_variance_it;

                tmp_ptr_array_neuron_units_transposed_mean_it += batch_size;
                tmp_ptr_array_neuron_units_transposed_variance_it += batch_size;
            }
        }
    }

    return true;
}

// TODO: Make "Reallocate__Batch__Neuron_Batch_Normalization_Reduce" and "Reallocate__Batch__Neuron_Reduce_Batch"
__device__ bool cuModel::Reallocate__Batch__Neuron_Batch_Normalization_Reduce(size_t const batch_size)
{
    size_t tmp_neurons_reduce_batch_size_so_far,
                      tmp_total_elements_to_reduce,
                      tmp_layer_reduce_batch_size,
                      tmp_number_neurons_in_layer,
                      tmp_index_dim3;
    
    if(this->use_Batch_Renormalization
      &&
      this->total_neuron_units_allocated != 0u
      &&
      this->ptr_array_neuron_units_reduce_batch_size != nullptr)
    {
        size_t *tmp_ptr_array_neuron_units_reduce_batch_size(this->ptr_array_neuron_units_reduce_batch_size);

        // ONLY FOR DENSE LAYER.
        // TODO: Make shortcut layer compatible.
        struct cuLayer const *const last_layer(this->ptr_last_layer);
        struct cuLayer *layer_it;
        
        struct cuNeuron const *tmp_ptr_last_neuron_unit;
        struct cuNeuron *tmp_ptr_neuron_unit_it;
        
        class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block,
                         tmp_dim3_grid_zero(1,1, 1u),
                         tmp_dim3_block_zero(1,1, 1u),
                         tmp_dim3_grid_copy(1,1, 1u),
                         tmp_dim3_block_copy(1,1, 1u);
        
        // COMPUTE REDUCE BATCH SIZE.
        for(tmp_neurons_reduce_batch_size_so_far = 0,
            tmp_ptr_neuron_unit_it = this->ptr_array_layers->ptr_array_neuron_units,
            tmp_ptr_last_neuron_unit = tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                                      ++tmp_ptr_array_neuron_units_reduce_batch_size)
        {
            // Number elements to reduce equal the size of batch.
            tmp_total_elements_to_reduce = batch_size;

            // If the neuron is a bias. Number of elements to reduce equal zero.
            if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections == 0u)
            { tmp_total_elements_to_reduce = 0u; }
            
            // If is not the bias. (The bias have no elements to reduce.)
            if(tmp_total_elements_to_reduce != 0u)
            {
                // Dimension required to reduce the number of elements.
                tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                      0,
                                                                                                      tmp_dim3_grid,
                                                                                                      tmp_dim3_block);
            
                // Get remaining elements to reduce.
                tmp_total_elements_to_reduce = tmp_dim3_grid.x;
            }

            // Maximum remaining elements to reduce.
            *tmp_ptr_array_neuron_units_reduce_batch_size = tmp_total_elements_to_reduce;

            // Summation of the total maximum number of batch result.
            tmp_neurons_reduce_batch_size_so_far += tmp_total_elements_to_reduce;
        }

        if(tmp_neurons_reduce_batch_size_so_far == 0u)
        {
            ERR(L"No elements to reduce.",);

            return false;
        }
        // |END| Compute dimension reduce batch. |END|
        // |END| COMPUTE REDUCE BATCH SIZE. |END|
        
        // COMPUTE DIMENSION REDUCE BATCH.
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(tmp_neurons_reduce_batch_size_so_far,
                                                                                                  this->neurons_total_reduce_batch_size,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  tmp_ptr_CUDA_Device,
                                                                                                  false);
        
        // Allocating neurons reduce batch mean.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        var **tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array(this->ptr_array_2D_neurons_reduce_batch_mean);
        
        var *tmp_ptr_array_neuron_units_reduce_batch_mean_results(Memory::reallocate_cpp<var>(*this->ptr_array_2D_neurons_reduce_batch_mean,
                                                                                                                                      tmp_neurons_reduce_batch_size_so_far,
                                                                                                                                      this->neurons_total_reduce_batch_size,
                                                                                                                                      &tmp_dim3_grid_zero,
                                                                                                                                      &tmp_dim3_block_zero,
                                                                                                                                      &tmp_dim3_grid_copy,
                                                                                                                                      &tmp_dim3_block_copy,
                                                                                                                                      false));
        if(tmp_ptr_array_neuron_units_reduce_batch_mean_results == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neurons reduce batch mean. |END|
        
        // Allocating neurons reduce batch variance.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        var **tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array(this->ptr_array_2D_neurons_reduce_batch_variance);
        
        var *tmp_ptr_array_neuron_units_reduce_batch_variance_results(Memory::reallocate_cpp<var>(*this->ptr_array_2D_neurons_reduce_batch_variance,
                                                                                                                                         tmp_neurons_reduce_batch_size_so_far,
                                                                                                                                         this->neurons_total_reduce_batch_size,
                                                                                                                                         &tmp_dim3_grid_zero,
                                                                                                                                         &tmp_dim3_block_zero,
                                                                                                                                         &tmp_dim3_grid_copy,
                                                                                                                                         &tmp_dim3_block_copy,
                                                                                                                                         false));
        if(tmp_ptr_array_neuron_units_reduce_batch_variance_results == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        // |END| Allocating neurons reduce batch variance. |END|
        
        // Allocating neurons reduce batch dim3 grid.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_batch(Memory::reallocate<struct dim3>(this->ptr_array_neuron_units_dim3_grid_reduce_batch,
                                                                                                                                       tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3),
                                                                                                                                       this->neurons_total_reduce_batch_size * sizeof(struct dim3),
                                                                                                                                       false));
        if(tmp_ptr_array_neuron_units_dim3_grid_batch == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        this->ptr_array_neuron_units_dim3_grid_reduce_batch = tmp_ptr_array_neuron_units_dim3_grid_batch;
        // |END| Allocating neurons reduce batch dim3 grid. |END|
            
        // Allocating neurons reduce batch dim3 block.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_batch(Memory::reallocate<struct dim3>(this->ptr_array_neuron_units_dim3_block_reduce_batch,
                                                                                                                                         tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3),
                                                                                                                                         this->neurons_total_reduce_batch_size * sizeof(struct dim3),
                                                                                                                                         false));
        if(tmp_ptr_array_neuron_units_dim3_block_batch == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        this->ptr_array_neuron_units_dim3_block_reduce_batch = tmp_ptr_array_neuron_units_dim3_block_batch;
        // |END| Allocating neurons reduce batch dim3 block. |END|
        
        // Loop through each layers.
        for(layer_it = this->ptr_array_layers; layer_it != last_layer; ++layer_it)
        {
            // Get neurons array from that layer.
            tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units;

            // Get the reduce batch size of each neurons in that layer.
            tmp_layer_reduce_batch_size = *tmp_ptr_neuron_unit_it->ptr_reduce_batch_size;
            
            // Get the number of neurons in layer.
            tmp_number_neurons_in_layer = *layer_it->ptr_number_neurons;
            
            // Loop through each neurons in the layer.
            for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array)
            {
                // Result.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array = tmp_ptr_array_neuron_units_reduce_batch_mean_results;
                *tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array = tmp_ptr_array_neuron_units_reduce_batch_variance_results;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_reduce_mean = tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array;
                tmp_ptr_neuron_unit_it->ptr_array_reduce_variance = tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array;
                // |END| Result. |END|
                
                // Number elements to reduce equal the size of batch
                tmp_total_elements_to_reduce = batch_size;
                
                // If the neuron is a bias. Number of elements to reduce equal zero.
                if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections == 0u)
                { tmp_total_elements_to_reduce = 0u; }
                
                // If is not the bias. (The bias have no elements to reduce.)
                if(tmp_total_elements_to_reduce != 0u)
                {
                    // Assign dim3 grid to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads = tmp_ptr_array_neuron_units_dim3_grid_batch++;
                    // Assign dim3 block to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_threads = tmp_ptr_array_neuron_units_dim3_block_batch++;

                    // Initialize index to zero.
                    tmp_index_dim3 = 0u;

                    // Loop to reduce "number of elements" to one at the end.
                    do
                    {
                        // Compute remaining results to reduce.
                        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                                0,
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)],
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)]);

                        // Get the remaining results to reduce.
                        tmp_total_elements_to_reduce = tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)].x;

                        // Increment index to dim3.
                        ++tmp_index_dim3;
                    } while(tmp_total_elements_to_reduce != 1u);
                    // |END| dim3. |END|

                    // Increment the begining results by the layer reduce batch size.
                    tmp_ptr_array_neuron_units_reduce_batch_mean_results += tmp_layer_reduce_batch_size;
                    tmp_ptr_array_neuron_units_reduce_batch_variance_results += tmp_layer_reduce_batch_size;
                }
            }
            
            // If some elements need to be reduce in the layer.
            if(tmp_layer_reduce_batch_size != 0u)
            {
                // Increment pointer by (number of neurons in layer minus bias) times (layer reduce batch size minus one).
                tmp_ptr_array_neuron_units_dim3_grid_batch += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_batch_size - 1u);
                tmp_ptr_array_neuron_units_dim3_block_batch += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_batch_size - 1u);
            }
        }
        // |END| COMPUTE DIMENSION REDUCE BATCH. |END|

        this->neurons_total_reduce_batch_size = tmp_neurons_reduce_batch_size_so_far;
    }

    return true;
}

__device__ bool cuModel::Reallocate__Thread__Parameter(size_t const number_threads_received)
{
    if(this->total_parameters_allocated != 0u)
    {
        struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                         tmp_dim3_block_zero(1,1, 1u),
                         tmp_dim3_grid_copy(1,1, 1u),
                         tmp_dim3_block_copy(1,1, 1u);
        
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_threads_received * this->total_parameters_allocated,
                                                                                                  this->number_threads * this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        var *tmp_ptr_array_derivatives_parameters(Memory::reallocate_cpp<var>(this->ptr_array_derivatives_parameters,
                                                                                                                number_threads_received * this->total_parameters_allocated,
                                                                                                                this->number_threads * this->total_parameters_allocated,
                                                                                                                &tmp_dim3_grid_zero,
                                                                                                                &tmp_dim3_block_zero,
                                                                                                                &tmp_dim3_grid_copy,
                                                                                                                &tmp_dim3_block_copy,
                                                                                                                false));
        if(tmp_ptr_array_derivatives_parameters == nullptr)
        {
            ERR(L"Can not allocate memory.");

            return false;
        }
        this->ptr_array_derivatives_parameters = tmp_ptr_array_derivatives_parameters;

        if(this->use_Batch_Renormalization)
        { this->Reset__Derivative_Parameter__Normalized_Unit(); }
    }

    return true;
}

__device__ bool cuModel::Reallocate__Parameter(size_t const number_parameters_received)
{
    if(this->total_parameters_allocated != 0u)
    {
        struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                         tmp_dim3_block_zero(1,1, 1u),
                         tmp_dim3_grid_copy(1,1, 1u),
                         tmp_dim3_block_copy(1,1, 1u);
        
        class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        // Parameters.
        if(this->ptr_array_parameters != nullptr)
        {
            this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                      this->total_parameters_allocated,
                                                                                                      tmp_dim3_grid_zero,
                                                                                                      tmp_dim3_block_zero,
                                                                                                      tmp_dim3_grid_copy,
                                                                                                      tmp_dim3_block_copy,
                                                                                                      tmp_ptr_CUDA_Device);

            var *tmp_ptr_array_parameters(Memory::reallocate_cpp<var>(this->ptr_array_parameters,
                                                                                                    number_parameters_received,
                                                                                                    this->total_parameters_allocated,
                                                                                                    &tmp_dim3_grid_zero,
                                                                                                    &tmp_dim3_block_zero,
                                                                                                    &tmp_dim3_grid_copy,
                                                                                                    &tmp_dim3_block_copy));
            if(tmp_ptr_array_parameters == nullptr)
            {
                ERR(L"Can not allocate memory.",);

                return false;
            }
            this->ptr_array_parameters = tmp_ptr_array_parameters;
                
            if(this->Reallocate__Parameter__Optimizer(number_parameters_received) == false)
            {
                ERR(L"From \"Reallocate__Parameter__Optimizer\".",);

                return false;
            }
            else if(this->Use__Regularization_Parameter() && this->Reallocate__Parameter__Regularization(number_parameters_received) == false)
            {
                ERR(L"From \"Reallocate__Parameter__Regularization\".",);

                return false;
            }
            else if(this->use_Dropout && this->Reallocate__Parameter__Dropout_Bernoulli(number_parameters_received) == false)
            {
                ERR(L"From \"Reallocate__Parameter__Dropout_Bernoulli\".",);

                return false;
            }
                
            if(this->use_Batch_Renormalization)
            { this->Reset__Parameter__Normalized_Unit(); }
        }
        // |END| Parameters. |END|

        // Derivates parameters.
        if(this->ptr_array_derivatives_parameters != nullptr)
        {
            this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(this->number_threads * number_parameters_received,
                                                                                                      this->number_threads * this->total_parameters_allocated,
                                                                                                      tmp_dim3_grid_zero,
                                                                                                      tmp_dim3_block_zero,
                                                                                                      tmp_dim3_grid_copy,
                                                                                                      tmp_dim3_block_copy,
                                                                                                      tmp_ptr_CUDA_Device,
                                                                                                      false);

            var *tmp_ptr_array_derivatives_parameters(Memory::reallocate_cpp<var>(this->ptr_array_derivatives_parameters,
                                                                                                                    this->number_threads * number_parameters_received,
                                                                                                                    this->number_threads * this->total_parameters_allocated,
                                                                                                                    &tmp_dim3_grid_zero,
                                                                                                                    &tmp_dim3_block_zero,
                                                                                                                    &tmp_dim3_grid_copy,
                                                                                                                    &tmp_dim3_block_copy,
                                                                                                                    false));
            if(tmp_ptr_array_derivatives_parameters == nullptr)
            {
                ERR(L"Can not allocate memory.",);

                return false;
            }
            this->ptr_array_derivatives_parameters = tmp_ptr_array_derivatives_parameters;

            if(this->use_Batch_Renormalization)
            { this->Reset__Derivative_Parameter__Normalized_Unit(); }
        }
        // |END| Derivates parameters. |END|
            
        this->total_parameters = number_parameters_received;
        this->total_parameters_allocated = number_parameters_received;

        // Prepare grids and blocks dimensions.
        this->Prepare__Parameters__Grids_Blocks_Dimensions();

        this->Prepare__Threads_Parameters__Grids_Blocks_Dimensions(this->number_threads);
        // |END| Prepare grids and blocks dimensions. |END|
    }

    return true;
}
    
__device__ bool cuModel::Reallocate__Parameter__Regularization(size_t const number_parameters_received)
{
    if(this->ptr_array_mask_regularized_parameters != nullptr)
    {
        struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                         tmp_dim3_block_zero(1,1, 1u),
                         tmp_dim3_grid_copy(1,1, 1u),
                         tmp_dim3_block_copy(1,1, 1u);
        
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                  this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        var *tmp_ptr_array_mask_rergularization_parameters(Memory::reallocate_cpp<var>(this->ptr_array_mask_regularized_parameters,
                                                                                                                                number_parameters_received,
                                                                                                                                this->total_parameters_allocated,
                                                                                                                                &tmp_dim3_grid_zero,
                                                                                                                                &tmp_dim3_block_zero,
                                                                                                                                &tmp_dim3_grid_copy,
                                                                                                                                &tmp_dim3_block_copy,
                                                                                                                                false));
        if(tmp_ptr_array_mask_rergularization_parameters == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_mask_regularized_parameters = tmp_ptr_array_mask_rergularization_parameters;
    }

    return true;
}
    
__device__ bool cuModel::Reallocate__Parameter__Dropout_Bernoulli(size_t const number_parameters_received)
{
    if(this->ptr_array_mask_dropout_parameters != nullptr)
    {
        struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                         tmp_dim3_block_zero(1,1, 1u),
                         tmp_dim3_grid_copy(1,1, 1u),
                         tmp_dim3_block_copy(1,1, 1u);
        
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                  this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device(),
                                                                                                  false);
        
        var *tmp_ptr_array_mask_dropout_parameters(Memory::reallocate_cpp<var>(this->ptr_array_mask_dropout_parameters,
                                                                                                                      number_parameters_received,
                                                                                                                      this->total_parameters_allocated,
                                                                                                                      &tmp_dim3_grid_zero,
                                                                                                                      &tmp_dim3_block_zero,
                                                                                                                      &tmp_dim3_grid_copy,
                                                                                                                      &tmp_dim3_block_copy,
                                                                                                                      false));
        if(tmp_ptr_array_mask_dropout_parameters == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_mask_dropout_parameters = tmp_ptr_array_mask_dropout_parameters;
        
        // If array increase in size, initialize the new entries to one.
        if(this->total_weights_allocated < number_parameters_received)
        {
            this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(number_parameters_received - this->total_weights_allocated,
                                                                                                                                                  0,
                                                                                                                                                  tmp_dim3_grid_copy,
                                                                                                                                                  tmp_dim3_block_copy);

            Memory::Fill_1D<var>(number_parameters_received - this->total_weights_allocated,
                                                                 tmp_ptr_array_mask_dropout_parameters + this->total_weights_allocated,
                                                                 1_r,
                                                                 &tmp_dim3_grid_copy,
                                                                 &tmp_dim3_block_copy);
        }
    }

    return true;
}
    
__device__ bool cuModel::Reallocate__Parameter__Optimizer(size_t const number_parameters_received)
{
    switch(this->type_optimizer_function)
    {
        case DL::OPTIMIZER::GD: return(this->Reallocate__Parameter__Gradient_Descent(number_parameters_received));
        case DL::OPTIMIZER::IRPROP_MINUS: return(this->Reallocate__Parameter__iRPROP_minus(number_parameters_received));
        case DL::OPTIMIZER::IRPROP_PLUS: return(this->Reallocate__Parameter__iRPROP_plus(number_parameters_received));
        case DL::OPTIMIZER::ADAM:
        case DL::OPTIMIZER::ADAMAX:
        case DL::OPTIMIZER::NOSADAM: return(this->Reallocate__Parameter__Adam(number_parameters_received));
        case DL::OPTIMIZER::AMSGRAD: return(this->Reallocate__Parameter__AMSGrad(number_parameters_received));
        default: return true;
    }
}

__device__ bool cuModel::Reallocate__Parameter__Gradient_Descent(size_t const number_parameters_received)
{
    if(this->learning_momentum != 0_r
        &&
        this->ptr_array_previous_delta_parameters != nullptr)
    {
        struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                         tmp_dim3_block_zero(1,1, 1u),
                         tmp_dim3_grid_copy(1,1, 1u),
                         tmp_dim3_block_copy(1,1, 1u);
        
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                  this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        // Previous delta parameters.
        var *tmp_ptr_array_previous_delta_parameters(Memory::reallocate_cpp<var>(this->ptr_array_previous_delta_parameters,
                                                                                                                      number_parameters_received,
                                                                                                                      this->total_parameters_allocated,
                                                                                                                      &tmp_dim3_grid_zero,
                                                                                                                      &tmp_dim3_block_zero,
                                                                                                                      &tmp_dim3_grid_copy,
                                                                                                                      &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_delta_parameters == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_previous_delta_parameters = tmp_ptr_array_previous_delta_parameters;
        // |END| Previous delta parameters. |END|
    }

    return true;
}

__device__ bool cuModel::Reallocate__Parameter__iRPROP_minus(size_t const number_parameters_received)
{
    struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                     tmp_dim3_block_zero(1,1, 1u),
                     tmp_dim3_grid_copy(1,1, 1u),
                     tmp_dim3_block_copy(1,1, 1u);
        
    if(this->ptr_array_previous_steps != nullptr || this->ptr_array_previous_derivatives_parameters != nullptr)
    {
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                  this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    }

    if(this->ptr_array_previous_steps != nullptr)
    {
        var *tmp_ptr_array_previous_steps(Memory::reallocate_cpp<var>(this->ptr_array_previous_steps,
                                                                                                     number_parameters_received,
                                                                                                     this->total_parameters_allocated,
                                                                                                     &tmp_dim3_grid_zero,
                                                                                                     &tmp_dim3_block_zero,
                                                                                                     &tmp_dim3_grid_copy,
                                                                                                     &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_steps == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_previous_steps = tmp_ptr_array_previous_steps;
        
        if(this->total_parameters_allocated < number_parameters_received)
        {
            Memory::Fill_1D<var>(number_parameters_received - this->total_parameters_allocated,
                                                                 tmp_ptr_array_previous_steps + this->total_parameters_allocated,
                                                                 this->rprop_delta_zero,
                                                                 &tmp_dim3_grid_zero,
                                                                 &tmp_dim3_block_zero);
        }
    }
    
    if(this->ptr_array_previous_derivatives_parameters != nullptr)
    {
        var *tmp_ptr_array_previous_derivatives_parameters(Memory::reallocate_cpp<var>(this->ptr_array_previous_derivatives_parameters,
                                                                                                                              number_parameters_received,
                                                                                                                              this->total_parameters_allocated,
                                                                                                                              &tmp_dim3_grid_zero,
                                                                                                                              &tmp_dim3_block_zero,
                                                                                                                              &tmp_dim3_grid_copy,
                                                                                                                              &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_derivatives_parameters == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_previous_derivatives_parameters = tmp_ptr_array_previous_derivatives_parameters;
    }

    return true;
}

__device__ bool cuModel::Reallocate__Parameter__iRPROP_plus(size_t const number_parameters_received)
{
    struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                     tmp_dim3_block_zero(1,1, 1u),
                     tmp_dim3_grid_copy(1,1, 1u),
                     tmp_dim3_block_copy(1,1, 1u);
        
    if(this->ptr_array_previous_steps != nullptr
      ||
      this->ptr_array_previous_delta_parameters != nullptr
      ||
      this->ptr_array_previous_derivatives_parameters != nullptr)
    {
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                    this->total_parameters_allocated,
                                                                                                    tmp_dim3_grid_zero,
                                                                                                    tmp_dim3_block_zero,
                                                                                                    tmp_dim3_grid_copy,
                                                                                                    tmp_dim3_block_copy,
                                                                                                    this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    }

    if(this->ptr_array_previous_steps != nullptr)
    {
        var *tmp_ptr_array_previous_steps(Memory::reallocate_cpp<var>(this->ptr_array_previous_steps,
                                                                                                    number_parameters_received,
                                                                                                    this->total_parameters_allocated,
                                                                                                    &tmp_dim3_grid_zero,
                                                                                                    &tmp_dim3_block_zero,
                                                                                                    &tmp_dim3_grid_copy,
                                                                                                    &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_steps == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_previous_steps = tmp_ptr_array_previous_steps;
        
        if(this->total_parameters_allocated < number_parameters_received)
        {
            Memory::Fill_1D<var>(number_parameters_received - this->total_parameters_allocated,
                                                                 tmp_ptr_array_previous_steps + this->total_parameters_allocated,
                                                                 this->rprop_delta_zero,
                                                                 &tmp_dim3_grid_zero,
                                                                 &tmp_dim3_block_zero);
        }
    }
    
    if(this->ptr_array_previous_delta_parameters != nullptr)
    {
        var *tmp_ptr_array_previous_delta_parameters(Memory::reallocate_cpp<var>(this->ptr_array_previous_delta_parameters,
                                                                                                                    number_parameters_received,
                                                                                                                    this->total_parameters_allocated,
                                                                                                                    &tmp_dim3_grid_zero,
                                                                                                                    &tmp_dim3_block_zero,
                                                                                                                    &tmp_dim3_grid_copy,
                                                                                                                    &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_delta_parameters == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_previous_delta_parameters = tmp_ptr_array_previous_delta_parameters;
    }
    
    if(this->ptr_array_previous_derivatives_parameters != nullptr)
    {
        var *tmp_ptr_array_previous_derivatives_parameters(Memory::reallocate_cpp<var>(this->ptr_array_previous_derivatives_parameters,
                                                                                                                            number_parameters_received,
                                                                                                                            this->total_parameters_allocated,
                                                                                                                            &tmp_dim3_grid_zero,
                                                                                                                            &tmp_dim3_block_zero,
                                                                                                                            &tmp_dim3_grid_copy,
                                                                                                                            &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_derivatives_parameters == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_previous_derivatives_parameters = tmp_ptr_array_previous_derivatives_parameters;
    }

    return true;
}

__device__ bool cuModel::Reallocate__Parameter__Adam(size_t const number_parameters_received)
{
    struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                     tmp_dim3_block_zero(1,1, 1u),
                     tmp_dim3_grid_copy(1,1, 1u),
                     tmp_dim3_block_copy(1,1, 1u);
        
    if(this->ptr_array_previous_biased_first_moment != nullptr || this->ptr_array_previous_biased_second_moment != nullptr)
    {
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                    this->total_parameters_allocated,
                                                                                                    tmp_dim3_grid_zero,
                                                                                                    tmp_dim3_block_zero,
                                                                                                    tmp_dim3_grid_copy,
                                                                                                    tmp_dim3_block_copy,
                                                                                                    this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    }

    if(this->ptr_array_previous_biased_first_moment != nullptr)
    {
        var *tmp_ptr_array_previous_biased_first_moment(Memory::reallocate_cpp<var>(this->ptr_array_previous_biased_first_moment,
                                                                                                                            number_parameters_received,
                                                                                                                            this->total_parameters_allocated,
                                                                                                                            &tmp_dim3_grid_zero,
                                                                                                                            &tmp_dim3_block_zero,
                                                                                                                            &tmp_dim3_grid_copy,
                                                                                                                            &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_biased_first_moment == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_previous_biased_first_moment = tmp_ptr_array_previous_biased_first_moment;
    }
    
    if(this->ptr_array_previous_biased_second_moment != nullptr)
    {
        var *tmp_ptr_array_previous_biased_second_moment(Memory::reallocate_cpp<var>(this->ptr_array_previous_biased_second_moment,
                                                                                                                                number_parameters_received,
                                                                                                                                this->total_parameters_allocated,
                                                                                                                                &tmp_dim3_grid_zero,
                                                                                                                                &tmp_dim3_block_zero,
                                                                                                                                &tmp_dim3_grid_copy,
                                                                                                                                &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_biased_second_moment == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_previous_biased_second_moment = tmp_ptr_array_previous_biased_second_moment;
    }

    return true;
}

__device__ bool cuModel::Reallocate__Parameter__AMSGrad(size_t const number_parameters_received)
{
    struct dim3 tmp_dim3_grid_zero(1,1, 1u),
                     tmp_dim3_block_zero(1,1, 1u),
                     tmp_dim3_grid_copy(1,1, 1u),
                     tmp_dim3_block_copy(1,1, 1u);
        
    if(this->ptr_array_previous_biased_first_moment != nullptr
      ||
      this->ptr_array_previous_biased_second_moment != nullptr
      ||
      this->ptr_array_previous_biased_second_moment_hat != nullptr)
    {
        this->ptr_Class_Storage_Dim3_Memcpy->Get__Dim3_Memcpy(number_parameters_received,
                                                                                                  this->total_parameters_allocated,
                                                                                                  tmp_dim3_grid_zero,
                                                                                                  tmp_dim3_block_zero,
                                                                                                  tmp_dim3_grid_copy,
                                                                                                  tmp_dim3_block_copy,
                                                                                                  this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    }
    
    if(this->ptr_array_previous_biased_first_moment != nullptr)
    {
        var *tmp_ptr_array_previous_biased_first_moment(Memory::reallocate_cpp<var>(this->ptr_array_previous_biased_first_moment,
                                                                                                                            number_parameters_received,
                                                                                                                            this->total_parameters_allocated,
                                                                                                                            &tmp_dim3_grid_zero,
                                                                                                                            &tmp_dim3_block_zero,
                                                                                                                            &tmp_dim3_grid_copy,
                                                                                                                            &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_biased_first_moment == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_previous_biased_first_moment = tmp_ptr_array_previous_biased_first_moment;
    }
    
    if(this->ptr_array_previous_biased_second_moment != nullptr)
    {
        var *tmp_ptr_array_previous_biased_second_moment(Memory::reallocate_cpp<var>(this->ptr_array_previous_biased_second_moment,
                                                                                                                                number_parameters_received,
                                                                                                                                this->total_parameters_allocated,
                                                                                                                                &tmp_dim3_grid_zero,
                                                                                                                                &tmp_dim3_block_zero,
                                                                                                                                &tmp_dim3_grid_copy,
                                                                                                                                &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_biased_second_moment == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_previous_biased_second_moment = tmp_ptr_array_previous_biased_second_moment;
    }

    if(this->ptr_array_previous_biased_second_moment_hat != nullptr)
    {
        var *tmp_ptr_array_previous_biased_second_moment_hat(Memory::reallocate_cpp<var>(this->ptr_array_previous_biased_second_moment_hat,
                                                                                                                                        number_parameters_received,
                                                                                                                                        this->total_parameters_allocated,
                                                                                                                                        &tmp_dim3_grid_zero,
                                                                                                                                        &tmp_dim3_block_zero,
                                                                                                                                        &tmp_dim3_grid_copy,
                                                                                                                                        &tmp_dim3_block_copy));
        if(tmp_ptr_array_previous_biased_second_moment_hat == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        this->ptr_array_previous_biased_second_moment_hat = tmp_ptr_array_previous_biased_second_moment_hat;
    }

    return true;
}

[[deprecated("Not properly implemented.")]] __device__ bool cuModel::Reallocate_Connections(size_t const total_connections_received)
{
    return true;
}

[[deprecated("Not properly implemented.")]] __device__ bool cuModel::Reallocate_Neurons(size_t const total_neuron_units_received, bool const reSet__neuron_position_received)
{
    return true;
}

[[deprecated("Not properly implemented.")]] __device__ bool cuModel::Reallocate_Layers(size_t const total_layers_received)
{
    return true;
}
