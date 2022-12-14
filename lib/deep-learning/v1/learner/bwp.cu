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

#include "deep-learning/v1/learner/model.cuh"
#include "deep-learning/ops/zero.cuh"
#include "deep-learning/ops/multiply.cuh"

__device__ void cuModel::backward_pass(size_t const batch_size) { return(this->FF__Backward_Pass_Batch(batch_size)); }

__device__ void cuModel::FF__Backward_Pass_Batch(size_t const batch_size)
{
    // By default the synchronized state is set to true.
    bool tmp_synchronized(true);

    struct cuLayer const *const tmp_ptr_second_layer(this->ptr_array_layers + 1);
    struct cuLayer *tmp_ptr_next_layer(this->ptr_last_layer - 1),
                                           *layer_it(tmp_ptr_next_layer - 1);
    
    // Variable to cache optimal size to launch dynamic parallelisme through the GPU.
    struct dim3 tmp_dim3_grid,
                     tmp_dim3_block;

    // If we can go into dynamic parallelisme, prepare the dimension kernel.
    if(batch_size >= warpSize)
    {
        size_t const tmp_batch_size_scale(std::min<size_t>(batch_size, this->number_threads));

        if(tmp_batch_size_scale == this->number_threads)
        {
            tmp_dim3_grid = this->ptr_array_dim3_grid[7];
            tmp_dim3_block = this->ptr_array_dim3_block[7];
        }
        else
        {
            this->ptr_array_layers->ptr_Class_Storage_Dim3_Batch->Get__Dim3_Dynamic_Parallelisme(tmp_batch_size_scale,
                                                                                                                                                tmp_dim3_grid,
                                                                                                                                                tmp_dim3_block,
                                                                                                                                                this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
        }
    }
    
    // If the network use batch renormalization.
    if(this->use_Batch_Renormalization)
    {
        // Set all derivative mean to zero.
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            this->ptr_array_normalized_batch_units_derivatives_means,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Set all derivative mean to zero. |END|

        // Set all derivative variance to zero.
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            this->ptr_array_normalized_batch_units_derivatives_variances,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Set all derivative variance to zero. |END|
        
        // Do we need to synchronise? Based on "Zero_1D" Function.
        // => Synchronisation before using the derivative mean and variance of the network.
        if(this->batch_size * this->total_neuron_units_allocated >= warpSize) { tmp_synchronized = false; }

        // If the network use dropout.
        if(this->use_Dropout)
        {
            // Loop from the last layer to the second layer.
            for(; tmp_ptr_next_layer > tmp_ptr_second_layer; --tmp_ptr_next_layer,
                                                                                    --layer_it)
            {
                // If the layer use batch normalization/renormalization
                if(layer_it->use_Batch_Renormalization)
                {
                    this->Backward_Pass__FC_to__Batch_Renormalization__Dropout(tmp_synchronized,
                                                                                                                        batch_size,
                                                                                                                        layer_it,
                                                                                                                        tmp_ptr_next_layer,
                                                                                                                        &tmp_dim3_grid,
                                                                                                                        &tmp_dim3_block);
                }
                // Else serialize each sample.
                else
                {
                    this->Backward_Pass__FC_to__Dropout(tmp_synchronized,
                                                                                    batch_size,
                                                                                    layer_it,
                                                                                    tmp_ptr_next_layer,
                                                                                    &tmp_dim3_grid,
                                                                                    &tmp_dim3_block);
                }
            }
        }
        // Default backpropagation process
        else
        {
            // Loop from the last layer to the second layer.
            for(; tmp_ptr_next_layer > tmp_ptr_second_layer; --tmp_ptr_next_layer,
                                                                                    --layer_it)
            {
                // If the layer use batch normalization/renormalization
                if(layer_it->use_Batch_Renormalization)
                {
                    this->Backward_Pass__FC_to__Batch_Renormalization(tmp_synchronized,
                                                                                                            batch_size,
                                                                                                            layer_it,
                                                                                                            tmp_ptr_next_layer,
                                                                                                            &tmp_dim3_grid,
                                                                                                            &tmp_dim3_block);
                }
                // Else serialize each sample.
                else
                {
                    this->Backward_Pass__FC_to(tmp_synchronized,
                                                                        batch_size,
                                                                        layer_it,
                                                                        tmp_ptr_next_layer,
                                                                        &tmp_dim3_grid,
                                                                        &tmp_dim3_block);
                }
            }
        }
    }
    else
    {
        // If the network use dropout.
        if(this->use_Dropout)
        {
            // Loop from the last layer to the second layer.
            for(; tmp_ptr_next_layer > tmp_ptr_second_layer; --tmp_ptr_next_layer,
                                                                                    --layer_it)
            {
                this->Backward_Pass__FC_to__Dropout(tmp_synchronized,
                                                                                batch_size,
                                                                                layer_it,
                                                                                tmp_ptr_next_layer,
                                                                                &tmp_dim3_grid,
                                                                                &tmp_dim3_block);
            }
        }
        // Default backpropagation process
        else
        {
            // Loop from the last layer to the second layer.
            for(; tmp_ptr_next_layer > tmp_ptr_second_layer; --tmp_ptr_next_layer,
                                                                                    --layer_it)
            {
                this->Backward_Pass__FC_to(tmp_synchronized,
                                                                    batch_size,
                                                                    layer_it,
                                                                    tmp_ptr_next_layer,
                                                                    &tmp_dim3_grid,
                                                                    &tmp_dim3_block);
            }
        }
    }

    // Synchronisation before using the output of the neural nework.
    CUDA__Device_Synchronise(tmp_synchronized, DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
}

__device__ void cuModel::Backward_Pass__FC_to(bool &ref_synchronized_received,
                                                                                            size_t const batch_size,
                                                                                            struct cuLayer *const layer_it,
                                                                                            struct cuLayer *const ptr_next_layer_received,
                                                                                            struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                            struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    this->Backward_Pass__FC_to_FC(ref_synchronized_received, 
                                                                batch_size, 
                                                                layer_it, 
                                                                ptr_next_layer_received,
                                                                ptr_dim3_batch_size_grid_received,
                                                                ptr_dim3_batch_size_block_received);
}

__device__ void cuModel::Backward_Pass__FC_to__Dropout(bool &ref_synchronized_received,
                                                                                                            size_t const batch_size,
                                                                                                            struct cuLayer *const layer_it,
                                                                                                            struct cuLayer *const ptr_next_layer_received,
                                                                                                            struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                            struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    this->Backward_Pass__FC_to_FC__Dropout(ref_synchronized_received, 
                                                                                batch_size, 
                                                                                layer_it, 
                                                                                ptr_next_layer_received,
                                                                                ptr_dim3_batch_size_grid_received,
                                                                                ptr_dim3_batch_size_block_received);
}
    
__device__ void cuModel::Backward_Pass__FC_to__Batch_Renormalization(bool &ref_synchronized_received,
                                                                                                                                size_t const batch_size,
                                                                                                                                struct cuLayer *const layer_it,
                                                                                                                                struct cuLayer *const ptr_next_layer_received,
                                                                                                                                struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    this->Backward_Pass__FC_to_FC__Batch_Renormalization(ref_synchronized_received, 
                                                                                                      batch_size, 
                                                                                                      layer_it, 
                                                                                                      ptr_next_layer_received,
                                                                                                      ptr_dim3_batch_size_grid_received,
                                                                                                      ptr_dim3_batch_size_block_received);
}

__device__ void cuModel::Backward_Pass__FC_to__Batch_Renormalization__Dropout(bool &ref_synchronized_received,
                                                                                                                                            size_t const batch_size,
                                                                                                                                            struct cuLayer *const layer_it,
                                                                                                                                            struct cuLayer *const ptr_next_layer_received,
                                                                                                                                            struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                            struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    this->Backward_Pass__FC_to_FC__Batch_Renormalization__Dropout(ref_synchronized_received, 
                                                                                                                    batch_size, 
                                                                                                                    layer_it, 
                                                                                                                    ptr_next_layer_received,
                                                                                                                    ptr_dim3_batch_size_grid_received,
                                                                                                                    ptr_dim3_batch_size_block_received);
}
