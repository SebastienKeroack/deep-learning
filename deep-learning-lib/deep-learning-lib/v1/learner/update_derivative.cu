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
#include "deep-learning-lib/ops/multiply.cuh"

__device__ void cuModel::update_derivatives(size_t const batch_size, size_t const time_step_index_received) { this->FF__Update_Derivative_Weight(batch_size); }

__device__ void cuModel::FF__Update_Derivative_Weight(size_t const batch_size)
{
    // By default the synchronized state is set to true.
    bool tmp_synchronized(true);

    struct cuLayer const *const last_layer(this->ptr_last_layer);
    struct cuLayer *tmp_ptr_previous_layer_it(this->ptr_array_layers),
                                            *layer_it(tmp_ptr_previous_layer_it + 1);
    
    // Variable to cache optimal size to launch dynamic parallelisme through the GPU.
    struct dim3 tmp_dim3_grid,
                     tmp_dim3_block;

    // If we can go into dynamic parallelisme, prepare the dimension kernel.
    if(batch_size >= warpSize)
    {
        size_t const tmp_batch_size_scale(std::min<var>(batch_size, this->number_threads));

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
    
    if(this->use_Dropout)
    {
        for(; layer_it != last_layer; ++layer_it,
                                                                        ++tmp_ptr_previous_layer_it)
        {
            this->Update_Derivative_Weight__FC_to__Dropout(tmp_synchronized,
                                                                                           batch_size,
                                                                                           layer_it,
                                                                                           tmp_ptr_previous_layer_it,
                                                                                           &tmp_dim3_grid,
                                                                                           &tmp_dim3_block);
        }
    }
    else
    {
        for(; layer_it != last_layer; ++layer_it,
                                                                        ++tmp_ptr_previous_layer_it)
        {
            this->Update_Derivative_Weight__FC_to(tmp_synchronized,
                                                                            batch_size,
                                                                            layer_it,
                                                                            tmp_ptr_previous_layer_it,
                                                                            &tmp_dim3_grid,
                                                                            &tmp_dim3_block);
        }
    }
    
    // Synchronisation before using the output of the neural nework.
    CUDA__Device_Synchronise(tmp_synchronized, DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
}
    
__device__ void cuModel::Update_Derivative_Weight__FC_to(bool &ref_synchronized_received,
                                                                                                          size_t const batch_size,
                                                                                                          struct cuLayer *const layer_it,
                                                                                                          struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                                          struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                          struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    this->Update_Derivative_Weight__FC_to_FC(ref_synchronized_received,
                                                                               batch_size,
                                                                               layer_it,
                                                                               ptr_previous_layer_it_received,
                                                                               ptr_dim3_batch_size_grid_received,
                                                                               ptr_dim3_batch_size_block_received);
}
    
__device__ void cuModel::Update_Derivative_Weight__FC_to__Dropout(bool &ref_synchronized_received,
                                                                                                                         size_t const batch_size,
                                                                                                                         struct cuLayer *const layer_it,
                                                                                                                         struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                                                         struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                         struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    this->Update_Derivative_Weight__FC_to_FC__Dropout(ref_synchronized_received,
                                                                                              batch_size,
                                                                                              layer_it,
                                                                                              ptr_previous_layer_it_received,
                                                                                              ptr_dim3_batch_size_grid_received,
                                                                                              ptr_dim3_batch_size_block_received);
}
