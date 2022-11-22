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
#include "deep-learning-lib/ops/zero.cuh"

__device__ void cuModel::forward_pass(size_t const batch_size, var const *const *const ptr_array_inputs_received) { this->FF__Forward_Pass_Batch(batch_size, ptr_array_inputs_received); }

__device__ void cuModel::FF__Forward_Pass_Batch(size_t const batch_size, var const *const *const Xm)
{
    // By default the synchronized state is set to true.
    bool tmp_synchronized(true);
    
    struct cuLayer const *const last_layer(this->ptr_last_layer);
    struct cuLayer *tmp_ptr_previous_layer_it(this->ptr_array_layers),
                                            *layer_it(tmp_ptr_previous_layer_it + 1);
    
    // Variable to cache optimal size to launch dynamic parallelisme through the GPU.
    struct dim3 tmp_dim3_grid,
                     tmp_dim3_block;

    if(batch_size > this->batch_size)
    {
        ERR(L"Batch size (%u) > number threads (%u).",
                                    batch_size,
                                    this->batch_size);

        return;
    }

    // Input layer.
    this->Assign_Inputs_Batch(tmp_synchronized,
                                            batch_size,
                                            Xm);
    // |END| Input layer. |END|
    
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
        // Set all mean to zero.
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            this->ptr_array_normalized_batch_units_means,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Set all mean to zero. |END|

        // Set all variance to zero.
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            this->ptr_array_normalized_batch_units_variances,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Set all variance to zero. |END|
        
        // Do we need to synchronise? Based on "Zero_1D" Function.
        // => Synchronisation before using the mean and variance of the network.
        if(this->batch_size * this->total_neuron_units_allocated >= warpSize) { tmp_synchronized = false; }

        if(this->type_state_propagation == DL::PROPAGATION::TRAINING)
        {
            // If the network use dropout.
            if(this->use_Dropout)
            {
                // Loop from the second layer to the last layer.
                for(; layer_it != last_layer; ++layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // If the layer use batch normalization/renormalization
                    if(layer_it->use_Batch_Renormalization)
                    {
                        this->Forward_Pass__FC_to__Batch_Renormalization__Dropout(tmp_synchronized,
                                                                                                                                        batch_size,
                                                                                                                                        layer_it,
                                                                                                                                        tmp_ptr_previous_layer_it,
                                                                                                                                        &tmp_dim3_grid,
                                                                                                                                        &tmp_dim3_block);
                    }
                    // Else propagate through dropout layer.
                    else
                    {
                        // Forward propagation through a layer.
                        // With dropout regulariation. At the training state.
                        this->Forward_Pass__FC_to__Dropout(tmp_synchronized,
                                                                                                    batch_size,
                                                                                                    layer_it,
                                                                                                    tmp_ptr_previous_layer_it,
                                                                                                    &tmp_dim3_grid,
                                                                                                    &tmp_dim3_block);
                    }
                }
            }
            else
            {
                // Loop from the second layer to the last layer.
                for(; layer_it != last_layer; ++layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // If the layer use batch normalization/renormalization
                    if(layer_it->use_Batch_Renormalization)
                    {
                        this->Forward_Pass__FC_to__Batch_Renormalization__Training(tmp_synchronized,
                                                                                                                         batch_size,
                                                                                                                         layer_it,
                                                                                                                         tmp_ptr_previous_layer_it,
                                                                                                                         &tmp_dim3_grid,
                                                                                                                         &tmp_dim3_block);
                    }
                    // Else propagate through default layer.
                    else
                    {
                        // Forward propagation through a layer.
                        this->Forward_Pass__FC_to(tmp_synchronized,
                                                                        batch_size,
                                                                        layer_it,
                                                                        tmp_ptr_previous_layer_it,
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
                // Loop from the second layer to the last layer.
                for(; layer_it != last_layer; ++layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // If the layer use batch normalization/renormalization
                    if(layer_it->use_Batch_Renormalization)
                    {
                        this->Forward_Pass__FC_to__Batch_Renormalization__Dropout_Bernoulli__Testing(tmp_synchronized,
                                                                                                                                        batch_size,
                                                                                                                                        layer_it,
                                                                                                                                        tmp_ptr_previous_layer_it,
                                                                                                                                        &tmp_dim3_grid,
                                                                                                                                        &tmp_dim3_block);
                    }
                    // Else propagate through dropout layer.
                    else
                    {
                        // Forward propagation through a layer.
                        // With dropout regulariation. At the testing state.
                        this->Forward_Pass__FC_to__Dropout_Bernoulli__Testing(tmp_synchronized,
                                                                                                    batch_size,
                                                                                                    layer_it,
                                                                                                    tmp_ptr_previous_layer_it,
                                                                                                    &tmp_dim3_grid,
                                                                                                    &tmp_dim3_block);
                    }
                }
            }
            else
            {
                // Loop from the second layer to the last layer.
                for(; layer_it != last_layer; ++layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // If the layer use batch normalization/renormalization
                    if(layer_it->use_Batch_Renormalization)
                    {
                        this->Forward_Pass__FC_to__Batch_Renormalization__Loop(tmp_synchronized,
                                                                                                                         batch_size,
                                                                                                                         layer_it,
                                                                                                                         tmp_ptr_previous_layer_it,
                                                                                                                         &tmp_dim3_grid,
                                                                                                                         &tmp_dim3_block);
                    }
                    // Else propagate through default layer.
                    else
                    {
                        // Forward propagation through a layer.
                        this->Forward_Pass__FC_to(tmp_synchronized,
                                                                        batch_size,
                                                                        layer_it,
                                                                        tmp_ptr_previous_layer_it,
                                                                        &tmp_dim3_grid,
                                                                        &tmp_dim3_block);
                    }
                }
            }
        }
    }
    else
    {
        // If the network use dropout.
        if(this->use_Dropout)
        {
            if(this->type_state_propagation == DL::PROPAGATION::TRAINING)
            {
                // Loop from the second layer to the last layer.
                for(; layer_it != last_layer; ++layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // Forward propagation through a layer.
                    // With dropout regulariation. At the training state.
                    this->Forward_Pass__FC_to__Dropout(tmp_synchronized,
                                                                                                batch_size,
                                                                                                layer_it,
                                                                                                tmp_ptr_previous_layer_it,
                                                                                                &tmp_dim3_grid,
                                                                                                &tmp_dim3_block);
                }
            }
            else
            {
                // Loop from the second layer to the last layer.
                for(; layer_it != last_layer; ++layer_it,
                                                                                ++tmp_ptr_previous_layer_it)
                {
                    // Forward propagation through a layer.
                    // With dropout regulariation. At the testing state.
                    this->Forward_Pass__FC_to__Dropout_Bernoulli__Testing(tmp_synchronized,
                                                                                                batch_size,
                                                                                                layer_it,
                                                                                                tmp_ptr_previous_layer_it,
                                                                                                &tmp_dim3_grid,
                                                                                                &tmp_dim3_block);
                }
            }
        }
        else
        {
            // Loop from the second layer to the last layer.
            for(; layer_it != last_layer; ++layer_it,
                                                                            ++tmp_ptr_previous_layer_it)
            {
                // Forward propagation through a layer.
                this->Forward_Pass__FC_to(tmp_synchronized,
                                                                batch_size,
                                                                layer_it,
                                                                tmp_ptr_previous_layer_it,
                                                                &tmp_dim3_grid,
                                                                &tmp_dim3_block);
            }
        }
    }
    
    // Synchronisation before using the output of the neural nework.
    CUDA__Device_Synchronise(tmp_synchronized, DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
}

__device__ void cuModel::Forward_Pass__FC_to(bool &ref_synchronized_received,
                                                                                         size_t const batch_size,
                                                                                         struct cuLayer *const layer_it,
                                                                                         struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                         struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                         struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(layer_it->type_activation)
    {
        case DL::LAYER_ACTIVATION::SYMMETRIC:
        case DL::LAYER_ACTIVATION::ASYMMETRIC:
        case DL::LAYER_ACTIVATION::RECTIFIER:
        case DL::LAYER_ACTIVATION::SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC(ref_synchronized_received,
                                                                      batch_size,
                                                                      layer_it,
                                                                      ptr_previous_layer_it_received,
                                                                      ptr_dim3_batch_size_grid_received,
                                                                      ptr_dim3_batch_size_block_received);
                break;
        case DL::LAYER_ACTIVATION::SOFTMAX:
            this->Forward_Pass__FC_to_FC__Softmax(ref_synchronized_received,
                                                                                    batch_size,
                                                                                    layer_it,
                                                                                    ptr_previous_layer_it_received,
                                                                                    ptr_dim3_batch_size_grid_received,
                                                                                    ptr_dim3_batch_size_block_received);
                break;
        default:
            ERR(L"Can not propagate forward with (%u) as the type activation.",
                                    layer_it->type_activation);
                break;
    }
}

__device__ void cuModel::Forward_Pass__FC_to__Dropout_Bernoulli__Testing(bool &ref_synchronized_received,
                                                                                                                    size_t const batch_size,
                                                                                                                    struct cuLayer *const layer_it,
                                                                                                                    struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(layer_it->type_activation)
    {
        case DL::LAYER_ACTIVATION::SYMMETRIC:
        case DL::LAYER_ACTIVATION::ASYMMETRIC:
        case DL::LAYER_ACTIVATION::RECTIFIER:
        case DL::LAYER_ACTIVATION::SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing(ref_synchronized_received,
                                                                                                 batch_size,
                                                                                                 layer_it,
                                                                                                 ptr_previous_layer_it_received,
                                                                                                 ptr_dim3_batch_size_grid_received,
                                                                                                 ptr_dim3_batch_size_block_received);
                break;
        case DL::LAYER_ACTIVATION::SOFTMAX:
            this->Forward_Pass__FC_to_FC__Softmax(ref_synchronized_received,
                                                                                    batch_size,
                                                                                    layer_it,
                                                                                    ptr_previous_layer_it_received,
                                                                                    ptr_dim3_batch_size_grid_received,
                                                                                    ptr_dim3_batch_size_block_received);
                break;
        default:
            ERR(L"Can not propagate forward with (%u) as the type activation.",
                                    layer_it->type_activation);
                break;
    }
}
    
__device__ void cuModel::Forward_Pass__FC_to__Dropout(bool &ref_synchronized_received,
                                                                                                                     size_t const batch_size,
                                                                                                                     struct cuLayer *const layer_it,
                                                                                                                     struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                                                     struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                     struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(layer_it->type_activation)
    {
        case DL::LAYER_ACTIVATION::SYMMETRIC:
        case DL::LAYER_ACTIVATION::ASYMMETRIC:
        case DL::LAYER_ACTIVATION::RECTIFIER:
        case DL::LAYER_ACTIVATION::SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Dropout(ref_synchronized_received,
                                                                                                  batch_size,
                                                                                                  layer_it,
                                                                                                  ptr_previous_layer_it_received,
                                                                                                  ptr_dim3_batch_size_grid_received,
                                                                                                  ptr_dim3_batch_size_block_received);
                break;
        case DL::LAYER_ACTIVATION::SOFTMAX:
            this->Forward_Pass__FC_to_FC__Softmax(ref_synchronized_received,
                                                                                    batch_size,
                                                                                    layer_it,
                                                                                    ptr_previous_layer_it_received,
                                                                                    ptr_dim3_batch_size_grid_received,
                                                                                    ptr_dim3_batch_size_block_received);
                break;
        default:
            ERR(L"Can not propagate forward with (%u) as the type activation.",
                                    layer_it->type_activation);
                break;
    }
}

__device__ void cuModel::Forward_Pass__FC_to__Batch_Renormalization__Loop(bool &ref_synchronized_received,
                                                                                                                                          size_t const batch_size,
                                                                                                                                          struct cuLayer *const layer_it,
                                                                                                                                          struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                                                                          struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                          struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(layer_it->type_activation)
    {
        case DL::LAYER_ACTIVATION::SYMMETRIC:
        case DL::LAYER_ACTIVATION::ASYMMETRIC:
        case DL::LAYER_ACTIVATION::RECTIFIER:
        case DL::LAYER_ACTIVATION::SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Batch_Renormalization__Loop(ref_synchronized_received,
                                                                                                                        batch_size,
                                                                                                                        layer_it,
                                                                                                                        ptr_previous_layer_it_received,
                                                                                                                        ptr_dim3_batch_size_grid_received,
                                                                                                                        ptr_dim3_batch_size_block_received);
            break;
        default:
            ERR(L"Can not propagate forward with (%u) as the type activation.",
                                    layer_it->type_activation);
                break;
    }
}

__device__ void cuModel::Forward_Pass__FC_to__Batch_Renormalization__Dropout_Bernoulli__Testing(bool &ref_synchronized_received,
                                                                                                                                                        size_t const batch_size,
                                                                                                                                                        struct cuLayer *const layer_it,
                                                                                                                                                        struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(layer_it->type_activation)
    {
        case DL::LAYER_ACTIVATION::SYMMETRIC:
        case DL::LAYER_ACTIVATION::ASYMMETRIC:
        case DL::LAYER_ACTIVATION::RECTIFIER:
        case DL::LAYER_ACTIVATION::SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing(ref_synchronized_received,
                                                                                                                                    batch_size,
                                                                                                                                    layer_it,
                                                                                                                                    ptr_previous_layer_it_received,
                                                                                                                                    ptr_dim3_batch_size_grid_received,
                                                                                                                                    ptr_dim3_batch_size_block_received);
            break;
        default:
            ERR(L"Can not propagate forward with (%u) as the type activation.",
                                    layer_it->type_activation);
                break;
    }
}

__device__ void cuModel::Forward_Pass__FC_to__Batch_Renormalization__Training(bool &ref_synchronized_received,
                                                                                                                                            size_t const batch_size,
                                                                                                                                            struct cuLayer *const layer_it,
                                                                                                                                            struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                                                                            struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                            struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(layer_it->type_activation)
    {
        case DL::LAYER_ACTIVATION::SYMMETRIC:
        case DL::LAYER_ACTIVATION::ASYMMETRIC:
        case DL::LAYER_ACTIVATION::RECTIFIER:
        case DL::LAYER_ACTIVATION::SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Batch_Renormalization__Training(ref_synchronized_received,
                                                                                                                        batch_size,
                                                                                                                        layer_it,
                                                                                                                        ptr_previous_layer_it_received,
                                                                                                                        ptr_dim3_batch_size_grid_received,
                                                                                                                        ptr_dim3_batch_size_block_received);
                break;
        default:
            ERR(L"Can not propagate forward with (%u) as the type activation.",
                        layer_it->type_activation);
                break;
    }
}

__device__ void cuModel::Forward_Pass__FC_to__Batch_Renormalization__Dropout(bool &ref_synchronized_received,
                                                                                                                                                        size_t const batch_size,
                                                                                                                                                        struct cuLayer *const layer_it,
                                                                                                                                                        struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    switch(layer_it->type_activation)
    {
        case DL::LAYER_ACTIVATION::SYMMETRIC:
        case DL::LAYER_ACTIVATION::ASYMMETRIC:
        case DL::LAYER_ACTIVATION::RECTIFIER:
        case DL::LAYER_ACTIVATION::SELF_NORMALIZATION:
            this->Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout(ref_synchronized_received,
                                                                                                                                    batch_size,
                                                                                                                                    layer_it,
                                                                                                                                    ptr_previous_layer_it_received,
                                                                                                                                    ptr_dim3_batch_size_grid_received,
                                                                                                                                    ptr_dim3_batch_size_block_received);
                break;
        default:
            ERR(L"Can not propagate forward with (%u) as the type activation.",
                        layer_it->type_activation);
                break;
    }
}
