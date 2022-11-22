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
#include "deep-learning-lib/v1/data/datasets.cuh"
#include "deep-learning-lib/ops/fill.cuh"
#include "deep-learning-lib/ops/zero.cuh"
#include "deep-learning-lib/ops/reduce.cuh"
#include "deep-learning-lib/ops/transpose.cuh"

#include <curand_kernel.h>

__device__ void Activation_Real(var &ref_value_received,
                                              var const summation_received,
                                              enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const type_activation_function_received)
{
    switch(type_activation_function_received)
    {
        case DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::SIGMOID: ref_value_received = Activation_Function_SIGMOID_real_t<var>(summation_received); break;
        case DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::LEAKY_RELU: ref_value_received = Activation_Function_LRELU_real_t<var>(summation_received); break;
        default:
            ERR(L"Activation function (%u) not implemented yet!",
                                     type_activation_function_received);
                break;
    }
}

__device__ __host__ cuModel::cuModel(void) { }

__global__ void kernel__cuModel__Add_CUDA_Device(int const index_device_received,
                                                                                            struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received,
                                                                                            class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Add_CUDA_Device(index_device_received, ptr_struct_cudaDeviceProp_received); }
    
__device__ bool cuModel::Add_CUDA_Device(int const index_device_received, struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received)
{
    if(this->_ptr_Class_Device_Information_Array == nullptr)
    { this->_ptr_Class_Device_Information_Array = new class cuDevicesProp; }

    return(this->_ptr_Class_Device_Information_Array->push_back(index_device_received, ptr_struct_cudaDeviceProp_received));
}

__host__ bool cuModel::Initialize_CUDA_Device(void)
{
    int device_id(0),
        tmp_number_CUDA_devices;
        
    struct cudaDeviceProp tmp_struct_cudaDeviceProp,
                                     *tmp_ptr_device_struct_cudaDeviceProp(NULL);

    CUDA__Safe_Call(cudaGetDeviceCount(&tmp_number_CUDA_devices));
        
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_struct_cudaDeviceProp, sizeof(struct cudaDeviceProp)));

    for(; device_id != tmp_number_CUDA_devices; ++device_id)
    {
        CUDA__Safe_Call(cudaGetDeviceProperties(&tmp_struct_cudaDeviceProp, device_id));

        CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_struct_cudaDeviceProp,
                                                        &tmp_struct_cudaDeviceProp,
                                                        sizeof(struct cudaDeviceProp),
                                                        cudaMemcpyKind::cudaMemcpyHostToDevice));

        kernel__cuModel__Add_CUDA_Device <<< 1, 1 >>> (device_id,
                                                                                                      tmp_ptr_device_struct_cudaDeviceProp,
                                                                                                      this);
            
        CUDA__Check_Error();
    }

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_struct_cudaDeviceProp));

    return true;
}

__device__ __host__ cuModel::~cuModel(void)
{ this->Deallocate(); }

__device__ class cuDevicesProp *cuModel::Get__Class_Device_Information_Array(void) const { return(this->_ptr_Class_Device_Information_Array); }

// Public function.
__device__ bool cuModel::Set__Batch_Renormalization(size_t const index_layer_received, bool const Set__received)
{
    if(index_layer_received >= this->total_layers)
    {
        ERR(L"Layer received (%u) as argument overflow the number of layers (%u) in the neural network.",
                        index_layer_received,
                        this->total_layers);

        return false;
    }
    else if(this->ptr_array_layers == nullptr)
    {
        ERR(L"The array of layers is a nullptr.",);

        return false;
    }

    return(this->Set__Batch_Renormalization(this->ptr_array_layers + index_layer_received, Set__received));
}

// Private function.
__device__ bool cuModel::Set__Batch_Renormalization(struct cuLayer *const ptr_layer_received, bool const Set__received)
{
    struct cuLayer const *last_layer;
    struct cuLayer *layer_it;
    
    if(ptr_layer_received == nullptr)
    {
        ERR(L"Layer received as argument is a nullptr.",);

        return false;
    }
    else if(ptr_layer_received == this->ptr_array_layers)
    {
        ERR(L"Layer received as argument is the input layer.",);

        return false;
    }
    else if(ptr_layer_received == this->ptr_last_layer - 1)
    {
        ERR(L"Layer received as argument is the output layer.",);

        return false;
    }

    if(ptr_layer_received->use_Batch_Renormalization != Set__received)
    {
        ptr_layer_received->use_Batch_Renormalization = Set__received;

        if(Set__received)
        {
            if(this->use_Batch_Renormalization == false)
            {
                if(this->Allocate__Batch_Normalization() == false)
                {
                    ERR(L"From \"Allocate__Batch_Normalization\".",);

                    return false;
                }
                else if(Allocate__Neurons_Reduce_Batch_Normalization() == false)
                {
                    ERR(L"From \"Allocate__Neurons_Reduce_Batch_Normalization\".",);

                    return false;
                }

                this->use_Batch_Renormalization = true;
            }
        }
        else // Check if we use batch renormalization
        {
            // TODO: Replace the checkup by a counter.
            bool tmp_use_Batch_Renormalization(false);
        
            // Loop through each layer to do a check if a layer use batch renormalization.
            for(last_layer = this->ptr_last_layer,
                layer_it = this->ptr_array_layers; layer_it != last_layer; ++layer_it)
            {
                if(layer_it->use_Batch_Renormalization)
                {
                    tmp_use_Batch_Renormalization = true;

                    break;
                }
            }
            
            this->use_Batch_Renormalization = tmp_use_Batch_Renormalization;
            // |END| Loop through each layer to do a check if a layer use batch renormalization. |END|

            if(tmp_use_Batch_Renormalization == false)
            {
                this->Deallocate_Batch_Reduce();
                this->Deallocate__Normalized_Unit__Batch_Normalization();
                this->Remove_Batch_Normalization();
            }
        }
    }

    return true;
}

__device__ void cuModel::Transpose_Layer_Forward__Batch_Normalization(struct cuLayer *const layer_it)
{
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    Transpose::Transpose<var>(this->batch_size * *layer_it->ptr_number_neurons,
                                            this->batch_size,
                                            *layer_it->ptr_number_neurons,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                            layer_it->ptr_dim3_grid_batch_neurons,
                                            layer_it->ptr_dim3_block_batch_neurons);

    Transpose::Transpose<var>(this->batch_size * *layer_it->ptr_number_neurons,
                                            this->batch_size,
                                            *layer_it->ptr_number_neurons,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                            layer_it->ptr_dim3_grid_batch_neurons,
                                            layer_it->ptr_dim3_block_batch_neurons);
}

__device__ void cuModel::Transpose_Layer_Backward__Batch_Normalization(struct cuLayer *const layer_it)
{
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    Transpose::Transpose<var>(this->batch_size * *layer_it->ptr_number_neurons,
                                            this->batch_size,
                                            *layer_it->ptr_number_neurons,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                            layer_it->ptr_dim3_grid_batch_neurons,
                                            layer_it->ptr_dim3_block_batch_neurons);

    Transpose::Transpose<var>(this->batch_size * *layer_it->ptr_number_neurons,
                                            this->batch_size,
                                            *layer_it->ptr_number_neurons,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                            layer_it->ptr_dim3_grid_batch_neurons,
                                            layer_it->ptr_dim3_block_batch_neurons);
}

__device__ void cuModel::Transpose_Weights(void)
{
    // By default the synchronized state is set to true.
    bool tmp_synchronized(true);

    struct cuLayer const *const last_layer(this->ptr_last_layer),
                                                    *layer_it(this->ptr_array_layers + 1);
    
    var const *tmp_ptr_array_parameters(this->ptr_array_parameters);
    var *tmp_ptr_array_weights_transposed(this->ptr_array_transposed_weights);

    size_t tmp_number_weights_in_layer,
                      tmp_number_neurons_in_layer,
                      tmp_number_connections_to_each_neurons;

    for(; layer_it != last_layer; ++layer_it,
                                                                   tmp_ptr_array_parameters += tmp_number_weights_in_layer,
                                                                   tmp_ptr_array_weights_transposed += tmp_number_weights_in_layer)
    {
        tmp_number_neurons_in_layer = *layer_it->ptr_number_neurons - 1_UZ; // Subtract bias.
        
        tmp_number_connections_to_each_neurons = *layer_it->ptr_array_neuron_units->ptr_number_forward_connections;

        tmp_number_weights_in_layer = tmp_number_neurons_in_layer * tmp_number_connections_to_each_neurons;

        Transpose::Transpose<var>(tmp_number_weights_in_layer,
                                                 tmp_number_neurons_in_layer,
                                                 tmp_number_connections_to_each_neurons,
                                                 tmp_ptr_array_weights_transposed,
                                                 tmp_ptr_array_parameters,
                                                 layer_it->ptr_dim3_grid_weights,
                                                 layer_it->ptr_dim3_block_weights);

        //INFO(L"Transposed");

        // Do we need to synchronise? Based on "Transpose" Function.
        // => Synchronisation before using the transposed weights of the layer.
        if(tmp_number_weights_in_layer >= warpSize) { tmp_synchronized = false; }
    }

    CUDA__Device_Synchronise(tmp_synchronized, DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
}

__device__ void cuModel::Prepare__Global__Grids_Blocks_Dimensions(void)
{
    this->Prepare__Layers__Grids_Blocks_Dimensions();
    this->Prepare__Neurons__Grids_Blocks_Dimensions();
    this->Prepare__Parameters__Grids_Blocks_Dimensions();

    this->Prepare__Threads__Grids_Blocks_Dimensions(this->number_threads);
    this->Prepare__Batch__Grids_Blocks_Dimensions(this->batch_size);
}

__device__ bool cuModel::Prepare__Layers__Grids_Blocks_Dimensions(void)
{
    size_t tmp_number_neurons_in_layer,
                      tmp_number_connections_to_each_neurons;

    struct cuLayer const *last_layer(this->ptr_last_layer);
    struct cuLayer *layer_it(this->ptr_array_layers);
    
    class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    for(; layer_it != last_layer; ++layer_it)
    {
        if((tmp_number_neurons_in_layer = *layer_it->ptr_number_neurons) != 0u)
        {
            --tmp_number_neurons_in_layer; // Subtract bias.
            
            tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(tmp_number_neurons_in_layer,
                                                                                     0,
                                                                                     *layer_it->ptr_dim3_grid_neurons,
                                                                                     *layer_it->ptr_dim3_block_neurons);
            
            tmp_ptr_CUDA_Device->Grid_Block_Dynamic_Parallelisme(tmp_number_neurons_in_layer,
                                                                                                   0,
                                                                                                   *layer_it->ptr_dim3_grid_neurons_DP,
                                                                                                   *layer_it->ptr_dim3_block_neurons_DP);
            
            tmp_ptr_CUDA_Device->Grid_Block_cuRAND_1Dimensions(tmp_number_neurons_in_layer,
                                                                                                   0,
                                                                                                   *layer_it->ptr_dim3_grid_neurons_cuRAND,
                                                                                                   *layer_it->ptr_dim3_block_neurons_cuRAND);
            
            tmp_number_connections_to_each_neurons = *layer_it->ptr_array_neuron_units->ptr_number_forward_connections;

            // If layer have some weights.
            if(tmp_number_neurons_in_layer * tmp_number_connections_to_each_neurons != 0u)
            {
                tmp_ptr_CUDA_Device->Grid_Block_Transpose_2Dimensions(tmp_number_neurons_in_layer,
                                                                                                          tmp_number_connections_to_each_neurons,
                                                                                                          0,
                                                                                                          *layer_it->ptr_dim3_grid_weights,
                                                                                                          *layer_it->ptr_dim3_block_weights);
            }
        }
    }

    return true;
}

__device__ bool cuModel::Prepare__Neurons__Grids_Blocks_Dimensions(void)
{
    class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    if(this->total_neuron_units != 0_UZ)
    {
        struct cuNeuron *tmp_ptr_neuron_unit_it((this->ptr_array_layers + 1)->ptr_array_neuron_units);
        struct cuNeuron const *tmp_ptr_last_neuron_unit((this->ptr_last_layer - 1)->ptr_last_neuron_unit - 1); // Subtract bias.
        
        // Grid | Block: [3]: Total neurons.
        tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(this->total_neuron_units,
                                                                                 0,
                                                                                 this->ptr_array_dim3_grid[3],
                                                                                 this->ptr_array_dim3_block[3]);
        
        // Grid | Block: [6]: Max norm constraints.
        tmp_ptr_CUDA_Device->Grid_Block_Dynamic_Parallelisme(this->total_neuron_units - this->n_inp - 1,
                                                                                               0,
                                                                                               this->ptr_array_dim3_grid[6],
                                                                                               this->ptr_array_dim3_block[6]);
        
        if(this->total_neuron_units_allocated != 0u)
        {
            for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
            {
                if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections != 0u) // If is not a bias.
                {
                    tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections,
                                                                                             0,
                                                                                             *tmp_ptr_neuron_unit_it->ptr_dim3_grid_connections,
                                                                                             *tmp_ptr_neuron_unit_it->ptr_dim3_block_connections);
                }
            }
        }
    }

    return true;
}

__device__ void cuModel::Prepare__Parameters__Grids_Blocks_Dimensions(void)
{
    class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    
    // Grid | Block: [1]: Total parameters.
    tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(this->total_parameters,
                                                                             0,
                                                                             this->ptr_array_dim3_grid[1],
                                                                             this->ptr_array_dim3_block[1]);
    
    // Grid | Block: [2]: Total weights.
    tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(this->total_weights,
                                                                             0,
                                                                             this->ptr_array_dim3_grid[2],
                                                                             this->ptr_array_dim3_block[2]);
    
    // Grid | Block: [9]: Total weights cuRAND MTGP32.
    tmp_ptr_CUDA_Device->Grid_Block_cuRAND_1Dimensions(this->total_weights,
                                                                                            0,
                                                                                            this->ptr_array_dim3_grid[8],
                                                                                            this->ptr_array_dim3_block[8]);
}

__device__ void cuModel::Prepare__Threads__Grids_Blocks_Dimensions(size_t const number_threads_received)
{
    class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
    
    // Grid | Block: [0]: Total threads
    tmp_ptr_CUDA_Device->Grid_Block_1Dimensions(number_threads_received,
                                                                             0,
                                                                             this->ptr_array_dim3_grid[0],
                                                                             this->ptr_array_dim3_block[0]);
    
    // Grid | Block: [7]: Total threads DP
    tmp_ptr_CUDA_Device->Grid_Block_Dynamic_Parallelisme(number_threads_received,
                                                                                           0,
                                                                                           this->ptr_array_dim3_grid[7],
                                                                                           this->ptr_array_dim3_block[7]);
    
    this->Prepare__Threads_Parameters__Grids_Blocks_Dimensions(number_threads_received);
}

__device__ void cuModel::Prepare__Threads_Parameters__Grids_Blocks_Dimensions(size_t const number_threads_received)
{
    // Grid | Block: [3]: (threads - 1) * total parameters
    if(number_threads_received > 1u)
    {
        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions((number_threads_received - 1u) * this->total_parameters,
                                                                                                                                               0,
                                                                                                                                               this->ptr_array_dim3_grid[4],
                                                                                                                                               this->ptr_array_dim3_block[4]);
    }
    else
    {
        this->ptr_array_dim3_grid[4].x = 1u;
        this->ptr_array_dim3_grid[4].y = 1u;
        this->ptr_array_dim3_grid[4].z = 1u;

        this->ptr_array_dim3_block[4].x = 1u;
        this->ptr_array_dim3_block[4].y = 1u;
        this->ptr_array_dim3_block[4].z = 1u;
    }
}

__device__ void cuModel::Prepare__Batch__Grids_Blocks_Dimensions(size_t const batch_size)
{
    this->Prepare__Batch_Neurons__Grids_Blocks_Dimensions(batch_size);
    this->Prepare__Batch_Layers__Grids_Blocks_Dimensions(batch_size);
}

__device__ void cuModel::Prepare__Batch_Layers__Grids_Blocks_Dimensions(size_t const batch_size)
{
    size_t tmp_number_neurons_in_layer;

    struct cuLayer const *last_layer(this->ptr_last_layer);
    struct cuLayer *layer_it(this->ptr_array_layers);
    
    class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    for(; layer_it != last_layer; ++layer_it)
    {
        if((tmp_number_neurons_in_layer = *layer_it->ptr_number_neurons) != 0u)
        {
            tmp_ptr_CUDA_Device->Grid_Block_Transpose_2Dimensions(tmp_number_neurons_in_layer,
                                                                                                      batch_size,
                                                                                                      0,
                                                                                                      *layer_it->ptr_dim3_grid_batch_neurons,
                                                                                                      *layer_it->ptr_dim3_block_batch_neurons);
        }
    }
}

__device__ void cuModel::Prepare__Batch_Neurons__Grids_Blocks_Dimensions(size_t const batch_size)
{
    if(this->total_neuron_units != 0_UZ)
    {
        // Grid | Block: [5]: batch * total neurons
        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(batch_size * this->total_neuron_units,
                                                                                                                                               0,
                                                                                                                                               this->ptr_array_dim3_grid[5],
                                                                                                                                               this->ptr_array_dim3_block[5]);
    }
}

__global__ void kernel__cuModel__Set__Normalization_Momentum_Average(var const momentum_average_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Set__Normalization_Momentum_Average(momentum_average_received); }

__host__ __device__ bool cuModel::Set__Normalization_Momentum_Average(var const momentum_average_received)
{
#ifndef __CUDA_ARCH__
    kernel__cuModel__Set__Normalization_Momentum_Average <<< 1, 1 >>> (momentum_average_received, this);

    CUDA__Check_Error();
#else
    if(this->normalization_momentum_average == momentum_average_received) { return true; }

    this->normalization_momentum_average = momentum_average_received;
#endif

    return true;
}
    
__global__ void kernel__cuModel__Set__Normalization_Epsilon(var const epsilon_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Set__Normalization_Epsilon(epsilon_received); }

__host__ __device__ bool cuModel::Set__Normalization_Epsilon(var const epsilon_received)
{
#ifndef __CUDA_ARCH__
    kernel__cuModel__Set__Normalization_Epsilon <<< 1, 1 >>> (epsilon_received, this);

    CUDA__Check_Error();
#else
    if(this->normalization_epsilon == epsilon_received) { return true; }

    this->normalization_epsilon = epsilon_received;
#endif

    return true;
}
    
__global__ void kernel__cuModel__Set__Batch_Renormalization_r_Correction_Maximum(var const r_correction_maximum_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Set__Batch_Renormalization_r_Correction_Maximum(r_correction_maximum_received); }

__host__ __device__ bool cuModel::Set__Batch_Renormalization_r_Correction_Maximum(var const r_correction_maximum_received)
{
#ifndef __CUDA_ARCH__
    kernel__cuModel__Set__Batch_Renormalization_r_Correction_Maximum <<< 1, 1 >>> (r_correction_maximum_received, this);

    CUDA__Check_Error();
#else
    if(this->batch_renormalization_r_correction_maximum == r_correction_maximum_received) { return true; }

    this->batch_renormalization_r_correction_maximum = r_correction_maximum_received;
#endif

    return true;
}
    
__global__ void kernel__cuModel__Set__Batch_Renormalization_d_Correction_Maximum(var const d_correction_maximum_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Set__Batch_Renormalization_d_Correction_Maximum(d_correction_maximum_received); }

__host__ __device__ bool cuModel::Set__Batch_Renormalization_d_Correction_Maximum(var const d_correction_maximum_received)
{
#ifndef __CUDA_ARCH__
    kernel__cuModel__Set__Batch_Renormalization_d_Correction_Maximum <<< 1, 1 >>> (d_correction_maximum_received, this);

    CUDA__Check_Error();
#else
    if(this->batch_renormalization_d_correction_maximum == d_correction_maximum_received) { return true; }

    this->batch_renormalization_d_correction_maximum = d_correction_maximum_received;
#endif

    return true;
}

__global__ void kernel__cuModel__Set__Regularization__Weight_Decay(var const regularization__weight_decay_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Set__Regularization__Weight_Decay(regularization__weight_decay_received); }

__host__ __device__ bool cuModel::Set__Regularization__Weight_Decay(var const regularization__weight_decay_received)
{
#ifndef __CUDA_ARCH__
    kernel__cuModel__Set__Regularization__Weight_Decay <<< 1, 1 >>> (regularization__weight_decay_received, this);

    CUDA__Check_Error();
#else
    if(this->weight_decay != regularization__weight_decay_received)
    {
        bool const tmp_use_regularization(this->Use__Regularization_Parameter()),
                        tmp_not_initialized_regularization(this->ptr_array_mask_regularized_parameters == nullptr);

        this->weight_decay = regularization__weight_decay_received;

        if(tmp_use_regularization == false && regularization__weight_decay_received != 0_r)
        {
            if(this->Allocate__Parameter__Regularization() == false)
            {
                ERR(L"Can not allocate regularization connections!",);
        
                return false;
            }

            if(tmp_not_initialized_regularization) { this->Indexing_Regularization_Parameters(); }
        }

        if(this->Use__Regularization_Parameter() == false)
        { this->Deallocate__Parameter__Regularization(); }
    }
#endif

    return true;
}

__device__ bool cuModel::Use__Regularization_Parameter(void) const
{
    if(this->regularization__l1 != 0_r
        ||
        this->regularization__l2 != 0_r
        ||
        this->weight_decay != 0_r)
    { return true; }
    
    return false;
}

__device__ void cuModel::Indexing_Regularization_Parameters(void)
{
    struct cuLayer const *const last_layer(this->ptr_last_layer),
                                                    *layer_it(this->ptr_array_layers + 1);
    
    for(; layer_it != last_layer; ++layer_it)
    {
        switch(layer_it->type_layer)
        {
            case DL::LAYER::FULLY_CONNECTED: this->Indexing_Regularization__Weights__FC__Forward(layer_it); break;
            //case DL::LAYER::LSTM: this->Indexing_Regularization__Weights__LSTM(layer_it); break;
        }
    }
        
    // Mask all others parameters that is not a weight.
    var const *tmp_ptr_last_mask_regularization(this->ptr_array_mask_regularized_parameters + this->total_parameters_allocated);
    var *tmp_ptr_mask_regularization_it(this->ptr_array_mask_regularized_parameters + this->total_weights_allocated);
    
    if(this->total_parameters_allocated - this->total_weights_allocated >= warpSize)
    {
        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;
        
        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(this->total_parameters_allocated - this->total_weights_allocated,
                                                                                                                                                0,
                                                                                                                                                tmp_dim3_grid,
                                                                                                                                                tmp_dim3_block);

        Zero_1D<var>(this->total_parameters_allocated - this->total_weights_allocated,
                            tmp_ptr_mask_regularization_it,
                            &tmp_dim3_grid,
                            &tmp_dim3_block);
    }
    else
    {
        for(; tmp_ptr_mask_regularization_it != tmp_ptr_last_mask_regularization; ++tmp_ptr_mask_regularization_it)
        { *tmp_ptr_mask_regularization_it = 0_r; }
    }
    // |END| Mask all others parameters that is not a weight. |END|
}

template<typename T>
__global__ void kernel__cuModel__Indexing_Regularization__Weights__FC(T *const ptr_array_mask_rergularization_parameters_received,
                                                                                                                            size_t const number_connections_received,
                                                                                                                            size_t const *const ptr_array_first_connection_index_received,
                                                                                                                            size_t const *const ptr_array_last_connection_index_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    Memory::Fill_1D<var>(number_connections_received,
                                    ptr_array_mask_rergularization_parameters_received + ptr_array_first_connection_index_received[tmp_thread_global_index],
                                    1_r,
                                    ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                    ptr_array_dim3_block_connections_received + tmp_thread_global_index);

    ptr_array_mask_rergularization_parameters_received[ptr_array_last_connection_index_received[tmp_thread_global_index] - 1u] = 0_r; // Bias.
}

template<typename T>
__global__ void kernel__cuModel__Indexing_Regularization__Weights__FC(size_t const size_received,
                                                                                                                            T *const ptr_array_mask_rergularization_parameters_received,
                                                                                                                            size_t const number_connections_received,
                                                                                                                            size_t const *const ptr_array_first_connection_index_received,
                                                                                                                            size_t const *const ptr_array_last_connection_index_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    if(tmp_thread_global_index < size_received)
    {
        Memory::Fill_1D<var>(number_connections_received,
                                                             ptr_array_mask_rergularization_parameters_received + ptr_array_first_connection_index_received[tmp_thread_global_index],
                                                             1_r,
                                                             ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                             ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_rergularization_parameters_received[ptr_array_last_connection_index_received[tmp_thread_global_index] - 1u] = 0_r; // Bias.
    }
}

template<typename T>
__global__ void kernel_while__cuModel__Indexing_Regularization__Weights__FC(size_t const size_received,
                                                                                                                                       T *const ptr_array_mask_rergularization_parameters_received,
                                                                                                                                       size_t const number_connections_received,
                                                                                                                                       size_t const *const ptr_array_first_connection_index_received,
                                                                                                                                       size_t const *const ptr_array_last_connection_index_received,
                                                                                                                                       struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                       struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    do
    {
        Memory::Fill_1D<var>(number_connections_received,
                                                             ptr_array_mask_rergularization_parameters_received + ptr_array_first_connection_index_received[tmp_thread_global_index],
                                                             1_r,
                                                             ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                             ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_rergularization_parameters_received[ptr_array_last_connection_index_received[tmp_thread_global_index] - 1u] = 0_r; // Bias.

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::Indexing_Regularization__Weights__FC__Forward(struct cuLayer const *const layer_it)
{
    struct cuNeuron const *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    if(*layer_it->ptr_number_neurons - 1u >= warpSize)
    {
        LAUNCH_KERNEL_POINTER_1D(cuModel__Indexing_Regularization__Weights__FC<var>,
                                                          layer_it->ptr_dim3_grid_neurons_DP,
                                                          layer_it->ptr_dim3_block_neurons_DP,
                                                          0_UZ,
                                                          *layer_it->ptr_number_neurons - 1,
                                                          this->ptr_array_mask_regularized_parameters,
                                                          *tmp_ptr_layer_it_first_neuron->ptr_number_forward_connections - 1, // Subtract bias.
                                                          tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                          tmp_ptr_layer_it_first_neuron->ptr_last_forward_connection_index,
                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections)
    }
    else
    {
        size_t const *tmp_ptr_array_first_connection_index(tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index),
                                    *const tmp_ptr_array_first_connection_index_last(tmp_ptr_array_first_connection_index + *layer_it->ptr_number_neurons - 1u), // Subtract bias.
                                    *tmp_ptr_array_last_connection_index(tmp_ptr_layer_it_first_neuron->ptr_last_forward_connection_index),
                                    tmp_number_connections(*tmp_ptr_layer_it_first_neuron->ptr_number_forward_connections - 1u); // Subtract bias.

        struct dim3 const *tmp_ptr_array_dim3_grid_connections(tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections),
                                  *tmp_ptr_array_dim3_block_connections(tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);

        for(; tmp_ptr_array_first_connection_index != tmp_ptr_array_first_connection_index_last; ++tmp_ptr_array_first_connection_index,
                                                                                                                                        ++tmp_ptr_array_last_connection_index,
                                                                                                                                        ++tmp_ptr_array_dim3_grid_connections,
                                                                                                                                        ++tmp_ptr_array_dim3_block_connections)
        {
            Memory::Fill_1D<var>(tmp_number_connections,
                                                                 this->ptr_array_mask_regularized_parameters + *tmp_ptr_array_first_connection_index,
                                                                 1_r,
                                                                 tmp_ptr_array_dim3_grid_connections,
                                                                 tmp_ptr_array_dim3_block_connections);

            this->ptr_array_mask_regularized_parameters[*tmp_ptr_array_last_connection_index - 1u] = 0_r; // Bias.
        }
    }
}

__device__ bool cuModel::Multi_Class_Classification(void) const
{ return(this->n_out > 1u); }

__device__ void cuModel::Remove_Batch_Normalization(void)
{
    if(this->ptr_array_parameters != nullptr)
    {
        size_t const tmp_new_size(this->total_parameters_allocated - 2u * this->total_neuron_units_allocated);
        
        if(this->Reallocate__Parameter(tmp_new_size) == false)
        {
            ERR(L"From \"Reallocate__Parameter\".",);

            return;
        }
    }
}

template<typename T>
__global__ void kernel__cuModel__Reset__Parameters_Neurons_Batch_Normalization(T *const ptr_array_parameters_scale_it_received,
                                                                                                                                                              T *const ptr_array_parameters_shift_it_received,
                                                                                                                                                              struct cuNeuron *const ptr_array_neuron_units_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    ptr_array_neuron_units_received[tmp_thread_global_index].ptr_scale = ptr_array_parameters_scale_it_received + tmp_thread_global_index;
    ptr_array_neuron_units_received[tmp_thread_global_index].ptr_shift = ptr_array_parameters_shift_it_received + tmp_thread_global_index;
}

template<typename T>
__global__ void kernel__cuModel__Reset__Parameters_Neurons_Batch_Normalization(size_t const size_received,
                                                                                                                                                              T *const ptr_array_parameters_scale_it_received,
                                                                                                                                                              T *const ptr_array_parameters_shift_it_received,
                                                                                                                                                              struct cuNeuron *const ptr_array_neuron_units_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_scale = ptr_array_parameters_scale_it_received + tmp_thread_global_index;
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_shift = ptr_array_parameters_shift_it_received + tmp_thread_global_index;
    }
}

template<typename T>
__global__ void kernel_while__cuModel__Reset__Parameters_Neurons_Batch_Normalization(size_t const size_received,
                                                                                                                                                                      T *const ptr_array_parameters_scale_it_received,
                                                                                                                                                                      T *const ptr_array_parameters_shift_it_received,
                                                                                                                                                                      struct cuNeuron *const ptr_array_neuron_units_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_scale = ptr_array_parameters_scale_it_received + tmp_thread_global_index;
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_shift = ptr_array_parameters_shift_it_received + tmp_thread_global_index;

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::Reset__Parameter__Normalized_Unit(void)
{
    var *tmp_ptr_array_parameters_scale_it(this->ptr_array_parameters + this->total_weights_allocated),
        *tmp_ptr_array_parameters_shift_it(this->ptr_array_parameters + this->total_weights_allocated + this->total_neuron_units_allocated);

    struct cuNeuron *tmp_ptr_neuron_unit_it(this->ptr_array_layers->ptr_array_neuron_units);
    struct cuNeuron const *const tmp_ptr_last_neuron_unit(tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated);
    
    if(USE_PARALLEL && this->total_neuron_units_allocated >= warpSize)
    {
        LAUNCH_KERNEL_1D(cuModel__Reset__Parameters_Neurons_Batch_Normalization<var>,
                                          this->ptr_array_dim3_grid[3],
                                          this->ptr_array_dim3_block[3],
                                          0_UZ,
                                          this->total_neuron_units_allocated,
                                          tmp_ptr_array_parameters_scale_it,
                                          tmp_ptr_array_parameters_shift_it,
                                          tmp_ptr_neuron_unit_it)

        CUDA__Check_Error();
    }
    else
    {
        for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                ++tmp_ptr_array_parameters_scale_it,
                                                                                ++tmp_ptr_array_parameters_shift_it)
        {
            tmp_ptr_neuron_unit_it->ptr_scale = tmp_ptr_array_parameters_scale_it;
            tmp_ptr_neuron_unit_it->ptr_shift = tmp_ptr_array_parameters_shift_it;
        }
    }
}

template<typename T>
__global__ void kernel__cuModel__Reset__Derivatives_Parameters_Neurons_Batch_Normalization(T *const ptr_array_derivatives_parameters_scale_it_received,
                                                                                                                                                              T *const ptr_array_derivatives_parameters_shift_it_received,
                                                                                                                                                              struct cuNeuron *const ptr_array_neuron_units_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_scales = ptr_array_derivatives_parameters_scale_it_received + tmp_thread_global_index;
    ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_shifts = ptr_array_derivatives_parameters_shift_it_received + tmp_thread_global_index;
}

template<typename T>
__global__ void kernel__cuModel__Reset__Derivatives_Parameters_Neurons_Batch_Normalization(size_t const size_received,
                                                                                                                                                              T *const ptr_array_derivatives_parameters_scale_it_received,
                                                                                                                                                              T *const ptr_array_derivatives_parameters_shift_it_received,
                                                                                                                                                              struct cuNeuron *const ptr_array_neuron_units_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_scales = ptr_array_derivatives_parameters_scale_it_received + tmp_thread_global_index;
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_shifts = ptr_array_derivatives_parameters_shift_it_received + tmp_thread_global_index;
    }
}

template<typename T>
__global__ void kernel_while__cuModel__Reset__Derivatives_Parameters_Neurons_Batch_Normalization(size_t const size_received,
                                                                                                                                                                      T *const ptr_array_derivatives_parameters_scale_it_received,
                                                                                                                                                                      T *const ptr_array_derivatives_parameters_shift_it_received,
                                                                                                                                                                      struct cuNeuron *const ptr_array_neuron_units_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_scales = ptr_array_derivatives_parameters_scale_it_received + tmp_thread_global_index;
        ptr_array_neuron_units_received[tmp_thread_global_index].ptr_array_derivatives_shifts = ptr_array_derivatives_parameters_shift_it_received + tmp_thread_global_index;

        tmp_thread_global_index += tmp_grid_stride;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::Reset__Derivative_Parameter__Normalized_Unit(void)
{
    var *tmp_ptr_array_derivatives_parameters_scale_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated),
         *tmp_ptr_array_derivatives_parameters_shift_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated + this->total_neuron_units_allocated);

    struct cuNeuron *tmp_ptr_neuron_unit_it(this->ptr_array_layers->ptr_array_neuron_units);
    struct cuNeuron const *const tmp_ptr_last_neuron_unit(tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated);
    
    if(USE_PARALLEL && this->total_neuron_units_allocated >= warpSize)
    {
        LAUNCH_KERNEL_1D(cuModel__Reset__Derivatives_Parameters_Neurons_Batch_Normalization<var>,
                                          this->ptr_array_dim3_grid[3],
                                          this->ptr_array_dim3_block[3],
                                          0_UZ,
                                          this->total_neuron_units_allocated,
                                          tmp_ptr_array_derivatives_parameters_scale_it,
                                          tmp_ptr_array_derivatives_parameters_shift_it,
                                          tmp_ptr_neuron_unit_it)

        CUDA__Check_Error();
    }
    else
    {
        for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                             ++tmp_ptr_array_derivatives_parameters_scale_it,
                                                                             ++tmp_ptr_array_derivatives_parameters_shift_it)
        {
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it;
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it;
        }
    }
}

__global__ void kernel__cuModel__Get__Limit_Device_Runtime_Pending_Launch_Count(size_t *const ptr_limit_device_runtime_pending_launch_count_received, class cuModel *const ptr_cuModel_received)
{ *ptr_limit_device_runtime_pending_launch_count_received = ptr_cuModel_received->Get__Limit_Device_Runtime_Pending_Launch_Count(); }

__device__ size_t cuModel::Get__Limit_Device_Runtime_Pending_Launch_Count(void)
{ return(this->limit_device_runtime_pending_launch_count); }

__host__ void cuModel::Set__Limit_Device_Runtime_Pending_Launch_Count(size_t limit_device_runtime_pending_launch_count_received)
{
    if(limit_device_runtime_pending_launch_count_received == 0u)
    {
        size_t *tmp_ptr_limit_device_runtime_pending_launch_count;

        CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_limit_device_runtime_pending_launch_count, sizeof(size_t)));

        kernel__cuModel__Get__Limit_Device_Runtime_Pending_Launch_Count <<< 1, 1 >>> (tmp_ptr_limit_device_runtime_pending_launch_count, this);
            
        CUDA__Check_Error();

        CUDA__Safe_Call(cudaMemcpy(&limit_device_runtime_pending_launch_count_received,
                                                        tmp_ptr_limit_device_runtime_pending_launch_count,
                                                        sizeof(size_t),
                                                        cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }

    CUDA__Safe_Call(cudaDeviceSetLimit(cudaLimit::cudaLimitDevRuntimePendingLaunchCount, limit_device_runtime_pending_launch_count_received));
}
    
__global__ void kernel__cuModel__Set__Available_Memory(size_t const available_memory_mbs_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Set__Maximum_Allowable_Memory(available_memory_mbs_received); }

__host__ __device__ void cuModel::Set__Maximum_Allowable_Memory(size_t const available_memory_mbs_received)
{
#ifdef __CUDA_ARCH__
    this->maximum_allowable_memory_bytes = available_memory_mbs_received;
#else
    kernel__cuModel__Set__Available_Memory <<< 1, 1 >>> (available_memory_mbs_received, this);
        
    CUDA__Check_Error();
#endif
}

__device__ void cuModel::merge_mp_derivatives(void)
{
    Reduce::Reduce_Array(this->number_threads,
                                       this->total_parameters_allocated,
                                       1_UZ,
                                       this->ptr_array_derivatives_parameters,
                                       this->ptr_array_dim3_grid_reduce_threads_DP,
                                       this->ptr_array_dim3_block_reduce_threads_DP,
                                       this->ptr_array_dim3_grid + 1,
                                       this->ptr_array_dim3_block + 1);

    Zero_1D<var>((this->number_threads - 1_UZ) * this->total_parameters_allocated,
                         this->ptr_array_derivatives_parameters + this->total_parameters_allocated,
                         this->ptr_array_dim3_grid + 4,
                         this->ptr_array_dim3_block + 4);
}

__Lch_Bds__(MAXIMUM_THREADS_PER_BLOCK, 1)
__global__ void kernel__cuModel__Update_Threads_Size(size_t const number_threads_received, class cuModel *const ptr_cuModel_received)
{
    if(ptr_cuModel_received->update_mem_thread_size(number_threads_received) == false)
    {
        ERR(L"From \"update_mem_thread_size\"",);
    }
}

__device__ void Compute_Minimum_Threads_Block_Requirements(size_t const number_threads_needed_per_example_received,
                                                                                                 size_t const number_grids_launch_needed_per_example_received,
                                                                                                 size_t const minimum_threads_trigger_received,
                                                                                                 size_t &ref_minimum_threads_per_example_received,
                                                                                                 size_t &ref_maximum_grids_launch_per_example_received)
{
    // If number of threads need per example is more that trigger.
    if(number_threads_needed_per_example_received > minimum_threads_trigger_received
        &&
    // Minimum number of threads per example is bigger than the argument.
        ref_minimum_threads_per_example_received > number_threads_needed_per_example_received)
    // Then assign minimum number of threads per example at the argument.
    { ref_minimum_threads_per_example_received = number_threads_needed_per_example_received; }

    // Maximum number of grids launch per example is smaller than the argument.
    if(ref_maximum_grids_launch_per_example_received < number_grids_launch_needed_per_example_received)
    // Then assign maximum number of grids launch per example at the argument.
    { ref_maximum_grids_launch_per_example_received = number_grids_launch_needed_per_example_received; }
}

__host__ __device__ bool cuModel::update_mem_thread_size(size_t number_threads_received)
{
#ifdef __CUDA_ARCH__
    if(number_threads_received <= this->cache_number_threads) { return true; }
    
    size_t const tmp_number_concurrent_kernel(this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Get__Number_Concurrent_Kernel());
    size_t tmp_number_threads,
        /* Minimum threads required per example for processing propagation through
           the neural network in the forward and backward passes.
           For example: The number of neurons in a layer that is parallelizable. */
              tmp_minimum_threads_per_example(std::numeric_limits<size_t>::max()),
              tmp_maximum_grids_launch_per_example(-std::numeric_limits<size_t>::max());
    
    struct cuLayer const *const last_layer(this->ptr_last_layer),
                                          *layer_it(this->ptr_array_layers);

    struct cuNeuron const *tmp_ptr_neuron_unit_it;

    class cuDeviceProp *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

    switch(this->type)
    {
        case DL::MODEL::FEEDFORWARD:
            // Loop through each layer.
            for(; layer_it != last_layer; ++layer_it)
            {
                // Store pointer of the first neuron of the dense layer.
                tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units;

                // If use neurons parallelisme (Subtract bias.)
                if(*layer_it->ptr_number_neurons - 1u >= tmp_ptr_CUDA_Device->Get__Warp_Size())
                {
                    // If use connections parallelisme. (Reduce, FMAC...)
                    if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections >= tmp_ptr_CUDA_Device->Get__Warp_Size())
                    {
                        Compute_Minimum_Threads_Block_Requirements(layer_it->ptr_dim3_grid_neurons->x * layer_it->ptr_dim3_block_neurons->x,
                                                                                                  layer_it->ptr_dim3_grid_neurons->x * layer_it->ptr_dim3_block_neurons->x,
                                                                                                  tmp_ptr_CUDA_Device->Get__Warp_Size(),
                                                                                                  tmp_minimum_threads_per_example,
                                                                                                  tmp_maximum_grids_launch_per_example);
                    }
                    else
                    {
                        Compute_Minimum_Threads_Block_Requirements(layer_it->ptr_dim3_grid_neurons->x * layer_it->ptr_dim3_block_neurons->x,
                                                                                                  1_UZ,
                                                                                                  tmp_ptr_CUDA_Device->Get__Warp_Size(),
                                                                                                  tmp_minimum_threads_per_example,
                                                                                                  tmp_maximum_grids_launch_per_example);
                    }
                }
                // If use connections parallelisme. (Reduce, FMAC...)
                else if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections >= tmp_ptr_CUDA_Device->Get__Warp_Size())
                {
                    Compute_Minimum_Threads_Block_Requirements(1_UZ,
                                                                                              layer_it->ptr_dim3_grid_neurons->x * layer_it->ptr_dim3_block_neurons->x,
                                                                                              tmp_ptr_CUDA_Device->Get__Warp_Size(),
                                                                                              tmp_minimum_threads_per_example,
                                                                                              tmp_maximum_grids_launch_per_example);
                }
            }

            if(tmp_minimum_threads_per_example == std::numeric_limits<size_t>::max())
            { tmp_minimum_threads_per_example = 1_UZ; }

            if(tmp_maximum_grids_launch_per_example == -std::numeric_limits<size_t>::max())
            { tmp_maximum_grids_launch_per_example = 1_UZ; }
                break;
        default:
            ERR(L"... with %d as the type network.",
                                     this->type);
                return false;
    }
    
    // Divide the total threads by the number of threads needed per exemple.
    tmp_number_threads = static_cast<size_t>(ceil(static_cast<double>(tmp_ptr_CUDA_Device->Get__Maximum_Threads()) / static_cast<double>(tmp_minimum_threads_per_example)));

    // Don't overflow the number of exemple received as argument.
    if(tmp_number_threads > number_threads_received)
    {
        WARN(L"Can not compute with the optimal number of threads (%zu). Number of threads reduce to %zu. Need more data to compute or a larger neural network!",
                                 tmp_number_threads,
                                 number_threads_received);

        tmp_number_threads = number_threads_received;
    }
    
    size_t tmp_batch_size_allocate(number_threads_received),
                      tmp_number_threads_allocate(number_threads_received);
    
    this->Allouable__Batch_Size(number_threads_received,
                                             tmp_number_threads,
                                             tmp_batch_size_allocate,
                                             tmp_number_threads_allocate);

    if(this->Reallocate__Thread(tmp_number_threads_allocate) == false)
    {
        ERR(L"From \"Reallocate__Thread\".",);

        return false;
    }

    this->number_threads = tmp_number_threads_allocate;
    this->cache_number_threads = number_threads_received;
    
    // Asign the new fixed pool size
    this->limit_device_runtime_pending_launch_count = tmp_number_threads_allocate * tmp_maximum_grids_launch_per_example + 1u;

    // number of threads <= batch size.
    if(this->update_mem_batch_size(tmp_number_threads_allocate) == false)
    {
        ERR(L"From \"update_mem_batch_size\"",);

        return false;
    }
    
    INFO(L"Number of threads desired: %zu", number_threads_received);
    INFO(L"Number of threads optimal: %zu", tmp_number_threads_allocate);
    INFO(L"Batch size: %zu", this->batch_size);
    INFO(L"Minimum number of threads required, per example: %zu", tmp_minimum_threads_per_example);
    INFO(L"Maximum grid launch required, per example: %zu", tmp_maximum_grids_launch_per_example);
    INFO(L"Limit device runtime pending launch count (fixed pool size): %zu", this->limit_device_runtime_pending_launch_count);
    INFO(L"Maximum allowable memory: %zu bytes", this->maximum_allowable_memory_bytes);
    INFO(L"Total size neural network: %zu bytes", this->Get__Sizeof());
#else
    kernel__cuModel__Update_Threads_Size <<< 1, 1 >>> (number_threads_received, this);
    
    CUDA__Check_Error();
#endif

    return true;
}

__device__ bool cuModel::Allouable__Batch_Size(size_t const batch_size,
                                                                                    size_t const maximum_threads_received,
                                                                                    size_t &ref_batch_size_allouable_received,
                                                                                    size_t &ref_number_threads_allouable_received)
{
    //this->Update_Available_Memory();

    if(this->number_threads <= 1_UZ)
    {
        ERR(L"Can not allocate batch. Calculate the required threads before running this function!",);

        ref_number_threads_allouable_received = 0_UZ;
        ref_batch_size_allouable_received = 0_UZ;

        return false;
    }
    
    // Size of a thread.
    size_t const  tmp_size_thread(this->Get__Threads_Sizeof(1_UZ)),
    // Size of a batch.
                       tmp_size_batch(this->Get__Batch_Sizeof(1_UZ)),
    // Size of a neural network with no batch.
                       tmp_size_neural_network(this->Get__Sizeof(1_UZ, 1_UZ) - (tmp_size_thread + tmp_size_batch)),
    // Available memory substraction size of the neural network without batch.
                       tmp_available_memory_mbs(this->maximum_allowable_memory_bytes - tmp_size_neural_network);
    
    INFO(L"Maximum allowable memory: %zu bytes", this->maximum_allowable_memory_bytes);
    INFO(L"Size neural network: %zu bytes", tmp_size_neural_network);
    INFO(L"Size for one thread: %zu bytes", tmp_size_thread);
    INFO(L"Size for a batch of size one: %zu bytes", tmp_size_batch);
    INFO(L"Total size neural network: %zu bytes", this->Get__Sizeof());

    // If can not allocate at least one thread, return false.
    if(static_cast<size_t>(tmp_available_memory_mbs / (tmp_size_thread + tmp_size_batch)) == 0_UZ)
    {
        ERR(L"Can not allocate threads. More memory need to be available!",);

        ref_number_threads_allouable_received = 0_UZ;
        ref_batch_size_allouable_received = 0_UZ;

        return false;
    }

    size_t tmp_batch_size_allocate(batch_size),
              tmp_threads_allocate(1);

    // Do... while allocatables threads meet the maximum threads allocatables.
    do
    {
        // Maximum batch size equal available memory minus allocates threads, then divide by one batch size.
        size_t const tmp_maximum_batch_size_allocatable(static_cast<size_t>((tmp_available_memory_mbs - tmp_threads_allocate * tmp_size_thread) / tmp_size_batch));

        // If threads allocates is greater than batch size.
        if(tmp_threads_allocate > tmp_maximum_batch_size_allocatable)
        {
            WARN(L"Can not allocate the optimal number of threads (%zu). Number of threads reduce to %zu. More memory need to be available!",
                                     tmp_threads_allocate,
                                     tmp_threads_allocate - 1_UZ);

            // Batch size equal available memory minus past allocates threads, then divide by one batch size.
            tmp_batch_size_allocate = static_cast<size_t>((tmp_available_memory_mbs - (tmp_threads_allocate - 1_UZ) * tmp_size_thread) / tmp_size_batch);

            break;
        }
        // If batch size is greater than maximum batch size allocatables.
        else if(tmp_batch_size_allocate > tmp_maximum_batch_size_allocatable)
        {
            WARN(L"Can not allocate the optimal batch size (%zu). Batch size reduce to %zu. More memory need to be available!",
                                     tmp_batch_size_allocate,
                                     tmp_maximum_batch_size_allocatable);

            // Batch size equal maximum batch size allocatables.
            tmp_batch_size_allocate = tmp_maximum_batch_size_allocatable;

            break;
        }
    } while(tmp_threads_allocate++ < maximum_threads_received);
    
    ref_number_threads_allouable_received = tmp_threads_allocate - 1_UZ;
    ref_batch_size_allouable_received = tmp_batch_size_allocate;

    return true;
}

__global__ void kernel__cuModel__Update_Batch_Size(size_t const batch_size, class cuModel *const ptr_cuModel_received)
{
    if(ptr_cuModel_received->update_mem_batch_size(batch_size) == false)
    {
        ERR(L"From \"update_mem_batch_size\"",);
    }
}

__host__ __device__ bool cuModel::update_mem_batch_size(size_t batch_size)
{
#ifdef __CUDA_ARCH__
    if(batch_size <= this->cache_batch_size) { return true; }
    else if(this->number_threads <= 1u) { return false; }
    
    size_t tmp_batch_size_allocate(batch_size),
                      tmp_number_threads_allocate(batch_size);
    
    this->Allouable__Batch_Size(batch_size,
                                             this->number_threads,
                                             tmp_batch_size_allocate,
                                             tmp_number_threads_allocate);

    // reallocate batch size with the new batch size meet.
    if(this->Reallocate__Batch(tmp_batch_size_allocate) == false)
    {
        ERR(L"From \"Reallocate__Batch\".",);

        return false;
    }

    this->batch_size = tmp_batch_size_allocate;
    this->cache_batch_size = batch_size;

    INFO(L"Batch size: %u", this->batch_size);
#else
    kernel__cuModel__Update_Batch_Size <<< 1, 1 >>> (batch_size, this);
    
    CUDA__Check_Error();
#endif

    return true;
}

[[deprecated("Not properly implemented.")]] __device__ void cuModel::Initialize_Candidate_Weights(size_t const first_connection_received,
                                                                                            size_t const last_connection_received,
                                                                                            float const scale_factor_received)
{
    /*
    size_t tmp_index_bias_weight(static_cast<size_t>(first_connection_received + (this->ptr_array_layers->ptr_last_neuron_unit - this->ptr_array_layers->ptr_array_neuron_units) - 1));
    
    var tmp_prev_step(0_r);

    if(this->type_optimizer_function == DL::OPTIMIZER::IRPROP_PLUS) { tmp_prev_step = this->rprop_delta_zero; }
    else { tmp_prev_step = 0_r; }
        
    for(size_t i(first_connection_received); i != last_connection_received; ++i)
    {
        this->ptr_array_parameters[i] = static_cast<var>(curand_uniform(&this->ptr_array_cuRAND_State_MTGP32_weighted[0])) * (scale_factor_received - -scale_factor_received) + -scale_factor_received;

        if(i != tmp_index_bias_weight) { this->ptr_array_parameters[i] = abs(this->ptr_array_parameters[i]); }

        this->ptr_array_derivatives_parameters[i] = 0_r;
        this->ptr_array_previous_steps[i] = tmp_prev_step;
        this->ptr_array_previous_derivatives_parameters[i] = 0_r;
    }
    */
}

template<typename T>
__global__ void kernel__cuModel__Randomize_Weights(T const minimum_weight_received,
                                                                                               T const maximum_weight_received,
                                                                                               T *const ptr_array_weights_received,
                                                                                               struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{ ptr_array_weights_received[blockIdx.x * blockDim.x + threadIdx.x] = static_cast<T>(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)) * (maximum_weight_received - minimum_weight_received) + minimum_weight_received; }

template<typename T>
__global__ void kernel__cuModel__Randomize_Weights(size_t const size_received,
                                                                                               T const minimum_weight_received,
                                                                                               T const maximum_weight_received,
                                                                                               T *const ptr_array_weights_received,
                                                                                               struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const tmp_curand_uniform(static_cast<T>(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)));

    if(tmp_thread_global_index < size_received)
    { ptr_array_weights_received[tmp_thread_global_index] = tmp_curand_uniform * (maximum_weight_received - minimum_weight_received) + minimum_weight_received; }
}

template<typename T>
__global__ void kernel_while__cuModel__Randomize_Weights(size_t const size_received,
                                                                                                        T const minimum_weight_received,
                                                                                                        T const maximum_weight_received,
                                                                                                        T *const ptr_array_weights_received,
                                                                                                        struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        ptr_array_weights_received[tmp_thread_global_index] = static_cast<T>(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)) * (maximum_weight_received - minimum_weight_received) + minimum_weight_received;

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__global__ void kernel__cuModel__Launch_Randomize_Weights(var const minimum_weight_received,
                                                                                                           var const maximum_weight_received,
                                                                                                           class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Launch_Randomize_Weights(minimum_weight_received, maximum_weight_received); }

__host__ void cuModel::Launch_Randomize_Weights(var const minimum_weight_received, var const maximum_weight_received)
{
#ifdef __CUDA_ARCH__
    LAUNCH_KERNEL_1D(cuModel__Randomize_Weights,
                                        this->ptr_array_dim3_grid[8],
                                        this->ptr_array_dim3_block[8],
                                        0_UZ,
                                        this->total_weights,
                                        minimum_weight_received,
                                        maximum_weight_received,
                                        this->ptr_array_parameters,
                                        this->ptr_array_cuRAND_State_MTGP32_weighted)

    CUDA__Check_Error();
#else
    kernel__cuModel__Launch_Randomize_Weights <<< 1, 1 >>> (minimum_weight_received,
                                                                                                                 maximum_weight_received,
                                                                                                                 this);
        
    CUDA__Check_Error();
#endif
}

__global__ void kernel__cuModel__Get__Batch_Sizeof(size_t *const size_t_received,
                                                                                          size_t const batch_size,
                                                                                          class cuModel const *const ptr_cuModel_received)
{ *size_t_received = ptr_cuModel_received->Get__Batch_Sizeof(batch_size); }

__host__ __device__ size_t cuModel::Get__Batch_Sizeof(size_t batch_size) const
{
#ifdef __CUDA_ARCH__
    size_t tmp_total_size_t(0_UZ);

    if(batch_size == 0u) { batch_size = this->batch_size; }
    
    // Neurons.
    if(this->ptr_array_neuron_units_summations != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_neuron_units_values != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_normalized_batch_units_values_hats != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_normalized_batch_units_values_normalizes != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_normalized_batch_units_means != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_normalized_batch_units_variances != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_neuron_units_transposed_mean != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_neuron_units_transposed_variance != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_normalized_batch_units_derivatives_means != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_normalized_batch_units_derivatives_variances != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_neuron_units_errors != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->total_neuron_units_allocated * sizeof(var); }
    
    if(this->ptr_array_2D_neurons_reduce_summation != nullptr && *this->ptr_array_2D_neurons_reduce_summation != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->neurons_total_reduce_summation_size * sizeof(var); }
    if(this->ptr_array_2D_neurons_reduce_error != nullptr && *this->ptr_array_2D_neurons_reduce_error != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->neurons_total_reduce_error_size * sizeof(var); }
    if(this->ptr_array_2D_neurons_reduce_batch_mean != nullptr && *this->ptr_array_2D_neurons_reduce_batch_mean != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->neurons_total_reduce_batch_size * sizeof(var); }
    if(this->ptr_array_2D_neurons_reduce_batch_variance != nullptr && *this->ptr_array_2D_neurons_reduce_batch_variance != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->neurons_total_reduce_batch_size * sizeof(var); }
    if(this->ptr_array_2D_neurons_reduce_norms != nullptr && *this->ptr_array_2D_neurons_reduce_norms != nullptr) { tmp_total_size_t += batch_size * this->seq_w * this->neurons_total_reduce_norms_size * sizeof(var); }

    return(tmp_total_size_t);
#else
    size_t tmp_size_t(0),
              *tmp_ptr_device_size_t(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_size_t, sizeof(size_t)));

    kernel__cuModel__Get__Batch_Sizeof <<< 1, 1 >>> (tmp_ptr_device_size_t,
                                                                                                batch_size,
                                                                                                this);
    
    CUDA__Safe_Call(cudaMemcpy(&tmp_size_t,
                                                    tmp_ptr_device_size_t,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_size_t));

    CUDA__Check_Error();

    return(tmp_size_t);
#endif
}

__global__ void kernel__cuModel__Get__Threads_Sizeof(size_t *const size_t_received,
                                                                                             size_t const number_threads_received,
                                                                                             class cuModel const *const ptr_cuModel_received)
{ *size_t_received = ptr_cuModel_received->Get__Threads_Sizeof(number_threads_received); }

__host__ __device__ size_t cuModel::Get__Threads_Sizeof(size_t number_threads_received) const
{
#ifdef __CUDA_ARCH__
    size_t tmp_total_size_t(0_UZ);

    if(number_threads_received == 0u) { number_threads_received = this->number_threads; }
    
    // Cost.
    if(this->ptr_array_number_loss != nullptr) { tmp_total_size_t += number_threads_received * sizeof(size_t); }
    if(this->ptr_array_number_bit_fail != nullptr) { tmp_total_size_t += number_threads_received * sizeof(size_t); }
    if(this->ptr_array_loss_values != nullptr) { tmp_total_size_t += number_threads_received * sizeof(var); }
    if(this->ptr_array_accuracy_values[0] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(var); }
    if(this->ptr_array_accuracy_values[1] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(var); }
    if(this->ptr_array_accuracy_values[2] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(var); }
    if(this->ptr_array_accuracy_values[3] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(var); }
    if(this->ptr_array_accuracy_values[4] != nullptr) { tmp_total_size_t += number_threads_received * sizeof(var); }

    // Parameters.
    if(this->ptr_array_derivatives_parameters != nullptr) { tmp_total_size_t += number_threads_received * this->total_parameters_allocated * sizeof(var); }

    return(tmp_total_size_t);
#else
    size_t tmp_size_t(0),
              *tmp_ptr_device_size_t(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_size_t, sizeof(size_t)));

    kernel__cuModel__Get__Threads_Sizeof <<< 1, 1 >>> (tmp_ptr_device_size_t,
                                                                                                   number_threads_received,
                                                                                                   this);
    
    CUDA__Safe_Call(cudaMemcpy(&tmp_size_t,
                                                    tmp_ptr_device_size_t,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_size_t));

    CUDA__Check_Error();

    return(tmp_size_t);
#endif
}

__global__ void kernel__cuModel__Get__Maximum_Allowable_Memory(size_t *const size_t_received, class cuModel const *const ptr_cuModel_received)
{ *size_t_received = ptr_cuModel_received->Get__Maximum_Allowable_Memory(); }

__host__ __device__ size_t cuModel::Get__Maximum_Allowable_Memory(void) const
{
#ifdef __CUDA_ARCH__
    return(this->maximum_allowable_memory_bytes);
#else
    size_t tmp_size_t(0),
              *tmp_ptr_device_size_t(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_size_t, sizeof(size_t)));

    kernel__cuModel__Get__Maximum_Allowable_Memory <<< 1, 1 >>> (tmp_ptr_device_size_t, this);
    
    CUDA__Safe_Call(cudaMemcpy(&tmp_size_t,
                                                    tmp_ptr_device_size_t,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_size_t));

    CUDA__Check_Error();

    return(tmp_size_t);
#endif
}

__global__ void kernel__cuModel__Get__Sizeof(size_t *const size_t_received,
                                                                                size_t const number_threads_received,
                                                                                size_t const batch_size,
                                                                                class cuModel const *const ptr_cuModel_received)
{ *size_t_received = ptr_cuModel_received->Get__Sizeof(number_threads_received, batch_size); }

__host__ __device__ size_t cuModel::Get__Sizeof(size_t number_threads_received, size_t batch_size) const
{
#ifdef __CUDA_ARCH__
    size_t tmp_total_size_t(0);

    tmp_total_size_t += sizeof(class cuModel); // this
    
    tmp_total_size_t += this->Get__Threads_Sizeof(number_threads_received == 0u ? this->number_threads : number_threads_received);
    
    tmp_total_size_t += this->Get__Batch_Sizeof(batch_size == 0u ? this->batch_size : batch_size);

    //tmp_total_size_t += X * sizeof(struct struct_Block_Parameters); // ptr_array_block_parameters
    
    // Dim3.
    if(this->ptr_array_dim3_grid != NULL) { tmp_total_size_t += TOTAL_KERNEL_PARALLEL * sizeof(struct dim3); }
    if(this->ptr_array_dim3_block != NULL) { tmp_total_size_t += TOTAL_KERNEL_PARALLEL * sizeof(struct dim3); }
    
    if(this->ptr_array_dim3_grid_reduce_threads != NULL) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(struct dim3); }
    if(this->ptr_array_dim3_block_reduce_threads != NULL) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(struct dim3); }
    
    if(this->ptr_array_dim3_grid_reduce_threads_DP != NULL) { tmp_total_size_t += this->total_reduce_batch_DP_size * sizeof(struct dim3); }
    if(this->ptr_array_dim3_block_reduce_threads_DP != NULL) { tmp_total_size_t += this->total_reduce_batch_DP_size * sizeof(struct dim3); }
    
    if(this->ptr_array_layers_dim3_grid_neurons != NULL) { tmp_total_size_t += this->total_layers * sizeof(struct dim3); }
    if(this->ptr_array_layers_dim3_block_neurons != NULL) { tmp_total_size_t += this->total_layers * sizeof(struct dim3); }
    
    if(this->ptr_array_layers_dim3_grid_weights != NULL) { tmp_total_size_t += this->total_layers * sizeof(struct dim3); }
    if(this->ptr_array_layers_dim3_block_weights != NULL) { tmp_total_size_t += this->total_layers * sizeof(struct dim3); }
    
    if(this->ptr_array_neuron_units_dim3_grid_connections != NULL) { tmp_total_size_t += this->total_neuron_units * sizeof(struct dim3); }
    if(this->ptr_array_neuron_units_dim3_block_connections != NULL) { tmp_total_size_t += this->total_neuron_units * sizeof(struct dim3); }

    if(this->ptr_array_neuron_units_dim3_grid_reduce_summation != NULL) { tmp_total_size_t += this->neurons_total_reduce_summation_size * sizeof(struct dim3); }
    if(this->ptr_array_neuron_units_dim3_block_reduce_summation != NULL) { tmp_total_size_t += this->neurons_total_reduce_summation_size * sizeof(struct dim3); }
    
    if(this->ptr_array_neuron_units_dim3_grid_reduce_error != NULL) { tmp_total_size_t += this->neurons_total_reduce_error_size * sizeof(struct dim3); }
    if(this->ptr_array_neuron_units_dim3_block_reduce_error != NULL) { tmp_total_size_t += this->neurons_total_reduce_error_size * sizeof(struct dim3); }
    
    if(this->ptr_array_neuron_units_dim3_grid_reduce_batch != NULL) { tmp_total_size_t += this->neurons_total_reduce_batch_size * sizeof(struct dim3); }
    if(this->ptr_array_neuron_units_dim3_block_reduce_batch != NULL) { tmp_total_size_t += this->neurons_total_reduce_batch_size * sizeof(struct dim3); }
    
    if(this->ptr_array_2D_neurons_dim3_grid_reduce_norms != NULL)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(struct dim3*);

        if(this->ptr_array_2D_neurons_dim3_grid_reduce_norms[0] != NULL) { tmp_total_size_t += this->neurons_total_reduce_norms_size * sizeof(struct dim3); }
    }

    if(this->ptr_array_2D_neurons_dim3_block_reduce_norms != NULL)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(struct dim3*);

        if(this->ptr_array_2D_neurons_dim3_block_reduce_norms[0] != NULL) { tmp_total_size_t += this->neurons_total_reduce_norms_size * sizeof(struct dim3); }
    }
    // |END| Dim3. |END|
    
    // Cost reduce.
    if(this->ptr_array_reduce_number_loss != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(size_t); }
    if(this->ptr_array_reduce_bit_fail_values != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(size_t); }
    if(this->ptr_array_reduce_loss_values != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(var); }
    if(this->ptr_array_reduce_accuracy_values[0] != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(var); }
    if(this->ptr_array_reduce_accuracy_values[1] != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(var); }
    if(this->ptr_array_reduce_accuracy_values[2] != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(var); }
    if(this->ptr_array_reduce_accuracy_values[3] != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(var); }
    if(this->ptr_array_reduce_accuracy_values[4] != nullptr) { tmp_total_size_t += this->total_reduce_batch_size * sizeof(var); }

    // Parameters.
    if(this->ptr_array_ptr_connections != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(void*); }

    if(this->ptr_array_transposed_weights != nullptr) { tmp_total_size_t += this->total_weights_allocated * sizeof(var); }
    if(this->ptr_array_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(var); }
    if(this->ptr_array_mask_regularized_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(var); }

    //    Optimizer iRPROP.
    if(this->ptr_array_previous_steps != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(var); }
    if(this->ptr_array_previous_delta_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(var); }
    if(this->ptr_array_previous_derivatives_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(var); }
    //    |END| Optimizer iRPROP. |END|

    //    Optimizer Adam.
    if(this->ptr_array_previous_biased_first_moment != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(var); }
    if(this->ptr_array_previous_biased_second_moment != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(var); }
    //    |END| Optimizer Adam. |END|

    //    Optimizer AMSGrad.
    if(this->ptr_array_previous_biased_second_moment_hat != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(var); }
    //    |END| Optimizer AMSGrad. |END|
    // |END| Parameters. |END|
    
    // Dropout variable.
    if(this->ptr_array_af_units_mask_dropout_bernoulli != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(bool); }
    if(this->ptr_array_cell_units_mask_dropout_zoneout != nullptr) { tmp_total_size_t += 2_UZ * this->total_cell_units_allocated * sizeof(bool); }
    if(this->ptr_array_mask_dropout_parameters != nullptr) { tmp_total_size_t += this->total_parameters_allocated * sizeof(var); }
    // |END| Dropout variable. |END|
    
    // TODO: Create into cuDevicesProp a function returning sizeof called Get__Sizeof().
    if(this->_ptr_Class_Device_Information_Array != nullptr)
    {
        tmp_total_size_t += sizeof(class cuDevicesProp);

        if(this->_ptr_Class_Device_Information_Array->Get__Number_CUDA_Devices() != 0u)
        {
            tmp_total_size_t += sizeof(class cuDeviceProp); // _ptr_Class_Device_Information_sum
            tmp_total_size_t += sizeof(class cuDeviceProp); // _ptr_Class_Device_Information_higher
            tmp_total_size_t += sizeof(class cuDeviceProp); // _ptr_Class_Device_Information_lower
            tmp_total_size_t += this->_ptr_Class_Device_Information_Array->Get__Number_CUDA_Devices() * sizeof(class cuDeviceProp); // _ptr_array_Class_Device_Information
        }
    }

    // Layers.
    if(this->ptr_array_number_neurons_by_layer != nullptr) { tmp_total_size_t += this->total_layers * sizeof(size_t); }

    if(this->ptr_array_layers != nullptr)
    {
        tmp_total_size_t += this->total_layers * sizeof(struct cuLayer);

        if(this->ptr_array_layers->ptr_array_neuron_units != nullptr)
        { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(struct cuNeuron); }
    }

    if(this->ptr_array_layers_Class_Storage_Dim3_Batch != nullptr) { tmp_total_size_t += this->total_layers * sizeof(class cuDims); }
    // |END| Layers. |END|

    // Neurons.
    if(this->ptr_array_neuron_units_first_forward_connection_index != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_last_forward_connection_index != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_number_forward_connections != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_reduce_summation_size != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_reduce_batch_size != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuron_units_reduce_norms_size != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }
    if(this->ptr_array_neuroyed_number_neurons_in_layer != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t); }

    if(this->ptr_array_normalized_batch_units_r_corrections != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_normalized_batch_units_d_corrections != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_normalized_batch_units_means_averages != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(var); }
    if(this->ptr_array_normalized_batch_units_variances_averages != nullptr) { tmp_total_size_t += this->total_neuron_units_allocated * sizeof(var); }

    if(this->ptr_array_2D_neurons_reduce_summation != nullptr)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(var*);
            
        if(*this->ptr_array_2D_neurons_reduce_summation != nullptr)
        { tmp_total_size_t += this->neurons_total_reduce_summation_size * sizeof(var); }
    }
    
    if(this->ptr_array_2D_neurons_reduce_error != nullptr)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(var*);
            
        if(*this->ptr_array_2D_neurons_reduce_error != nullptr)
        { tmp_total_size_t += this->neurons_total_reduce_error_size * sizeof(var); }
    }
    
    if(this->ptr_array_2D_neurons_reduce_batch_mean != nullptr)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(var*);
            
        if(*this->ptr_array_2D_neurons_reduce_batch_mean != nullptr)
        { tmp_total_size_t += this->neurons_total_reduce_batch_size * sizeof(var); }
    }
    
    if(this->ptr_array_2D_neurons_reduce_batch_variance != nullptr)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(var*);
            
        if(*this->ptr_array_2D_neurons_reduce_batch_variance != nullptr)
        { tmp_total_size_t += this->neurons_total_reduce_batch_size * sizeof(var); }
    }
    
    if(this->ptr_array_2D_neurons_reduce_norms != nullptr)
    {
        tmp_total_size_t += this->total_neuron_units_allocated * sizeof(var*);
            
        if(*this->ptr_array_2D_neurons_reduce_norms != nullptr)
        { tmp_total_size_t += this->neurons_total_reduce_norms_size * sizeof(var); }
    }
    // |END| Neurons. |END|
        
    // cuRAND.
    if(this->ptr_array_cuRAND_State_MTGP32_weighted != nullptr)
    {
        tmp_total_size_t += this->number_cuRAND_State_MTGP32_weighted * sizeof(struct curandStateMtgp32);
        tmp_total_size_t += this->number_cuRAND_State_MTGP32_weighted * sizeof(mtgp32_kernel_params_t);
    }
    
    if(this->ptr_array_cuRAND_State_MTGP32_neuroyed != nullptr)
    {
        tmp_total_size_t += this->number_cuRAND_State_MTGP32_neuroyed * sizeof(struct curandStateMtgp32);
        tmp_total_size_t += this->number_cuRAND_State_MTGP32_neuroyed * sizeof(mtgp32_kernel_params_t);
    }
    // |END| cuRAND. |END|

    return(tmp_total_size_t);
#else
    size_t tmp_size_t(0),
              *tmp_ptr_device_size_t(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_size_t, sizeof(size_t)));

    kernel__cuModel__Get__Sizeof <<< 1, 1 >>> (tmp_ptr_device_size_t,
                                                                                      number_threads_received,
                                                                                      batch_size,
                                                                                      this);
    
    CUDA__Safe_Call(cudaMemcpy(&tmp_size_t,
                                                    tmp_ptr_device_size_t,
                                                    sizeof(size_t),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_size_t));

    CUDA__Check_Error();

    return(tmp_size_t);
#endif
}

[[deprecated("Not properly implemented.")]] __device__ void cuModel::Printf_Parameters(bool const full_description_received)
{
    INFO(L"Input layer : %u neuson(s), 1 bias.", this->n_inp);
    INFO(L"Output layer : %u neuron(s).", this->n_out);
}
