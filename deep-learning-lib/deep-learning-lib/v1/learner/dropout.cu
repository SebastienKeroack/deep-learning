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
#include "deep-learning-lib/ops/mask.cuh"

#include <curand_kernel.h>

// Public function
__device__ bool cuModel::Set__Probability_Retained_Unit(size_t const index_layer_received,
                                                                                                real const retention_probability_received,
                                                                                                bool const scale_weights_received)
{
    if(index_layer_received >= this->total_layers)
    {
        ERR(L"Layer received (%d) overflow the number of layers (%d) in the neural network.",
                                index_layer_received,
                                this->total_layers);

        return false;
    }
    else if(this->ptr_array_layers == nullptr)
    {
        ERR(L"The array of layers is a nullptr.",);

        return false;
    }
        
    return(this->Set__Probability_Retained_Unit(this->ptr_array_layers + index_layer_received,
                                                                    retention_probability_received,
                                                                    scale_weights_received));
}

__device__ void cuModel::Scale_Weight__Dropout(var const scale_factor_received, struct cuLayer const *const layer_it)
{
    switch(layer_it->type_layer)
    {
        case DL::LAYER::FULLY_CONNECTED: this->Scale_Weight__FC__Forward__Dropout(scale_factor_received, layer_it); break;
    }
}

__device__ void cuModel::Scale_Weight__FC__Forward__Dropout(var const scale_factor_received, struct cuLayer const *const layer_it)
{
    struct cuNeuron const *const tmp_ptr_neuron_unit_it(layer_it->ptr_array_neuron_units);
    
    var *tmp_ptr_array_parameters(this->ptr_array_parameters + *tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index);
    var const *const tmp_ptr_array_parameters_end(tmp_ptr_array_parameters + *layer_it->ptr_number_neurons * *tmp_ptr_neuron_unit_it->ptr_number_forward_connections);
    
    for(; tmp_ptr_array_parameters != tmp_ptr_array_parameters_end; ++tmp_ptr_array_parameters)
    { *tmp_ptr_array_parameters *= scale_factor_received; }
}

// Private function.
__device__ bool cuModel::Set__Probability_Retained_Unit(struct cuLayer *ptr_layer_received,
                                                                                                        real const retention_probability_received,
                                                                                                        bool const scale_weights_received)
{
    struct cuLayer const *last_layer;
    struct cuLayer *layer_it;
    
    if(ptr_layer_received == nullptr)
    {
        ERR(L"Layer received is a nullptr.",);

        return false;
    }
    else if(retention_probability_received < 0.0f)
    {
        ERR(L"probability for retained a unit (%f) in the layer, underflow the requirement minimum of 0.0.",
                        retention_probability_received);

        return false;
    }
    else if(retention_probability_received > 1.0f)
    {
        ERR(L"probability for retained a unit (%f) in the layer, overflow the requirement maximum of 1.0.",
                        retention_probability_received);

        return false;
    }
        
    if(ptr_layer_received->dropout_values[0] != retention_probability_received)
    {
        if(scale_weights_received && ptr_layer_received != this->ptr_array_layers) { this->Scale_Weight__Dropout(ptr_layer_received->dropout_values[0] / retention_probability_received, ptr_layer_received); }

        ptr_layer_received->dropout_values[0] = retention_probability_received;

        if(retention_probability_received != 1.0f)
        {
            this->use_Dropout = true;

            if(this->Allocate__Neuron__Mask_Dropout_Bernoulli() == false)
            {
                ERR(L"Can not allocate neurons mask dropout!",);
                
                return false;
            }
        }
        else // Check if we use dropout
        {
            bool tmp_use_Dropout(false);
            
            // Loop through each layer to do a check if a layer use dropout.
            for(last_layer = this->ptr_last_layer,
                layer_it = this->ptr_array_layers; layer_it != last_layer; ++layer_it)
            {
                if(layer_it->dropout_values[0] != 1_r)
                {
                    tmp_use_Dropout = true;

                    break;
                }
            }
            
            this->use_Dropout = tmp_use_Dropout;
            // |END| Loop through each layer to do a check if a layer use dropout. |END|

            if(tmp_use_Dropout == false)
            {
                this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
            }
        }
    }

    return true;
}

__device__ void cuModel::Reset__Parameter__AF_Units__Mask_Dropout(bool *ptr_array_neuron_units_mask_dropout_received)
{
    struct cuNeuron *tmp_ptr_neuron_unit_it(this->ptr_array_layers->ptr_array_neuron_units);
    struct cuNeuron const *const tmp_ptr_last_neuron_unit(tmp_ptr_neuron_unit_it + this->total_neuron_units);

    this->ptr_array_af_units_mask_dropout_bernoulli = ptr_array_neuron_units_mask_dropout_received;

    for(; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                         ++ptr_array_neuron_units_mask_dropout_received)
    { tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli = ptr_array_neuron_units_mask_dropout_received; }
}

//__managed__ size_t tmp_count_dropped = 0u;
//__managed__ size_t tmp_count_total = 0u;

template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons(bool *const ptr_array_mask_dropout_received,
                                                                T const probability_retained_unit_received,
                                                                struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
    { ptr_array_mask_dropout_received[tmp_thread_global_index] = true; }
    else // Dropout neuron
    { ptr_array_mask_dropout_received[tmp_thread_global_index] = false; }
}
    
template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons(size_t const size_received,
                                                                bool *const ptr_array_mask_dropout_received,
                                                                T const probability_retained_unit_received,
                                                                struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    float const tmp_curand_uniform(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x));

    if(tmp_thread_global_index < size_received)
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, tmp_curand_uniform))
        { ptr_array_mask_dropout_received[tmp_thread_global_index] = true; }
        else // Dropout neuron
        { ptr_array_mask_dropout_received[tmp_thread_global_index] = false; }
    }
}

template<typename T>
__global__ void kernel_while__Dropout_Bernoulli__Neurons(size_t const size_received,
                                                                        bool *const ptr_array_mask_dropout_received,
                                                                        T const probability_retained_unit_received,
                                                                        struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
        { ptr_array_mask_dropout_received[tmp_thread_global_index] = true; }
        else // Dropout neuron
        { ptr_array_mask_dropout_received[tmp_thread_global_index] = false; }
            
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::Dropout(void)
{
    bool const tmp_use_parameters_dropout(false);

    /* sync code:
        0: Synchronized,
        1: Critical kernel launch,
        2: Optinal kernel launch. */
    size_t tmp_sync_code(0u),
                      tmp_neuron_index;

    struct cuLayer const *const last_layer(this->ptr_last_layer);
    struct cuLayer *tmp_ptr_previous_layer_it(this->ptr_array_layers),
                                            *layer_it(tmp_ptr_previous_layer_it);

    struct cuNeuron const *const tmp_ptr_neuron_unit_it(layer_it->ptr_array_neuron_units);
    
    bool *tmp_ptr_array_mask_dropout(tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli);
    bool const *const tmp_ptr_array_mask_dropout_end(tmp_ptr_array_mask_dropout + *layer_it->ptr_number_neurons - 1u); // Subtract bias.

    // Input layer.
    if(USE_PARALLEL && *layer_it->ptr_number_neurons - 1u >= warpSize)
    {
        // Critical synchronization required. To see the previous neurons flag.
        tmp_sync_code = 1u;
        
        LAUNCH_KERNEL_POINTER_1D(Dropout_Bernoulli__Neurons<var>,
                                                          layer_it->ptr_dim3_grid_neurons_cuRAND,
                                                          layer_it->ptr_dim3_block_neurons_cuRAND,
                                                          0_UZ,
                                                          *layer_it->ptr_number_neurons - 1, // Subtract bias.
                                                          tmp_ptr_array_mask_dropout,
                                                          layer_it->dropout_values[0],
                                                          this->ptr_array_cuRAND_State_MTGP32_neuroyed)
    }
    else
    {
        for(tmp_neuron_index = 0_UZ; tmp_ptr_array_mask_dropout != tmp_ptr_array_mask_dropout_end; ++tmp_ptr_array_mask_dropout,
                                                                                                                                                   ++tmp_neuron_index)
        {
            if(cuRAND_Bernoulli(layer_it->dropout_values[0], curand_uniform(this->ptr_array_cuRAND_State_MTGP32_neuroyed)))
            { *tmp_ptr_array_mask_dropout = true; }
            else // Dropout neuron
            { *tmp_ptr_array_mask_dropout = false; }
        }
    }
    // |END| Input layer. |END|
    
    if(this->use_Batch_Renormalization && tmp_use_parameters_dropout)
    {
        for(++layer_it; layer_it != last_layer; ++layer_it,
                                                                                                    ++tmp_ptr_previous_layer_it)
        {
            if(layer_it->use_Batch_Renormalization)
            {
                switch(tmp_ptr_previous_layer_it->type_layer)
                {
                    case DL::LAYER::FULLY_CONNECTED:
                        this->Dropout__FC_to__Batch_Normalization(tmp_sync_code,
                                                                                                layer_it,
                                                                                                tmp_ptr_previous_layer_it);
                            break;
                }
            }
            else
            {
                switch(tmp_ptr_previous_layer_it->type_layer)
                {
                    case DL::LAYER::FULLY_CONNECTED:
                        this->Dropout__FC_to(tmp_sync_code,
                                                              tmp_use_parameters_dropout,
                                                              layer_it,
                                                              tmp_ptr_previous_layer_it);
                            break;
                }
            }
        }
    }
    else
    {
        for(++layer_it; layer_it != last_layer; ++layer_it,
                                                                                                    ++tmp_ptr_previous_layer_it)
        {
            switch(tmp_ptr_previous_layer_it->type_layer)
            {
                case DL::LAYER::FULLY_CONNECTED:
                    this->Dropout__FC_to(tmp_sync_code,
                                                          tmp_use_parameters_dropout,
                                                          layer_it,
                                                          tmp_ptr_previous_layer_it);
                        break;
            }
        }
    }

    // If the state of the synchronized is not at zero. We synchronize.
    if(tmp_sync_code != 0u) { CUDA__Check_Error(); }
}

__device__ void cuModel::Dropout__FC_to(size_t &ref_sync_code_received,
                                                                                bool const use_parameters_dropout_received,
                                                                                struct cuLayer *const layer_it,
                                                                                struct cuLayer const *const ptr_previous_layer_it_received)
{
    switch(layer_it->type_layer)
    {
        case DL::LAYER::FULLY_CONNECTED:
            this->Dropout_Bernoulli__FC_to_FC(ref_sync_code_received,
                                                            use_parameters_dropout_received,
                                                            layer_it,
                                                            ptr_previous_layer_it_received);
                break;
    }
}

__device__ void cuModel::Dropout__FC_to__Batch_Normalization(size_t &ref_sync_code_received,
                                                                                                                 struct cuLayer *const layer_it,
                                                                                                                 struct cuLayer const *const ptr_previous_layer_it_received)
{
    switch(layer_it->type_layer)
    {
        case DL::LAYER::FULLY_CONNECTED:
            this->Dropout_Bernoulli__FC_to_FC__Batch_Renormalization(ref_sync_code_received,
                                                                                              layer_it,
                                                                                              ptr_previous_layer_it_received);
                break;
    }
}

template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons(size_t const number_connections_received,
                                                                bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                bool *const ptr_array_mask_dropout_received,
                                                                T *const ptr_array_mask_dropout_parameters_received,
                                                                T const probability_retained_unit_received,
                                                                struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
    {
        ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
        Flag_1D<T>(number_connections_received,
                            ptr_array_previous_layer_mask_dropout_received,
                            ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                            ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                            ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
    }
    else // Dropout neuron
    {
        ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
        Zero_1D<T>(number_connections_received,
                            ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                            ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                            ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
    }
}
    
template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons(size_t const size_received,
                                                                size_t const number_connections_received,
                                                                bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                bool *const ptr_array_mask_dropout_received,
                                                                T *const ptr_array_mask_dropout_parameters_received,
                                                                T const probability_retained_unit_received,
                                                                struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    float const tmp_curand_uniform(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x));

    if(tmp_thread_global_index < size_received)
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, tmp_curand_uniform))
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
            Flag_1D<T>(number_connections_received,
                            ptr_array_previous_layer_mask_dropout_received,
                            ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                            ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                            ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
        }
        else // Dropout neuron
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
            Zero_1D<T>(number_connections_received,
                            ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                            ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                            ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
        }
    }
}

template<typename T>
__global__ void kernel_while__Dropout_Bernoulli__Neurons(size_t const size_received,
                                                                        size_t const number_connections_received,
                                                                        bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                        bool *const ptr_array_mask_dropout_received,
                                                                        T *const ptr_array_mask_dropout_parameters_received,
                                                                        T const probability_retained_unit_received,
                                                                        struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                        struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                        struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
            Flag_1D<T>(number_connections_received,
                              ptr_array_previous_layer_mask_dropout_received,
                              ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                              ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                              ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
        }
        else // Dropout neuron
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
            Zero_1D<T>(number_connections_received,
                              ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                              ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                              ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
        }
            
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::Dropout_Bernoulli__FC_to_FC(size_t &ref_sync_code_received,
                                                                                           bool const use_parameters_dropout_received,
                                                                                           struct cuLayer *const layer_it,
                                                                                           struct cuLayer const *const ptr_previous_layer_it_received)
{
    struct cuNeuron const *const tmp_ptr_neuron_unit_it(layer_it->ptr_array_neuron_units);

    bool *tmp_ptr_array_mask_dropout(tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli);
    bool const *const tmp_ptr_array_mask_dropout_end(tmp_ptr_array_mask_dropout + *layer_it->ptr_number_neurons - 1u), // Subtract bias.
                    *tmp_ptr_array_previous_layer_mask_dropout(ptr_previous_layer_it_received->ptr_array_neuron_units->ptr_mask_dropout_bernoulli);

    size_t const tmp_number_connections(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections - 1u); // Subtract bias.
    size_t tmp_neuron_index;

    var *tmp_ptr_array_mask_dropout_parameters(this->ptr_array_mask_dropout_parameters + *tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index);
    
    struct dim3 const *tmp_ptr_array_dim3_grid_connections(tmp_ptr_neuron_unit_it->ptr_dim3_grid_connections),
                              *tmp_ptr_array_dim3_block_connections(tmp_ptr_neuron_unit_it->ptr_dim3_block_connections);

    if(USE_PARALLEL && *layer_it->ptr_number_neurons - 1u >= warpSize)
    {
        // Critical synchronization required. To see the previous neurons flag.
        ref_sync_code_received = 1u;

        if(use_parameters_dropout_received)
        {
            LAUNCH_KERNEL_POINTER_1D(Dropout_Bernoulli__Neurons<var>,
                                                              layer_it->ptr_dim3_grid_neurons_cuRAND,
                                                              layer_it->ptr_dim3_block_neurons_cuRAND,
                                                              0_UZ,
                                                              *layer_it->ptr_number_neurons - 1, // Subtract bias.
                                                              tmp_number_connections,
                                                              tmp_ptr_array_previous_layer_mask_dropout,
                                                              tmp_ptr_array_mask_dropout,
                                                              tmp_ptr_array_mask_dropout_parameters,
                                                              layer_it->dropout_values[0],
                                                              this->ptr_array_cuRAND_State_MTGP32_neuroyed,
                                                              tmp_ptr_array_dim3_grid_connections,
                                                              tmp_ptr_array_dim3_block_connections)
        }
        else
        {
            LAUNCH_KERNEL_POINTER_1D(Dropout_Bernoulli__Neurons<var>,
                                                              layer_it->ptr_dim3_grid_neurons_cuRAND,
                                                              layer_it->ptr_dim3_block_neurons_cuRAND,
                                                              0_UZ,
                                                              *layer_it->ptr_number_neurons - 1, // Subtract bias.
                                                              tmp_ptr_array_mask_dropout,
                                                              layer_it->dropout_values[0],
                                                              this->ptr_array_cuRAND_State_MTGP32_neuroyed)
        }
    }
    else
    {
        // Need a synchronization to see the previous neurons flag.
        if(ref_sync_code_received == 1u)
        {
            // Set the state at zero (Synchronized).
            ref_sync_code_received = 0u;

            CUDA__Check_Error();
        }

        for(tmp_neuron_index = 0_UZ; tmp_ptr_array_mask_dropout != tmp_ptr_array_mask_dropout_end; ++tmp_ptr_array_mask_dropout,
                                                                                                                                                   ++tmp_neuron_index,
                                                                                                                                                   ++tmp_ptr_array_dim3_grid_connections,
                                                                                                                                                   ++tmp_ptr_array_dim3_block_connections,
                                                                                                                                                   tmp_ptr_array_mask_dropout_parameters += tmp_number_connections + 1u) // Add bias.
        {
            if(cuRAND_Bernoulli(layer_it->dropout_values[0], curand_uniform(this->ptr_array_cuRAND_State_MTGP32_neuroyed)))
            {
                *tmp_ptr_array_mask_dropout = true;
                
                if(use_parameters_dropout_received)
                {
                    Flag_1D<var>(tmp_number_connections,
                                        tmp_ptr_array_previous_layer_mask_dropout,
                                        tmp_ptr_array_mask_dropout_parameters,
                                        tmp_ptr_array_dim3_grid_connections,
                                        tmp_ptr_array_dim3_block_connections);

                    tmp_ptr_array_mask_dropout_parameters[tmp_number_connections] = 1_r; // Bias.
                }
            }
            else // Dropout neuron
            {
                *tmp_ptr_array_mask_dropout = false;
                    
                if(use_parameters_dropout_received)
                {
                    Zero_1D<var>(tmp_number_connections,
                                        tmp_ptr_array_mask_dropout_parameters,
                                        tmp_ptr_array_dim3_grid_connections,
                                        tmp_ptr_array_dim3_block_connections);

                    tmp_ptr_array_mask_dropout_parameters[tmp_number_connections] = 0_r; // Bias.
                }
            }
        }
        
        // If number of connections is bigger or equal to 32. We need a synchronization to see the connections flag.
        if(use_parameters_dropout_received && tmp_number_connections >= warpSize) { ref_sync_code_received = 2u; }
    }

    //INFO(L"tmp_count_dropped: %u / %u, %f%%",
    //                        tmp_count_dropped,
    //                        tmp_count_total,
    //                        static_cast<float>(tmp_count_dropped) / static_cast<float>(tmp_count_total) * 100.0f);
}

template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons__Batch_Renormalization(size_t const number_connections_received,
                                                                                                                    bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                                                                    bool *const ptr_array_mask_dropout_received,
                                                                                                                    T const *const ptr_array_parameters_received,
                                                                                                                    T const *const ptr_array_parameters_scale_received,
                                                                                                                    T const *const ptr_array_parameters_shift_received,
                                                                                                                    T *const ptr_array_original_mask_dropout_parameters_received,
                                                                                                                    T *const ptr_array_mask_dropout_parameters_received,
                                                                                                                    T const probability_retained_unit_received,
                                                                                                                    struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                                                                    struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                    struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
    {
        ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
        Flag_1D<T>(number_connections_received,
                          ptr_array_previous_layer_mask_dropout_received,
                          ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                          ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                          ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
            
        // TODO: Optimize with a real index.
        ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization scale.
        ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization shift.
    }
    else // Dropout neuron
    {
        ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
        Zero_1D<T>(number_connections_received,
                          ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                          ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                          ptr_array_dim3_block_connections_received + tmp_thread_global_index);

        ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
            
        // TODO: Optimize with a real index.
        ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization scale.
        ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization shift.
    }
}
    
template<typename T>
__global__ void kernel__Dropout_Bernoulli__Neurons__Batch_Renormalization(size_t const size_received,
                                                                                                                    size_t const number_connections_received,
                                                                                                                    bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                                                                    bool *const ptr_array_mask_dropout_received,
                                                                                                                    T *const ptr_array_parameters_received,
                                                                                                                    T *const ptr_array_parameters_scale_received,
                                                                                                                    T *const ptr_array_parameters_shift_received,
                                                                                                                    T *const ptr_array_original_mask_dropout_parameters_received,
                                                                                                                    T *const ptr_array_mask_dropout_parameters_received,
                                                                                                                    T const probability_retained_unit_received,
                                                                                                                    struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                                                                    struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                    struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    float const tmp_curand_uniform(curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x));

    if(tmp_thread_global_index < size_received)
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, tmp_curand_uniform))
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
            Flag_1D<T>(number_connections_received,
                              ptr_array_previous_layer_mask_dropout_received,
                              ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                              ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                              ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
            
            // TODO: Optimize with a real index.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization scale.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization shift.
        }
        else // Dropout neuron
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
            Zero_1D<T>(number_connections_received,
                              ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                              ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                              ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
            
            // TODO: Optimize with a real index.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization scale.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization shift.
        }
    }
}

template<typename T>
__global__ void kernel_while__Dropout_Bernoulli__Neurons__Batch_Renormalization(size_t const size_received,
                                                                                                                            size_t const number_connections_received,
                                                                                                                            bool const *const ptr_array_previous_layer_mask_dropout_received,
                                                                                                                            bool *const ptr_array_mask_dropout_received,
                                                                                                                            T *const ptr_array_parameters_received,
                                                                                                                            T *const ptr_array_parameters_scale_received,
                                                                                                                            T *const ptr_array_parameters_shift_received,
                                                                                                                            T *const ptr_array_original_mask_dropout_parameters_received,
                                                                                                                            T *const ptr_array_mask_dropout_parameters_received,
                                                                                                                            T const probability_retained_unit_received,
                                                                                                                            struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                            struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        if(cuRAND_Bernoulli(probability_retained_unit_received, curand_uniform(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x)))
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = true;
            
            Flag_1D<T>(number_connections_received,
                               ptr_array_previous_layer_mask_dropout_received,
                               ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                               ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                               ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(1); // Bias.
            
            // TODO: Optimize with a real index.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization scale.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(1); // Batch normalization shift.
        }
        else // Dropout neuron
        {
            ptr_array_mask_dropout_received[tmp_thread_global_index] = false;
            
            Zero_1D<T>(number_connections_received,
                               ptr_array_mask_dropout_parameters_received + tmp_thread_global_index * (number_connections_received + 1u),
                               ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                               ptr_array_dim3_block_connections_received + tmp_thread_global_index);

            ptr_array_mask_dropout_parameters_received[(number_connections_received + 1u) * tmp_thread_global_index + number_connections_received] = T(0); // Bias.
            
            // TODO: Optimize with a real index.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_scale_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization scale.
            ptr_array_original_mask_dropout_parameters_received[static_cast<size_t>(&ptr_array_parameters_shift_received[tmp_thread_global_index] - ptr_array_parameters_received)] = T(0); // Batch normalization shift.
        }
            
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::Dropout_Bernoulli__FC_to_FC__Batch_Renormalization(size_t &ref_sync_code_received,
                                                                                                                            struct cuLayer *const layer_it,
                                                                                                                            struct cuLayer const *const ptr_previous_layer_it_received)
{
    struct cuNeuron const *const tmp_ptr_neuron_unit_it(layer_it->ptr_array_neuron_units);

    bool *tmp_ptr_array_mask_dropout(tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli);
    bool const *const tmp_ptr_array_mask_dropout_end(tmp_ptr_array_mask_dropout + *layer_it->ptr_number_neurons - 1u), // Subtract bias.
                    *tmp_ptr_array_previous_layer_mask_dropout(ptr_previous_layer_it_received->ptr_array_neuron_units->ptr_mask_dropout_bernoulli);

    size_t const tmp_number_connections(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections - 1u); // Subtract bias.
    size_t tmp_neuron_index;

    var *tmp_ptr_array_mask_dropout_parameters(this->ptr_array_mask_dropout_parameters + *tmp_ptr_neuron_unit_it->ptr_first_forward_connection_index),
         *tmp_ptr_array_scales(tmp_ptr_neuron_unit_it->ptr_scale),
         *tmp_ptr_array_shifts(tmp_ptr_neuron_unit_it->ptr_shift);
    
    struct dim3 const *tmp_ptr_array_dim3_grid_connections(tmp_ptr_neuron_unit_it->ptr_dim3_grid_connections),
                              *tmp_ptr_array_dim3_block_connections(tmp_ptr_neuron_unit_it->ptr_dim3_block_connections);

    if(USE_PARALLEL && *layer_it->ptr_number_neurons - 1u >= warpSize)
    {
        // Critical synchronization required. To see the previous neurons flag.
        ref_sync_code_received = 1u;
        
        LAUNCH_KERNEL_POINTER_1D(Dropout_Bernoulli__Neurons__Batch_Renormalization<var>,
                                                            layer_it->ptr_dim3_grid_neurons_cuRAND,
                                                            layer_it->ptr_dim3_block_neurons_cuRAND,
                                                            0_UZ,
                                                            *layer_it->ptr_number_neurons - 1, // Subtract bias.
                                                            tmp_number_connections,
                                                            tmp_ptr_array_previous_layer_mask_dropout,
                                                            tmp_ptr_array_mask_dropout,
                                                            this->ptr_array_parameters,
                                                            tmp_ptr_array_scales,
                                                            tmp_ptr_array_shifts,
                                                            this->ptr_array_mask_dropout_parameters,
                                                            tmp_ptr_array_mask_dropout_parameters,
                                                            layer_it->dropout_values[0],
                                                            this->ptr_array_cuRAND_State_MTGP32_neuroyed,
                                                            tmp_ptr_array_dim3_grid_connections,
                                                            tmp_ptr_array_dim3_block_connections)
    }
    else
    {
        // Need a synchronization to see the previous neurons flag.
        if(ref_sync_code_received == 1u)
        {
            // Set the state at zero (Synchronized).
            ref_sync_code_received = 0u;

            CUDA__Check_Error();
        }
        
        for(tmp_neuron_index = 0_UZ; tmp_ptr_array_mask_dropout != tmp_ptr_array_mask_dropout_end; ++tmp_ptr_array_mask_dropout,
                                                                                                                                                   ++tmp_neuron_index,
                                                                                                                                                   ++tmp_ptr_array_scales,
                                                                                                                                                   ++tmp_ptr_array_shifts,
                                                                                                                                                   ++tmp_ptr_array_dim3_grid_connections,
                                                                                                                                                   ++tmp_ptr_array_dim3_block_connections,
                                                                                                                                                   tmp_ptr_array_mask_dropout_parameters += tmp_number_connections + 1u) // Add bias.
        {
            if(cuRAND_Bernoulli(layer_it->dropout_values[0], curand_uniform(this->ptr_array_cuRAND_State_MTGP32_neuroyed)))
            {
                *tmp_ptr_array_mask_dropout = true;
                
                Flag_1D<var>(tmp_number_connections,
                                    tmp_ptr_array_previous_layer_mask_dropout,
                                    tmp_ptr_array_mask_dropout_parameters,
                                    tmp_ptr_array_dim3_grid_connections,
                                    tmp_ptr_array_dim3_block_connections);

                tmp_ptr_array_mask_dropout_parameters[tmp_number_connections] = 1_r; // Bias.

                // TODO: Optimize with a real index.
                this->ptr_array_mask_dropout_parameters[static_cast<size_t>(tmp_ptr_array_scales - this->ptr_array_parameters)] = 1_r; // Batch normalization scale.
                this->ptr_array_mask_dropout_parameters[static_cast<size_t>(tmp_ptr_array_shifts - this->ptr_array_parameters)] = 1_r; // Batch normalization shift.
            }
            else // Dropout neuron
            {
                *tmp_ptr_array_mask_dropout = false;
                
                Zero_1D<var>(tmp_number_connections,
                                    tmp_ptr_array_mask_dropout_parameters,
                                    tmp_ptr_array_dim3_grid_connections,
                                    tmp_ptr_array_dim3_block_connections);

                tmp_ptr_array_mask_dropout_parameters[tmp_number_connections] = 0_r; // Bias.

                // TODO: Optimize with a real index.
                this->ptr_array_mask_dropout_parameters[static_cast<size_t>(tmp_ptr_array_scales - this->ptr_array_parameters)] = 0_r; // Batch normalization scale.
                this->ptr_array_mask_dropout_parameters[static_cast<size_t>(tmp_ptr_array_shifts - this->ptr_array_parameters)] = 0_r; // Batch normalization shift.
            }
        }

        // If number of connections is bigger or equal to 32. We need a synchronization to see the connections flag.
        if(tmp_number_connections >= warpSize) { ref_sync_code_received = 2u; }
    }

    //INFO(L"tmp_count_dropped: %u / %u, %f%%",
    //                        tmp_count_dropped,
    //                        tmp_count_total,
    //                        static_cast<float>(tmp_count_dropped) / static_cast<float>(tmp_count_total) * 100.0f);
}
