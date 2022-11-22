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
#include "deep-learning-lib/ops/reduce.cuh"
#include "deep-learning-lib/ops/multiply.cuh"

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons(size_t const data_index_received,
                                                                                                                                              size_t const number_neurons_received,
                                                                                                                                              size_t const next_layer_number_neurons_received,
                                                                                                                                              size_t const neurons_total_reduce_error_size_received,
                                                                                                                                              T const *const ptr_array_layer_it_summations_received,
                                                                                                                                              T const *const ptr_array_layer_it_values_received,
                                                                                                                                              T *const ptr_array_layer_it_errors_received,
                                                                                                                                              T **const ptr_array_layer_it_reduce_errors_received,
                                                                                                                                              T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                              T const *const ptr_array_next_layer_errors_received,
                                                                                                                                              enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                              struct dim3 const *const ptr_array_dim3_grid_reduce_errors_received,
                                                                                                                                              struct dim3 const *const ptr_array_dim3_block_reduce_errors_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T *const tmp_ptr_array_reduce_error(ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received);
    
    Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                        number_neurons_received,
                                        tmp_ptr_array_reduce_error,
                                        ptr_array_next_layer_parameters_received + tmp_thread_global_index * next_layer_number_neurons_received,
                                        ptr_array_next_layer_errors_received,
                                        ptr_array_dim3_grid_reduce_errors_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_errors_received + tmp_thread_global_index);

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced error of the neuron.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    ptr_array_layer_it_errors_received[tmp_thread_global_index] = *tmp_ptr_array_reduce_error; // Reduced error.

    ptr_array_layer_it_errors_received[tmp_thread_global_index] *= Activation_Derived(1_r,
                                                                                                                           ptr_array_layer_it_summations_received[tmp_thread_global_index],
                                                                                                                           ptr_array_layer_it_values_received[tmp_thread_global_index],
                                                                                                                           ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                              size_t const data_index_received,
                                                                                                                                              size_t const number_neurons_received,
                                                                                                                                              size_t const next_layer_number_neurons_received,
                                                                                                                                              size_t const neurons_total_reduce_error_size_received,
                                                                                                                                              T  const *const ptr_array_layer_it_summations_received,
                                                                                                                                              T const *const ptr_array_layer_it_values_received,
                                                                                                                                              T *const ptr_array_layer_it_errors_received,
                                                                                                                                              T **const ptr_array_layer_it_reduce_errors_received,
                                                                                                                                              T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                              T const *const ptr_array_next_layer_errors_received,
                                                                                                                                              enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                              struct dim3 const *const ptr_array_dim3_grid_reduce_errors_received,
                                                                                                                                              struct dim3 const *const ptr_array_dim3_block_reduce_errors_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T *tmp_ptr_array_reduce_error;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_reduce_error = ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received;
        
        Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                            number_neurons_received,
                                            tmp_ptr_array_reduce_error,
                                            ptr_array_next_layer_parameters_received + tmp_thread_global_index * next_layer_number_neurons_received,
                                            ptr_array_next_layer_errors_received,
                                            ptr_array_dim3_grid_reduce_errors_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_errors_received + tmp_thread_global_index);
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced error of the neuron.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_layer_it_errors_received[tmp_thread_global_index] = *tmp_ptr_array_reduce_error; // Reduced error.
        
        ptr_array_layer_it_errors_received[tmp_thread_global_index] *= Activation_Derived(1_r,
                                                                                                                               ptr_array_layer_it_summations_received[tmp_thread_global_index],
                                                                                                                               ptr_array_layer_it_values_received[tmp_thread_global_index],
                                                                                                                               ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                        size_t const data_index_received,
                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                        size_t const next_layer_number_neurons_received,
                                                                                                                                                        size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                        T const *const ptr_array_layer_it_summations_received,
                                                                                                                                                        T const *const ptr_array_layer_it_values_received,
                                                                                                                                                        T *const ptr_array_layer_it_errors_received,
                                                                                                                                                        T **const ptr_array_layer_it_reduce_errors_received,
                                                                                                                                                        T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                                        T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                        enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_reduce_errors_received,
                                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_reduce_errors_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    // Loop through each neurons.
    do
    {
        Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                            number_neurons_received,
                                            ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received,
                                            ptr_array_next_layer_parameters_received + tmp_thread_global_index * next_layer_number_neurons_received,
                                            ptr_array_next_layer_errors_received,
                                            ptr_array_dim3_grid_reduce_errors_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_errors_received + tmp_thread_global_index);
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced error of the neuron.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop through each neurons.
    do
    {
        ptr_array_layer_it_errors_received[tmp_thread_global_index] = *(ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received); // Reduced error.
        
        ptr_array_layer_it_errors_received[tmp_thread_global_index] *= Activation_Derived(1_r,
                                                                                                                               ptr_array_layer_it_summations_received[tmp_thread_global_index],
                                                                                                                               ptr_array_layer_it_values_received[tmp_thread_global_index],
                                                                                                                               ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                            size_t const next_layer_number_neurons_received,
                                                                                                                                            size_t const neurons_total_reduce_error_size_received,
                                                                                                                                            T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                            T const *const ptr_array_next_layer_errors_received,
                                                                                                                                            struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T>,
                                                      layer_it->ptr_dim3_grid_neurons,
                                                      layer_it->ptr_dim3_block_neurons,
                                                      0_UZ,
                                                      number_neurons_received - 1, // Subtract bias.
                                                      tmp_thread_global_index,
                                                      number_neurons_received,
                                                      next_layer_number_neurons_received,
                                                      neurons_total_reduce_error_size_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                      ptr_array_next_layer_parameters_received,
                                                      ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                      tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error)
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                            size_t const next_layer_number_neurons_received,
                                                                                                                                            size_t const neurons_total_reduce_error_size_received,
                                                                                                                                            T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                            T const *const ptr_array_next_layer_errors_received,
                                                                                                                                            struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);
    
    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T>,
                                                          layer_it->ptr_dim3_grid_neurons,
                                                          layer_it->ptr_dim3_block_neurons,
                                                          0_UZ,
                                                          number_neurons_received - 1, // Subtract bias.
                                                          tmp_thread_global_index,
                                                          number_neurons_received,
                                                          next_layer_number_neurons_received,
                                                          neurons_total_reduce_error_size_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                          ptr_array_next_layer_parameters_received,
                                                          ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                          tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error)
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                     size_t const number_neurons_received,
                                                                                                                                                     size_t const next_layer_number_neurons_received,
                                                                                                                                                     size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                     T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                                     T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                     struct cuLayer *const layer_it)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    // KERNEL LAUNCH
    //    1: Launching do-while elements.
    if(layer_it->ptr_dim3_grid_neurons->x * layer_it->ptr_dim3_block_neurons->x < number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel_while__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                        tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                        number_neurons_received,
                                                                                                                                                                                                                                                                                                        next_layer_number_neurons_received,
                                                                                                                                                                                                                                                                                                        neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                        ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                                                                                                        ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    //    2: Launching size condition.
    else if(layer_it->ptr_dim3_grid_neurons->x * layer_it->ptr_dim3_block_neurons->x > number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                number_neurons_received,
                                                                                                                                                                                                                                                                                                next_layer_number_neurons_received,
                                                                                                                                                                                                                                                                                                neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                                                                                                ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    //    3: Standard.
    else
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                number_neurons_received,
                                                                                                                                                                                                                                                                                                next_layer_number_neurons_received,
                                                                                                                                                                                                                                                                                                neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                                                                                                ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons(size_t const number_neurons_received,
                                                                                                                                            size_t const next_layer_number_neurons_received,
                                                                                                                                            size_t const neurons_total_reduce_error_size_received,
                                                                                                                                            T const *ptr_array_next_layer_parameters_received,
                                                                                                                                            T const *const ptr_array_next_layer_errors_received,
                                                                                                                                            struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                            struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    var const *const tmp_ptr_next_layer_errors(ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u)); // Add bias.

    struct cuNeuron *tmp_ptr_neuron_unit_it;

    // Loop through each neurons.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                            ptr_array_next_layer_parameters_received += next_layer_number_neurons_received)
    {
        Reduce::Reduce_XX<var>(next_layer_number_neurons_received,
                                                number_neurons_received,
                                                *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received,
                                                ptr_array_next_layer_parameters_received,
                                                tmp_ptr_next_layer_errors,
                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
    }

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // Loop through each neurons.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
    {
        tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received); // Reduced error.

        tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] *= Activation_Derived(tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
    }
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                            size_t const next_layer_number_neurons_received,
                                                                                                                                            size_t const neurons_total_reduce_error_size_received,
                                                                                                                                            T const *ptr_array_next_layer_parameters_received,
                                                                                                                                            T const *const ptr_array_next_layer_errors_received,
                                                                                                                                            struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                            struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    var const *tmp_ptr_next_layer_errors;
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_next_layer_errors = ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u); // Add bias.

        // Loop through each neurons.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             ptr_array_next_layer_parameters_received += next_layer_number_neurons_received)
        {
            Reduce::Reduce_XX<var>(next_layer_number_neurons_received,
                                                 number_neurons_received,
                                                 *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received,
                                                 ptr_array_next_layer_parameters_received,
                                                 tmp_ptr_next_layer_errors,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
        }
    }

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received); // Reduced error.

            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] *= Activation_Derived(tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                    tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                    *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
        }
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                    size_t const next_layer_number_neurons_received,
                                                                                                                                                    size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                    T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                                    T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                    struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                    struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    var const *tmp_ptr_next_layer_parameters,
                 *tmp_ptr_next_layer_errors;
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;

    // Loop through each sample.
    do
    {
        tmp_ptr_next_layer_parameters = ptr_array_next_layer_parameters_received;
        tmp_ptr_next_layer_errors = ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u); // Add bias.

        // Loop through each neurons.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             tmp_ptr_next_layer_parameters += next_layer_number_neurons_received)
        {
            Reduce::Reduce_XX<var>(next_layer_number_neurons_received,
                                                 number_neurons_received,
                                                 *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received,
                                                 tmp_ptr_next_layer_parameters,
                                                 tmp_ptr_next_layer_errors,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Loop through each sample.
    do
    {
        // Loop through each neurons.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received); // Reduced error.

            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] *= Activation_Derived(tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                    tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                                                                                                                                                    *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::Backward_Pass__FC_to_FC(bool &ref_synchronized_received,
                                                                                                        size_t const batch_size,
                                                                                                        struct cuLayer *const layer_it,
                                                                                                        struct cuLayer *const ptr_next_layer_received,
                                                                                                        struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                        struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    size_t const tmp_number_neuron_units(*layer_it->ptr_number_neurons),
                                tmp_next_layer_number_neurons(*ptr_next_layer_received->ptr_number_neurons - 1u); // Subtract bias.
    size_t tmp_data_index;
    
    struct cuNeuron const *const tmp_ptr_next_layer_first_neuron(ptr_next_layer_received->ptr_array_neuron_units);
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units),
                                             *tmp_ptr_neuron_unit_it;
    
    var const *tmp_ptr_next_layer_parameters,
                  *tmp_ptr_next_layer_errors;
    
    // Condition to enter into dynamic parallelisme of each sample.
    if(USE_PARALLEL && batch_size >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // Condition to enter into dynamic parallelisme of each sample and neurons.
        if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
        {
            LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Parallel_Batch__Parallel_Neurons<var>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_UZ,
                                                              batch_size,
                                                              tmp_number_neuron_units,
                                                              tmp_next_layer_number_neurons,
                                                              this->neurons_total_reduce_error_size,
                                                              this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_next_layer_first_neuron->ptr_array_errors,
                                                              layer_it)
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Parallel_Batch__Serialize_Neurons<var>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_UZ,
                                                              batch_size,
                                                              tmp_number_neuron_units,
                                                              tmp_next_layer_number_neurons,
                                                              this->neurons_total_reduce_error_size,
                                                              this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_next_layer_first_neuron->ptr_array_errors,
                                                              tmp_ptr_layer_it_first_neuron,
                                                              layer_it->ptr_last_neuron_unit - 1u) // Subtract bias.
        }
    }
    // Condition to enter into dynamic parallelisme of each neurons.
    if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // KERNEL LAUNCH
        //    1: Launching do-while elements.
        if(layer_it->ptr_dim3_grid_neurons_DP->x * layer_it->ptr_dim3_block_neurons_DP->x < tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel_while__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                            tmp_data_index,
                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                            tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                            this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                            this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_next_layer_first_neuron->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
            }
        }
        //    2: Launching size condition.
        else if(layer_it->ptr_dim3_grid_neurons_DP->x * layer_it->ptr_dim3_block_neurons_DP->x > tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                    tmp_data_index,
                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                    this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                    this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_next_layer_first_neuron->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
            }
        }
        //    3: Standard.
        else
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__Backward_Pass__FC_to_FC__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                    tmp_data_index,
                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                    this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                    this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_next_layer_first_neuron->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
            }
        }
        // |END| KERNEL LAUNCH |END|
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        struct cuNeuron const *const tmp_ptr_last_neuron_unit(layer_it->ptr_last_neuron_unit - 1);

        // Synchronisation before using the transposed error of the layer.
        CUDA__Device_Synchronise(ref_synchronized_received, DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
        
        // Loop through each sample.
        for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
        {
            tmp_ptr_next_layer_parameters = this->ptr_array_transposed_weights + *tmp_ptr_next_layer_first_neuron->ptr_first_forward_connection_index;
            tmp_ptr_next_layer_errors = tmp_ptr_next_layer_first_neuron->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u); // Add bias.

            // Loop through each neurons.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                         tmp_ptr_next_layer_parameters += tmp_next_layer_number_neurons)
            {
                Reduce::Reduce_XX<var>(tmp_next_layer_number_neurons,
                                                      tmp_number_neuron_units,
                                                      *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_data_index * this->neurons_total_reduce_error_size,
                                                      tmp_ptr_next_layer_parameters,
                                                      tmp_ptr_next_layer_errors,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
            }
        }

        // Do we need to synchronise? Based on "Reduce_XX" Function.
        // => Synchronisation before using the reduced error of the neuron.
        if(tmp_next_layer_number_neurons >= warpSize * 2u) { CUDA__Check_Error(); }
        
        // Loop through each sample.
        for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
        {
            // Loop through each neurons.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
            {
                tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_data_index * tmp_number_neuron_units] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_data_index * this->neurons_total_reduce_error_size); // Reduced error.

                tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_data_index * tmp_number_neuron_units] *= Activation_Derived(tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units],
                                                                                                                                                     tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units],
                                                                                                                                                     *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
            }
        }
    }
}
