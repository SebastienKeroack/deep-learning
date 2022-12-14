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
#include "deep-learning/ops/reduce.cuh"
#include "deep-learning/ops/multiply.cuh"

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons(size_t const data_index_received,
                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                        T const dropout_values,
                                                                                                                                                                        T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                        T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                        T *const ptr_array_layer_it_values_received,
                                                                                                                                                                        T const *const ptr_array_parameters_received,
                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                        enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u), // Add bias.
                               tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *const tmp_ptr_array_parameters(ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased);
    T *const tmp_ptr_array_reduce_summation(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received);

    Reduce::Reduce_XX<T>(number_connections_received,
                                        number_neurons_received,
                                        tmp_ptr_array_reduce_summation,
                                        tmp_ptr_array_parameters,
                                        ptr_array_previous_layer_outputs_received,
                                        ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);

    ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    ptr_array_layer_it_summations_received[tmp_thread_global_index] += *tmp_ptr_array_reduce_summation; // Reduced summation.
    
    Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                            ptr_array_layer_it_summations_received[tmp_thread_global_index],
                            ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);

    ptr_array_layer_it_values_received[tmp_thread_global_index] *= dropout_values;
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                        size_t const data_index_received,
                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                        T const dropout_values,
                                                                                                                                                                        T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                        T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                        T *const ptr_array_layer_it_values_received,
                                                                                                                                                                        T const *const ptr_array_parameters_received,
                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                        enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u), // Add bias.
                               tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *const tmp_ptr_array_parameters(ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased);
    T *tmp_ptr_array_reduce_summation;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_reduce_summation = ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received;

        Reduce::Reduce_XX<T>(number_connections_received,
                                              number_neurons_received,
                                              tmp_ptr_array_reduce_summation,
                                              tmp_ptr_array_parameters,
                                               ptr_array_previous_layer_outputs_received,
                                              ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                              ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
        
        ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_layer_it_summations_received[tmp_thread_global_index] += *tmp_ptr_array_reduce_summation; // Reduced summation.
        
        Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                               ptr_array_layer_it_summations_received[tmp_thread_global_index],
                               ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);

        ptr_array_layer_it_values_received[tmp_thread_global_index] *= dropout_values;
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                size_t const data_index_received,
                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                T const dropout_values,
                                                                                                                                                                                T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                                T *const ptr_array_layer_it_values_received,
                                                                                                                                                                                T const *const ptr_array_parameters_received,
                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u); // Add bias.
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *tmp_ptr_array_parameters;

    // Loop through each neurons.
    do
    {
        tmp_ptr_array_parameters = ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased;

        Reduce::Reduce_XX<T>(number_connections_received,
                                            number_neurons_received,
                                            ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received,
                                            tmp_ptr_array_parameters,
                                             ptr_array_previous_layer_outputs_received,
                                            ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
        
        ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop through each neurons.
    do
    {
        ptr_array_layer_it_summations_received[tmp_thread_global_index] += *(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received); // Reduced summation.
        
        Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                               ptr_array_layer_it_summations_received[tmp_thread_global_index],
                               ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);

        ptr_array_layer_it_values_received[tmp_thread_global_index] *= dropout_values;
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Parallel_Batch__Serialize_Neurons(size_t const number_neurons_received,
                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                        T const dropout_values,
                                                                                                                                                                        T const *ptr_array_parameters_received,
                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                        struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                        struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    var const *const tmp_ptr_array_previous_layer_outputs(ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u)); // Add bias.
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;
    
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                         ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
    {
        Reduce::Reduce_XX<var>(number_connections_received,
                                              number_neurons_received,
                                              *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                              ptr_array_parameters_received,
                                              tmp_ptr_array_previous_layer_outputs,
                                              tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                              tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

        tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = ptr_array_parameters_received[number_connections_received]; // Bias.
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // Loop through each neurons for retrieve reduced summation and then do the activation function.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
    {
        tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received); // Reduced summation.
        
        Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

        tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] *= dropout_values;
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                    size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                    T const dropout_values,
                                                                                                                                                                    T const *ptr_array_parameters_received,
                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                    struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                    struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    var const *tmp_ptr_array_previous_layer_outputs;
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;
    
    if(tmp_thread_global_index < size_received)
    {
        for(tmp_ptr_array_previous_layer_outputs = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
            tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
        {
            Reduce::Reduce_XX<var>(number_connections_received,
                                                  number_neurons_received,
                                                  *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                  ptr_array_parameters_received,
                                                  tmp_ptr_array_previous_layer_outputs,
                                                  tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                  tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = ptr_array_parameters_received[number_connections_received]; // Bias.
        }
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons for retrieve reduced summation and then do the activation function.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received); // Reduced summation.
        
            Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                   tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                   *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

            tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] *= dropout_values;
        }
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                T const dropout_values,
                                                                                                                                                                                T const *const ptr_array_parameters_received,
                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    var const *tmp_ptr_array_parameters,
                  *tmp_ptr_array_previous_layer_outputs;
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;

    // Loop through each sample.
    do
    {
        tmp_ptr_array_parameters = ptr_array_parameters_received;
        tmp_ptr_array_previous_layer_outputs = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u); // Add bias.

        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             tmp_ptr_array_parameters += number_connections_received + 1u) // Add bias.
        {
            Reduce::Reduce_XX<var>(number_connections_received,
                                                  number_neurons_received,
                                                  *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                  tmp_ptr_array_parameters,
                                                  tmp_ptr_array_previous_layer_outputs,
                                                  tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                  tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Loop through each sample.
    do
    {
        // Loop through each neurons for retrieve reduced summation and then do the activation function.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received); // Reduced summation.
        
            Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                   tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                   *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

            tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] *= dropout_values;
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Parallel_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                    size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                    T const dropout_values,
                                                                                                                                                                    T const *const ptr_array_parameters_received,
                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                    struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T>,
                                                        layer_it->ptr_dim3_grid_neurons,
                                                        layer_it->ptr_dim3_block_neurons,
                                                        0_UZ,
                                                        number_neurons_received - 1, // Subtract bias.
                                                        tmp_thread_global_index,
                                                        number_neurons_received,
                                                        number_connections_received,
                                                        neurons_total_reduce_summation_size_received,
                                                        dropout_values,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                        ptr_array_parameters_received,
                                                        ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation)
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                    size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                    T const dropout_values,
                                                                                                                                                                    T const *const ptr_array_parameters_received,
                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                    struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T>,
                                                          layer_it->ptr_dim3_grid_neurons,
                                                          layer_it->ptr_dim3_block_neurons,
                                                          0_UZ,
                                                          number_neurons_received - 1, // Subtract bias.
                                                          tmp_thread_global_index,
                                                          number_neurons_received,
                                                          number_connections_received,
                                                          neurons_total_reduce_summation_size_received,
                                                          dropout_values,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                          ptr_array_parameters_received,
                                                          ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                          tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation)
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                T const dropout_values,
                                                                                                                                                                                T const *const ptr_array_parameters_received,
                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
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
            kernel_while__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                    tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                                    number_neurons_received,
                                                                                                                                                                                                                                                                                                                                    number_connections_received,
                                                                                                                                                                                                                                                                                                                                    neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                                    dropout_values,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                    ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                                                    ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
        
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
            kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                        tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                        number_neurons_received,
                                                                                                                                                                                                                                                                                                                        number_connections_received,
                                                                                                                                                                                                                                                                                                                        neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                        dropout_values,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                                        ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
        
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
            kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                        number_neurons_received,
                                                                                                                                                                                                                                                                                                                        number_connections_received,
                                                                                                                                                                                                                                                                                                                        neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                        dropout_values,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                                        ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

__device__ void cuModel::Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing(bool &ref_synchronized_received,
                                                                                                                                size_t const batch_size,
                                                                                                                                struct cuLayer *const layer_it,
                                                                                                                                struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                                                                struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    size_t tmp_data_index;
    
    struct cuNeuron const *const tmp_ptr_previous_layer_first_neuron(ptr_previous_layer_it_received->ptr_array_neuron_units);
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units),
                                             *tmp_ptr_neuron_unit_it;
    
    // TODO: Remove bias term in nConnections.
    // By subtracting the bias the variable "ptr_dim3_grid_connections" become a false dimension.
    size_t const tmp_number_connections(*tmp_ptr_layer_it_first_neuron->ptr_number_forward_connections - 1u), // Subtract bias.
                                tmp_number_neuron_units(*layer_it->ptr_number_neurons);

    var const tmp_probability_retained_unit(layer_it->dropout_values[0]),
                  *tmp_ptr_array_previous_layer_outputs,
                  *tmp_ptr_array_parameters;
    
    // Condition to enter into dynamic parallelisme of each sample.
    if(USE_PARALLEL && batch_size >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // Condition to enter into dynamic parallelisme of each sample and neurons.
        if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
        {
            LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Parallel_Batch__Parallel_Neurons<var>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_UZ,
                                                              batch_size,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->neurons_total_reduce_summation_size,
                                                              tmp_probability_retained_unit,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_array_values,
                                                              layer_it)
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Parallel_Batch__Serialize_Neurons<var>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_UZ,
                                                              batch_size,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->neurons_total_reduce_summation_size,
                                                              tmp_probability_retained_unit,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_array_values,
                                                              tmp_ptr_layer_it_first_neuron,
                                                              layer_it->ptr_last_neuron_unit - 1) // Subtract bias.
        }
    }
    // Condition to enter into dynamic parallelisme of each neurons.
    else if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
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
                kernel_while__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                   tmp_data_index,
                                                                                                                                                                                                                                                                                                                                                   tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                   tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                                   this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                                   tmp_probability_retained_unit,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                   this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
            }
        }
        //    2: Launching size condition.
        else if(layer_it->ptr_dim3_grid_neurons_DP->x * layer_it->ptr_dim3_block_neurons_DP->x > tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                            tmp_data_index,
                                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                            this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                            tmp_probability_retained_unit,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
            }
        }
        //    3: Standard.
        else
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_data_index,
                                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                            this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                            tmp_probability_retained_unit,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
            }
        }
        // |END| KERNEL LAUNCH |END|
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        struct cuNeuron const *const tmp_ptr_last_neuron_unit(layer_it->ptr_last_neuron_unit - 1); // Subtract bias.
        
        // Synchronize if needed to see the output of the previous layer.
        CUDA__Device_Synchronise(ref_synchronized_received, DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
        
        // Loop through each sample.
        for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
        {
            tmp_ptr_array_parameters = this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index;
            tmp_ptr_array_previous_layer_outputs = tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u);

            // Loop through each neurons for doing a reduction of summation.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                         tmp_ptr_array_parameters += tmp_number_connections + 1u) // Add bias.
            {
                Reduce::Reduce_XX<var>(tmp_number_connections,
                                                      tmp_number_neuron_units,
                                                      *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_data_index * this->neurons_total_reduce_summation_size,
                                                      tmp_ptr_array_parameters,
                                                      tmp_ptr_array_previous_layer_outputs,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units] = tmp_ptr_array_parameters[tmp_number_connections]; // Bias.
            }
    
            // Do we need to synchronise? Based on "Reduce_XX" Function.
            // => Synchronize if needed to see the summation reduced of the layer.
            if(tmp_number_connections >= warpSize * 2u) { CUDA__Check_Error(); }

            // Loop through each neurons for retrieve reduced summation and then do the activation function.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
            {
                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_data_index * this->neurons_total_reduce_summation_size); // Reduced summation.
        
                Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units],
                                       tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units],
                                       *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

                tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units] *= tmp_probability_retained_unit;
            }
        }
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Serialize_Batch__Parallel_Neurons(size_t const data_index_received,
                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                        bool const *const ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                        T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                        T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                        T *const ptr_array_layer_it_values_received,
                                                                                                                                                                        T const *const ptr_array_parameters_received,
                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                        enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u), // Add bias.
                               tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *const tmp_ptr_array_parameters(ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased);
    T *const tmp_ptr_array_reduce_summation(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received);

    if(ptr_array_layer_it_dropout_mask_received[tmp_thread_global_index])
    {
        Reduce::Reduce_XX<T>(number_connections_received,
                                            number_neurons_received,
                                            tmp_ptr_array_reduce_summation,
                                            tmp_ptr_array_parameters,
                                            ptr_array_previous_layer_outputs_received,
                                            ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                            ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);

        ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
    }
    else { ptr_array_layer_it_values_received[tmp_thread_global_index] = T(0); }

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    if(ptr_array_layer_it_dropout_mask_received[tmp_thread_global_index])
    {
        ptr_array_layer_it_summations_received[tmp_thread_global_index] += *tmp_ptr_array_reduce_summation; // Reduced summation.
    
        Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                                ptr_array_layer_it_summations_received[tmp_thread_global_index],
                                ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                        size_t const data_index_received,
                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                        bool const *const ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                        T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                        T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                        T *const ptr_array_layer_it_values_received,
                                                                                                                                                                        T const *const ptr_array_parameters_received,
                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                        enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u), // Add bias.
                               tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *const tmp_ptr_array_parameters(ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased);
    T *tmp_ptr_array_reduce_summation;

    if(tmp_thread_global_index < size_received)
    {
        if(ptr_array_layer_it_dropout_mask_received[tmp_thread_global_index])
        {
            tmp_ptr_array_reduce_summation = ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received;

            Reduce::Reduce_XX<T>(number_connections_received,
                                                  number_neurons_received,
                                                  tmp_ptr_array_reduce_summation,
                                                  tmp_ptr_array_parameters,
                                                   ptr_array_previous_layer_outputs_received,
                                                  ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                                  ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
        
            ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
        }
        else { ptr_array_layer_it_values_received[tmp_thread_global_index] = T(0); }
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received && ptr_array_layer_it_dropout_mask_received[tmp_thread_global_index])
    {
        ptr_array_layer_it_summations_received[tmp_thread_global_index] += *tmp_ptr_array_reduce_summation; // Reduced summation.
        
        Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                               ptr_array_layer_it_summations_received[tmp_thread_global_index],
                               ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                size_t const data_index_received,
                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                bool const *const ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                                T *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                T **const ptr_array_layer_it_reduce_summations_received,
                                                                                                                                                                                T *const ptr_array_layer_it_values_received,
                                                                                                                                                                                T const *const ptr_array_parameters_received,
                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_grid_reduce_summations_received,
                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_block_reduce_summations_received)
{
    size_t const tmp_number_connections_biased(number_connections_received + 1u); // Add bias.
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *tmp_ptr_array_parameters;

    // Loop through each neurons.
    do
    {
        if(ptr_array_layer_it_dropout_mask_received[tmp_thread_global_index])
        {
            tmp_ptr_array_parameters = ptr_array_parameters_received + tmp_thread_global_index * tmp_number_connections_biased;

            Reduce::Reduce_XX<T>(number_connections_received,
                                                number_neurons_received,
                                                ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received,
                                                tmp_ptr_array_parameters,
                                                 ptr_array_previous_layer_outputs_received,
                                                ptr_array_dim3_grid_reduce_summations_received + tmp_thread_global_index,
                                                ptr_array_dim3_block_reduce_summations_received + tmp_thread_global_index);
        
            ptr_array_layer_it_summations_received[tmp_thread_global_index] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
        }
        else { ptr_array_layer_it_values_received[tmp_thread_global_index] = T(0); }

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop through each neurons.
    do
    {
        if(ptr_array_layer_it_dropout_mask_received[tmp_thread_global_index])
        {
            ptr_array_layer_it_summations_received[tmp_thread_global_index] += *(ptr_array_layer_it_reduce_summations_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_summation_size_received); // Reduced summation.
        
            Activation_Real(ptr_array_layer_it_values_received[tmp_thread_global_index],
                                   ptr_array_layer_it_summations_received[tmp_thread_global_index],
                                   ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);
        }

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Parallel_Batch__Serialize_Neurons(size_t const number_neurons_received,
                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                        bool const *const ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                        T const *ptr_array_parameters_received,
                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                        struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                        struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    var const *const tmp_ptr_array_previous_layer_outputs(ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u)); // Add bias.
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;
    
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                         ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
    {
        if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
        {
            Reduce::Reduce_XX<var>(number_connections_received,
                                                  number_neurons_received,
                                                  *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                  ptr_array_parameters_received,
                                                  tmp_ptr_array_previous_layer_outputs,
                                                  tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                  tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = ptr_array_parameters_received[number_connections_received]; // Bias.
        }
        else { tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] = T(0); }
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // Loop through each neurons for retrieve reduced summation and then do the activation function.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
    {
        if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
        {
            tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received); // Reduced summation.
        
            Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                    tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                    *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
        }
    }
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                        bool const *const ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                        T const *ptr_array_parameters_received,
                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                        struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                        struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    var const *tmp_ptr_array_previous_layer_outputs;
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;
    
    if(tmp_thread_global_index < size_received)
    {
        for(tmp_ptr_array_previous_layer_outputs = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
            tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             ptr_array_parameters_received += number_connections_received + 1u) // Add bias.
        {
            if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
            {
                Reduce::Reduce_XX<var>(number_connections_received,
                                                      number_neurons_received,
                                                      *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                      ptr_array_parameters_received,
                                                      tmp_ptr_array_previous_layer_outputs,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = ptr_array_parameters_received[number_connections_received]; // Bias.
            }
            else { tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] = T(0); }
        }
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons for retrieve reduced summation and then do the activation function.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
            {
                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received); // Reduced summation.
        
                Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                       tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                       *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
            }
        }
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                bool const *const ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                                T const *const ptr_array_parameters_received,
                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    var const *tmp_ptr_array_parameters,
                  *tmp_ptr_array_previous_layer_outputs;
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;

    // Loop through each sample.
    do
    {
        tmp_ptr_array_parameters = ptr_array_parameters_received;
        tmp_ptr_array_previous_layer_outputs = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u); // Add bias.

        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                             tmp_ptr_array_parameters += number_connections_received + 1u) // Add bias.
        {
            if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
            {
                Reduce::Reduce_XX<var>(number_connections_received,
                                                      number_neurons_received,
                                                      *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received,
                                                      tmp_ptr_array_parameters,
                                                      tmp_ptr_array_previous_layer_outputs,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_parameters[number_connections_received]; // Bias.
            }
            else { tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received] = T(0); }
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronize if needed to see the summation reduced of the layer.
    if(number_connections_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Loop through each sample.
    do
    {
        // Loop through each neurons for retrieve reduced summation and then do the activation function.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
            {
                tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_thread_global_index * neurons_total_reduce_summation_size_received); // Reduced summation.
        
                Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                       tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received],
                                       *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
            }
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Parallel_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                    size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                    bool const *const ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                    T const *const ptr_array_parameters_received,
                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                    struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Serialize_Batch__Parallel_Neurons<T>,
                                                        layer_it->ptr_dim3_grid_neurons,
                                                        layer_it->ptr_dim3_block_neurons,
                                                        0_UZ,
                                                        number_neurons_received - 1, // Subtract bias.
                                                        tmp_thread_global_index,
                                                        number_neurons_received,
                                                        number_connections_received,
                                                        neurons_total_reduce_summation_size_received,
                                                        ptr_array_layer_it_dropout_mask_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                        ptr_array_parameters_received,
                                                        ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation)
}

template<typename T>
__global__ void kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                        size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                        bool const *const ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                        T const *const ptr_array_parameters_received,
                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                        struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Serialize_Batch__Parallel_Neurons<T>,
                                                          layer_it->ptr_dim3_grid_neurons,
                                                          layer_it->ptr_dim3_block_neurons,
                                                          0_UZ,
                                                          number_neurons_received - 1, // Subtract bias.
                                                          tmp_thread_global_index,
                                                          number_neurons_received,
                                                          number_connections_received,
                                                          neurons_total_reduce_summation_size_received,
                                                          ptr_array_layer_it_dropout_mask_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                          ptr_array_parameters_received,
                                                          ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                          tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation)
    }
}

template<typename T>
__global__ void kernel_while__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                size_t const neurons_total_reduce_summation_size_received,
                                                                                                                                                                                bool const *const ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                                T const *const ptr_array_parameters_received,
                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
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
            kernel_while__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                    tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                                    number_neurons_received,
                                                                                                                                                                                                                                                                                                                                    number_connections_received,
                                                                                                                                                                                                                                                                                                                                    neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                                    ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                    ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                                                    ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
        
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
            kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                        tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                        number_neurons_received,
                                                                                                                                                                                                                                                                                                                        number_connections_received,
                                                                                                                                                                                                                                                                                                                        neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                        ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                                        ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
        
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
            kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                        number_neurons_received,
                                                                                                                                                                                                                                                                                                                        number_connections_received,
                                                                                                                                                                                                                                                                                                                        neurons_total_reduce_summation_size_received,
                                                                                                                                                                                                                                                                                                                        ptr_array_layer_it_dropout_mask_received,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                        ptr_array_parameters_received,
                                                                                                                                                                                                                                                                                                                        ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

__device__ void cuModel::Forward_Pass__FC_to_FC__Dropout(bool &ref_synchronized_received,
                                                                                                                                  size_t const batch_size,
                                                                                                                                  struct cuLayer *const layer_it,
                                                                                                                                  struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                                                                  struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                  struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    size_t tmp_data_index;
    
    struct cuNeuron const *const tmp_ptr_previous_layer_first_neuron(ptr_previous_layer_it_received->ptr_array_neuron_units);
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units),
                                             *tmp_ptr_neuron_unit_it;
    
    // TODO: Remove bias term in nConnections.
    // By subtracting the bias the variable "ptr_dim3_grid_connections" become a false dimension.
    size_t const tmp_number_connections(*tmp_ptr_layer_it_first_neuron->ptr_number_forward_connections - 1u), // Subtract bias.
                                tmp_number_neuron_units(*layer_it->ptr_number_neurons);

    var const *tmp_ptr_array_previous_layer_outputs,
                  *tmp_ptr_array_parameters;
    
    // Condition to enter into dynamic parallelisme of each sample.
    if(USE_PARALLEL && batch_size >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // Condition to enter into dynamic parallelisme of each sample and neurons.
        if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
        {
            LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Parallel_Batch__Parallel_Neurons<var>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_UZ,
                                                              batch_size,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->neurons_total_reduce_summation_size,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_mask_dropout_bernoulli,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_array_values,
                                                              layer_it)
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            LAUNCH_KERNEL_POINTER_1D(Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Parallel_Batch__Serialize_Neurons<var>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_UZ,
                                                              batch_size,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->neurons_total_reduce_summation_size,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_mask_dropout_bernoulli,
                                                              this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_array_values,
                                                              tmp_ptr_layer_it_first_neuron,
                                                              layer_it->ptr_last_neuron_unit - 1) // Subtract bias.
        }
    }
    // Condition to enter into dynamic parallelisme of each neurons.
    else if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
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
                kernel_while__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                   tmp_data_index,
                                                                                                                                                                                                                                                                                                                                                   tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                   tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                                   this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_previous_layer_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                   this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                                                   tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
            }
        }
        //    2: Launching size condition.
        else if(layer_it->ptr_dim3_grid_neurons_DP->x * layer_it->ptr_dim3_block_neurons_DP->x > tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                            tmp_data_index,
                                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                            this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_previous_layer_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
            }
        }
        //    3: Standard.
        else
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__Forward_Pass__FC_to_FC__Dropout_Bernoulli__Training__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_data_index,
                                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                            this->neurons_total_reduce_summation_size,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_previous_layer_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_summation,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_summation,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_summation);
            }
        }
        // |END| KERNEL LAUNCH |END|
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        struct cuNeuron const *const tmp_ptr_last_neuron_unit(layer_it->ptr_last_neuron_unit - 1); // Subtract bias.
        
        // Synchronize if needed to see the output of the previous layer.
        CUDA__Device_Synchronise(ref_synchronized_received, DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);
        
        // Loop through each sample.
        for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
        {
            tmp_ptr_array_parameters = this->ptr_array_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index;
            tmp_ptr_array_previous_layer_outputs = tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u);

            // Loop through each neurons for doing a reduction of summation.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                         tmp_ptr_array_parameters += tmp_number_connections + 1u) // Add bias.
            {
                if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
                {
                    Reduce::Reduce_XX<var>(tmp_number_connections,
                                                          tmp_number_neuron_units,
                                                          *tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_data_index * this->neurons_total_reduce_summation_size,
                                                          tmp_ptr_array_parameters,
                                                          tmp_ptr_array_previous_layer_outputs,
                                                          tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation,
                                                          tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation);

                    tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units] = tmp_ptr_array_parameters[tmp_number_connections]; // Bias.
                }
                else { tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units] = 0_r; }
            }
        }

        // Do we need to synchronise? Based on "Reduce_XX" Function.
        // => Synchronize if needed to see the summation reduced of the layer.
        if(tmp_number_connections >= warpSize * 2u) { CUDA__Check_Error(); }
        
        // Loop through each sample.
        for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
        {
            // Loop through each neurons for retrieve reduced summation and then do the activation function.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
            {
                if(*tmp_ptr_neuron_unit_it->ptr_mask_dropout_bernoulli)
                {
                    tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units] += *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_summation + tmp_data_index * this->neurons_total_reduce_summation_size); // Reduced summation.
        
                    Activation_Real(tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units],
                                           tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units],
                                           *tmp_ptr_neuron_unit_it->ptr_type_activation_function);
                }
            }
        }
    }
}
