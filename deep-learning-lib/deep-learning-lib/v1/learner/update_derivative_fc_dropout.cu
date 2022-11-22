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

template<typename T>
__global__ void kernel__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                                        bool const *const ptr_array_mask_dropout_received,
                                                                                                                                                                                        T *const ptr_array_derivatives_paramters_received,
                                                                                                                                                                                        T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(ptr_array_mask_dropout_received[tmp_thread_global_index])
    {
        T const tmp_error(ptr_array_neuron_units_errors_received[tmp_thread_global_index]);

        Multiply::FMAC_X_YX_1D<T>(number_connections_received,
                                                    ptr_array_derivatives_paramters_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                    tmp_error,
                                                    ptr_array_previous_layer_outputs_received,
                                                    ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                    ptr_array_dim3_block_connections_received + tmp_thread_global_index);
    
        ptr_array_derivatives_paramters_received[tmp_thread_global_index * (number_connections_received + 1u) + number_connections_received] += tmp_error; // Bias.
    }
}

template<typename T>
__global__ void kernel__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                                        bool const *const ptr_array_mask_dropout_received,
                                                                                                                                                                                        T *const ptr_array_derivatives_paramters_received,
                                                                                                                                                                                        T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                        struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(tmp_thread_global_index < size_received && ptr_array_mask_dropout_received[tmp_thread_global_index])
    {
        T const tmp_error(ptr_array_neuron_units_errors_received[tmp_thread_global_index]);
        
        Multiply::FMAC_X_YX_1D<T>(number_connections_received,
                                                    ptr_array_derivatives_paramters_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                    tmp_error,
                                                    ptr_array_previous_layer_outputs_received,
                                                    ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                    ptr_array_dim3_block_connections_received + tmp_thread_global_index);
        
        ptr_array_derivatives_paramters_received[tmp_thread_global_index * (number_connections_received + 1u) + number_connections_received] += tmp_error; // Bias.
    }
}

template<typename T>
__global__ void kernel_while__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                                bool const *const ptr_array_mask_dropout_received,
                                                                                                                                                                                                T *const ptr_array_derivatives_paramters_received,
                                                                                                                                                                                                T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_error;

    do
    {
        if(ptr_array_mask_dropout_received[tmp_thread_global_index])
        {
            tmp_error = ptr_array_neuron_units_errors_received[tmp_thread_global_index];
        
            Multiply::FMAC_X_YX_1D<T>(number_connections_received,
                                                        ptr_array_derivatives_paramters_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                        tmp_error,
                                                        ptr_array_previous_layer_outputs_received,
                                                        ptr_array_dim3_grid_connections_received + tmp_thread_global_index,
                                                        ptr_array_dim3_block_connections_received + tmp_thread_global_index);
        
            ptr_array_derivatives_paramters_received[tmp_thread_global_index * (number_connections_received + 1u) + number_connections_received] += tmp_error; // Bias.
        }

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                                    size_t const total_parameters_allocated_received,
                                                                                                                                                                                    T *const ptr_array_derivatives_parameters_received,
                                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                    struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T>,
                                                      layer_it->ptr_dim3_grid_neurons,
                                                      layer_it->ptr_dim3_block_neurons,
                                                      0_UZ,
                                                      number_neurons_received - 1, // Subtract bias.
                                                      number_neurons_received,
                                                      number_connections_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                      ptr_array_derivatives_parameters_received + blockIdx.x * total_parameters_allocated_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                      ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                      tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                      tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections)
}

template<typename T>
__global__ void kernel__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                                    size_t const number_connections_received,
                                                                                                                                                                                    size_t const total_parameters_allocated_received,
                                                                                                                                                                                    T *const ptr_array_derivatives_parameters_received,
                                                                                                                                                                                    T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                    struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);
    
    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T>,
                                                          layer_it->ptr_dim3_grid_neurons,
                                                          layer_it->ptr_dim3_block_neurons,
                                                          0_UZ,
                                                          number_neurons_received - 1, // Subtract bias.
                                                          number_neurons_received,
                                                          number_connections_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                          ptr_array_derivatives_parameters_received + blockIdx.x * total_parameters_allocated_received,
                                                          tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                          ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections)
    }
}

template<typename T>
__global__ void kernel_while__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                                size_t const total_parameters_allocated_received,
                                                                                                                                                                                                T *const ptr_array_derivatives_parameters_received,
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
            kernel_while__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                      number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                      number_connections_received,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                                      ptr_array_derivatives_parameters_received + blockIdx.x * total_parameters_allocated_received,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                                      ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
        
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
            kernel__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                             number_neurons_received,
                                                                                                                                                                                                                                                                                                                                             number_connections_received,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                             ptr_array_derivatives_parameters_received + blockIdx.x * total_parameters_allocated_received,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                             ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
        
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
            kernel__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<T> <<< *layer_it->ptr_dim3_grid_neurons, *layer_it->ptr_dim3_block_neurons >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                          number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          number_connections_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                          ptr_array_derivatives_parameters_received + blockIdx.x * total_parameters_allocated_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

template<typename T>
__global__ void kernel__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Serialize_Neurons(size_t const number_neurons_received,
                                                                                                                                                                                         size_t const number_connections_received,
                                                                                                                                                                                         size_t const total_parameters_allocated_received,
                                                                                                                                                                                         bool const *ptr_array_mask_dropout_received,
                                                                                                                                                                                         T *const ptr_array_derivatives_parameters_received,
                                                                                                                                                                                         T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                         T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                         struct dim3 const *ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                         struct dim3 const *ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    var const *const tmp_ptr_array_previous_layer_neurons_values(ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u)), // Add bias.
                 *tmp_ptr_array_errors(ptr_array_neuron_units_errors_received + tmp_thread_global_index * number_neurons_received),
                 *const tmp_ptr_array_errors_end(tmp_ptr_array_errors + number_neurons_received - 1u); // Subtract bias.
    var *tmp_ptr_array_derivatives_parameters(ptr_array_derivatives_parameters_received + tmp_thread_global_index * total_parameters_allocated_received),
         tmp_error;

    for(; tmp_ptr_array_errors != tmp_ptr_array_errors_end; ++tmp_ptr_array_errors,
                                                                                    ++ptr_array_mask_dropout_received,
                                                                                    ++ptr_array_dim3_grid_connections_received,
                                                                                    ++ptr_array_dim3_block_connections_received,
                                                                                    tmp_ptr_array_derivatives_parameters += number_connections_received + 1u) // Add bias.
    {
        if(*ptr_array_mask_dropout_received)
        {
            tmp_error = *tmp_ptr_array_errors;

            Multiply::FMAC_X_YX_1D<var>(number_connections_received,
                                                            tmp_ptr_array_derivatives_parameters,
                                                            tmp_error,
                                                            tmp_ptr_array_previous_layer_neurons_values,
                                                            ptr_array_dim3_grid_connections_received,
                                                            ptr_array_dim3_block_connections_received);
            
            tmp_ptr_array_derivatives_parameters[number_connections_received] += tmp_error; // Bias.
        }
    }
}

template<typename T>
__global__ void kernel__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                                        size_t const number_connections_received,
                                                                                                                                                                                        size_t const total_parameters_allocated_received,
                                                                                                                                                                                        bool const *ptr_array_mask_dropout_received,
                                                                                                                                                                                        T *const ptr_array_derivatives_parameters_received,
                                                                                                                                                                                        T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                        T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                        struct dim3 const *ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                        struct dim3 const *ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    var const *tmp_ptr_array_previous_layer_neurons_values,
                 *tmp_ptr_array_errors,
                 *tmp_ptr_array_errors_end;
    var *tmp_ptr_array_derivatives_parameters,
         tmp_error;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_previous_layer_neurons_values = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u); // Add bias.

        tmp_ptr_array_errors = ptr_array_neuron_units_errors_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_errors_end = tmp_ptr_array_errors + number_neurons_received - 1_UZ; // Subtract bias.
        
        tmp_ptr_array_derivatives_parameters = ptr_array_derivatives_parameters_received + tmp_thread_global_index * total_parameters_allocated_received;

        for(; tmp_ptr_array_errors != tmp_ptr_array_errors_end; ++tmp_ptr_array_errors,
                                                                                        ++ptr_array_mask_dropout_received,
                                                                                        ++ptr_array_dim3_grid_connections_received,
                                                                                        ++ptr_array_dim3_block_connections_received,
                                                                                        tmp_ptr_array_derivatives_parameters += number_connections_received + 1u) // Add bias.
        {
            if(*ptr_array_mask_dropout_received)
            {
                tmp_error = *tmp_ptr_array_errors;

                Multiply::FMAC_X_YX_1D<var>(number_connections_received,
                                                              tmp_ptr_array_derivatives_parameters,
                                                              tmp_error,
                                                              tmp_ptr_array_previous_layer_neurons_values,
                                                              ptr_array_dim3_grid_connections_received,
                                                              ptr_array_dim3_block_connections_received);
            
                tmp_ptr_array_derivatives_parameters[number_connections_received] += tmp_error; // Bias.
            }
        }
    }
}

template<typename T>
__global__ void kernel_while__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                                size_t const number_connections_received,
                                                                                                                                                                                                size_t const total_parameters_allocated_received,
                                                                                                                                                                                                bool const *const ptr_array_mask_dropout_received,
                                                                                                                                                                                                T *const ptr_array_derivatives_parameters_received,
                                                                                                                                                                                                T const *const ptr_array_neuron_units_errors_received,
                                                                                                                                                                                                T const *const ptr_array_previous_layer_outputs_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_grid_connections_received,
                                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_block_connections_received)
{
    size_t const tmp_thread_grid_index(blockIdx.x * blockDim.x + threadIdx.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    bool const *tmp_ptr_array_mask_dropout;

    var const *tmp_ptr_array_previous_layer_neurons_values,
                 *tmp_ptr_array_errors,
                 *tmp_ptr_array_errors_end;
    var *tmp_ptr_array_derivatives_parameters,
         tmp_error;

    struct dim3 const *tmp_ptr_array_dim3_grid_connections,
                              *tmp_ptr_array_dim3_block_connections;
    
    // Loop through each sample.
    do
    {
        tmp_ptr_array_mask_dropout = ptr_array_mask_dropout_received;

        tmp_ptr_array_previous_layer_neurons_values = ptr_array_previous_layer_outputs_received + tmp_thread_global_index * (number_connections_received + 1u); // Add bias.

        tmp_ptr_array_errors = ptr_array_neuron_units_errors_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_errors_end = tmp_ptr_array_errors + number_neurons_received - 1_UZ; // Subtract bias.
        
        tmp_ptr_array_derivatives_parameters = ptr_array_derivatives_parameters_received + tmp_thread_grid_index * total_parameters_allocated_received;

        tmp_ptr_array_dim3_grid_connections = ptr_array_dim3_grid_connections_received;
        tmp_ptr_array_dim3_block_connections = ptr_array_dim3_block_connections_received;

        for(; tmp_ptr_array_errors != tmp_ptr_array_errors_end; ++tmp_ptr_array_errors,
                                                                                        ++tmp_ptr_array_mask_dropout,
                                                                                        ++tmp_ptr_array_dim3_grid_connections,
                                                                                        ++tmp_ptr_array_dim3_block_connections,
                                                                                        tmp_ptr_array_derivatives_parameters += number_connections_received + 1u) // Add bias.
        {
            if(*tmp_ptr_array_mask_dropout)
            {
                tmp_error = *tmp_ptr_array_errors;

                Multiply::FMAC_X_YX_1D<var>(number_connections_received,
                                                              tmp_ptr_array_derivatives_parameters,
                                                              tmp_error,
                                                              tmp_ptr_array_previous_layer_neurons_values,
                                                              tmp_ptr_array_dim3_grid_connections,
                                                              tmp_ptr_array_dim3_block_connections);
            
                tmp_ptr_array_derivatives_parameters[number_connections_received] += tmp_error; // Bias.
            }
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::Update_Derivative_Weight__FC_to_FC__Dropout(bool &ref_synchronized_received,
                                                                                                                                    size_t const batch_size,
                                                                                                                                    struct cuLayer *const layer_it,
                                                                                                                                    struct cuLayer const *const ptr_previous_layer_it_received,
                                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                    struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    size_t tmp_data_index;
    
    struct cuNeuron const *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units),
                                                      *const tmp_ptr_previous_layer_first_neuron(ptr_previous_layer_it_received->ptr_array_neuron_units);
    
    // TODO: Remove bias term in nConnections.
    // By subtracting the bias the variable "ptr_dim3_grid_connections" become a false dimension.
    size_t const tmp_number_connections(*tmp_ptr_layer_it_first_neuron->ptr_number_forward_connections - 1u), // Subtract bias.
                                tmp_number_neuron_units(*layer_it->ptr_number_neurons);

    bool const *tmp_ptr_array_mask_dropout;

    var const *tmp_ptr_array_previous_layer_neurons_values,
                 *tmp_ptr_array_errors,
                 *tmp_ptr_array_errors_end;
    var *tmp_ptr_array_derivatives_parameters,
         tmp_error;

    struct dim3 const *tmp_ptr_array_dim3_grid_connections,
                              *tmp_ptr_array_dim3_block_connections;
    
    // Condition to enter into dynamic parallelisme of each sample.
    if(USE_PARALLEL && batch_size >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // Condition to enter into dynamic parallelisme of each sample and neurons.
        if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
        {
            LAUNCH_KERNEL_POINTER_1D(cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Parallel_Neurons<var>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_UZ,
                                                              batch_size,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->total_parameters_allocated,
                                                              this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_array_values,
                                                              layer_it)
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            LAUNCH_KERNEL_POINTER_1D(cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Parallel_Batch__Serialize_Neurons<var>,
                                                              ptr_dim3_batch_size_grid_received,
                                                              ptr_dim3_batch_size_block_received,
                                                              0_UZ,
                                                              batch_size,
                                                              tmp_number_neuron_units,
                                                              tmp_number_connections,
                                                              this->total_parameters_allocated,
                                                              tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                              this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                              tmp_ptr_layer_it_first_neuron->ptr_array_errors,
                                                              tmp_ptr_previous_layer_first_neuron->ptr_array_values,
                                                              tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                              tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections)
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
                kernel_while__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                                    tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                                                    this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
            }
        }
        //    2: Launching size condition.
        else if(layer_it->ptr_dim3_grid_neurons_DP->x * layer_it->ptr_dim3_block_neurons_DP->x > tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                            tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                                            this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
            }
        }
        //    3: Standard.
        else
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__cuModel__Update_Derivative_Weight__FC__Dropout_Bernoulli__Serialize_Batch__Parallel_Neurons<var> <<< *layer_it->ptr_dim3_grid_neurons_DP, *layer_it->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                            tmp_number_connections,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli,
                                                                                                                                                                                                                                                                                                                                                            this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections,
                                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections);
            }
        }
        // |END| KERNEL LAUNCH |END|
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        // Loop through each sample.
        for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
        {
            tmp_ptr_array_mask_dropout = tmp_ptr_layer_it_first_neuron->ptr_mask_dropout_bernoulli;

            tmp_ptr_array_previous_layer_neurons_values = tmp_ptr_previous_layer_first_neuron->ptr_array_values + tmp_data_index * (tmp_number_connections + 1u); // Add bias.

            tmp_ptr_array_errors = tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units;
            tmp_ptr_array_errors_end = tmp_ptr_array_errors + tmp_number_neuron_units - 1_UZ; // Subtract bias.
            
            tmp_ptr_array_derivatives_parameters = this->ptr_array_derivatives_parameters + *tmp_ptr_layer_it_first_neuron->ptr_first_forward_connection_index;

            tmp_ptr_array_dim3_grid_connections = tmp_ptr_layer_it_first_neuron->ptr_dim3_grid_connections;
            tmp_ptr_array_dim3_block_connections = tmp_ptr_layer_it_first_neuron->ptr_dim3_block_connections;

            for(; tmp_ptr_array_errors != tmp_ptr_array_errors_end; ++tmp_ptr_array_errors,
                                                                                            ++tmp_ptr_array_mask_dropout,
                                                                                            ++tmp_ptr_array_dim3_grid_connections,
                                                                                            ++tmp_ptr_array_dim3_block_connections,
                                                                                            tmp_ptr_array_derivatives_parameters += tmp_number_connections + 1u) // Add bias.
            {
                if(*tmp_ptr_array_mask_dropout)
                {
                    tmp_error = *tmp_ptr_array_errors;

                    Multiply::FMAC_X_YX_1D<var>(tmp_number_connections,
                                                                  tmp_ptr_array_derivatives_parameters,
                                                                  tmp_error,
                                                                  tmp_ptr_array_previous_layer_neurons_values,
                                                                  tmp_ptr_array_dim3_grid_connections,
                                                                  tmp_ptr_array_dim3_block_connections);
                
                    tmp_ptr_array_derivatives_parameters[tmp_number_connections] += tmp_error; // Bias.
                }
            }
        }
    }
}
