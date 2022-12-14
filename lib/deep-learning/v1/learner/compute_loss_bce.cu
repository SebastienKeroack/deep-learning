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

template<typename T>
__global__ void kernel__cuModel__FF__Test__Binary_Cross_Entropy__Serialize_Batch__Parallel_Neurons(float *const ptr_array_accuracy_values_received,
                                                                                                                                                                        float *const ptr_array_loss_values_received,
                                                                                                                                                                        float const accurancy_variance_received,
                                                                                                                                                                        T const *const ptr_array_output_layer_outputs_received,
                                                                                                                                                                        T const *const ptr_array_desired_outputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_output_value,
       tmp_desired_value,
       tmp_difference;
    
    tmp_output_value = ptr_array_output_layer_outputs_received[tmp_thread_global_index];

    tmp_desired_value = ptr_array_desired_outputs_received[tmp_thread_global_index];

    tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
    
    Update_Error__Binary_Cross_Entropy(tmp_output_value,
                                                                        tmp_desired_value,
                                                                        ptr_array_loss_values_received + tmp_thread_global_index);
    
    Update_Accuracy(tmp_difference,
                                            accurancy_variance_received,
                                            ptr_array_accuracy_values_received);
}

template<typename T>
__global__ void kernel__cuModel__FF__Test__Binary_Cross_Entropy__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                        float *const ptr_array_accuracy_values_received,
                                                                                                                                                                        float *const ptr_array_loss_values_received,
                                                                                                                                                                        float const accurancy_variance_received,
                                                                                                                                                                        T const *const ptr_array_output_layer_outputs_received,
                                                                                                                                                                        T const *const ptr_array_desired_outputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_output_value,
       tmp_desired_value,
       tmp_difference;
    
    if(tmp_thread_global_index < size_received)
    {
        tmp_output_value = ptr_array_output_layer_outputs_received[tmp_thread_global_index];

        tmp_desired_value = ptr_array_desired_outputs_received[tmp_thread_global_index];

        tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
        
        Update_Error__Binary_Cross_Entropy(tmp_output_value,
                                                                            tmp_desired_value,
                                                                            ptr_array_loss_values_received + tmp_thread_global_index);
        
        Update_Accuracy(tmp_difference,
                                               accurancy_variance_received,
                                               ptr_array_accuracy_values_received);
    }
}

template<typename T>
__global__ void kernel_while__cuModel__FF__Test__Binary_Cross_Entropy__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                float *const ptr_array_accuracy_values_received,
                                                                                                                                                                                float *const ptr_array_loss_values_received,
                                                                                                                                                                                float const accurancy_variance_received,
                                                                                                                                                                                T const *const ptr_array_output_layer_outputs_received,
                                                                                                                                                                                T const *const ptr_array_desired_outputs_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_output_value,
       tmp_desired_value,
       tmp_difference;
    
    do
    {
        tmp_output_value = ptr_array_output_layer_outputs_received[tmp_thread_global_index];

        tmp_desired_value = ptr_array_desired_outputs_received[tmp_thread_global_index];

        tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
        
        Update_Error__Binary_Cross_Entropy(tmp_output_value,
                                                                            tmp_desired_value,
                                                                            ptr_array_loss_values_received + tmp_thread_global_index);
        
        Update_Accuracy(tmp_difference,
                                               accurancy_variance_received,
                                               ptr_array_accuracy_values_received);

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__cuModel__FF__Test__Binary_Cross_Entropy__Parallel_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                                                      float *const ptr_array_accuracy_values_received,
                                                                                                                                                                      float *const ptr_array_loss_values_received,
                                                                                                                                                                      float const accurancy_variance_received,
                                                                                                                                                                      T const *const ptr_array_output_layer_outputs_received,
                                                                                                                                                                      T **const ptr_array_desired_outputs_received,
                                                                                                                                                                      struct dim3 const *const ptr_array_dim3_grid_received,
                                                                                                                                                                      struct dim3 const *const ptr_array_dim3_block_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    LAUNCH_KERNEL_POINTER_1D(cuModel__FF__Test__Binary_Cross_Entropy__Serialize_Batch__Parallel_Neurons<T>,
                                                        ptr_array_dim3_grid_received,
                                                        ptr_array_dim3_block_received,
                                                        0_UZ,
                                                        number_neurons_received - 1, // Subtract bias.
                                                        ptr_array_accuracy_values_received + blockIdx.x,
                                                        ptr_array_loss_values_received + blockIdx.x,
                                                        accurancy_variance_received,
                                                        ptr_array_output_layer_outputs_received + tmp_thread_global_index * number_neurons_received,
                                                        ptr_array_desired_outputs_received[tmp_thread_global_index])
}

template<typename T>
__global__ void kernel__cuModel__FF__Test__Binary_Cross_Entropy__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                      size_t const number_neurons_received,
                                                                                                                                                                      float *const ptr_array_accuracy_values_received,
                                                                                                                                                                      float *const ptr_array_loss_values_received,
                                                                                                                                                                      float const accurancy_variance_received,
                                                                                                                                                                      T const *const ptr_array_output_layer_outputs_received,
                                                                                                                                                                      T **const ptr_array_desired_outputs_received,
                                                                                                                                                                      struct dim3 const *const ptr_array_dim3_grid_received,
                                                                                                                                                                      struct dim3 const *const ptr_array_dim3_block_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(cuModel__FF__Test__Binary_Cross_Entropy__Serialize_Batch__Parallel_Neurons<T>,
                                                          ptr_array_dim3_grid_received,
                                                          ptr_array_dim3_block_received,
                                                          0_UZ,
                                                          number_neurons_received - 1, // Subtract bias.
                                                          ptr_array_accuracy_values_received + blockIdx.x,
                                                          ptr_array_loss_values_received + blockIdx.x,
                                                          accurancy_variance_received,
                                                          ptr_array_output_layer_outputs_received + tmp_thread_global_index * number_neurons_received,
                                                          ptr_array_desired_outputs_received[tmp_thread_global_index])
    }
}

template<typename T>
__global__ void kernel_while__cuModel__FF__Test__Binary_Cross_Entropy__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                float *const ptr_array_accuracy_values_received,
                                                                                                                                                                                float *const ptr_array_loss_values_received,
                                                                                                                                                                                float const accurancy_variance_received,
                                                                                                                                                                                T const *const ptr_array_output_layer_outputs_received,
                                                                                                                                                                                T **const ptr_array_desired_outputs_received,
                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_grid_received,
                                                                                                                                                                                struct dim3 const *const ptr_array_dim3_block_received)
{
    size_t const tmp_grid_stride(gridDim.x * blockDim.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    // KERNEL LAUNCH
    //    1: Launching do-while elements.
    if(ptr_array_dim3_grid_received->x * ptr_array_dim3_block_received->x < number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel_while__cuModel__FF__Test__Binary_Cross_Entropy__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_array_dim3_grid_received, *ptr_array_dim3_block_received >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                       ptr_array_accuracy_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                                       ptr_array_loss_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                                       accurancy_variance_received,
                                                                                                                                                                                                                                                                                       ptr_array_output_layer_outputs_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                       ptr_array_desired_outputs_received[tmp_thread_global_index]);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    //    2: Launching size condition.
    else if(ptr_array_dim3_grid_received->x * ptr_array_dim3_block_received->x > number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel__cuModel__FF__Test__Binary_Cross_Entropy__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_array_dim3_grid_received, *ptr_array_dim3_block_received >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                              ptr_array_accuracy_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                              ptr_array_loss_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                              accurancy_variance_received,
                                                                                                                                                                                                                                                                              ptr_array_output_layer_outputs_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                              ptr_array_desired_outputs_received[tmp_thread_global_index]);
        
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
            kernel__cuModel__FF__Test__Binary_Cross_Entropy__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_array_dim3_grid_received, *ptr_array_dim3_block_received >>> (ptr_array_accuracy_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                              ptr_array_loss_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                              accurancy_variance_received,
                                                                                                                                                                                                                                                                              ptr_array_output_layer_outputs_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                              ptr_array_desired_outputs_received[tmp_thread_global_index]);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

template<typename T>
__global__ void kernel__cuModel__FF__Test__Binary_Cross_Entropy__Parallel_Batch__Serialize_Neurons(size_t const number_neurons_received,
                                                                                                                                                                        float *const ptr_array_accuracy_values_received,
                                                                                                                                                                        float *const ptr_array_loss_values_received,
                                                                                                                                                                        float const accurancy_variance_received,
                                                                                                                                                                        T const *const ptr_array_output_layer_outputs_received,
                                                                                                                                                                        T **const ptr_array_desired_outputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *tmp_ptr_array_desireds_outputs(ptr_array_desired_outputs_received[tmp_thread_global_index]),
                 *tmp_ptr_array_output_layer_outputs(ptr_array_output_layer_outputs_received + tmp_thread_global_index * number_neurons_received),
                 *const tmp_ptr_array_output_layer_outputs_end(tmp_ptr_array_output_layer_outputs + number_neurons_received - 1u); // Subtract bias.
    T tmp_output_value,
       tmp_desired_value,
       tmp_difference;
    
    for(; tmp_ptr_array_output_layer_outputs != tmp_ptr_array_output_layer_outputs_end; ++tmp_ptr_array_output_layer_outputs,
                                                                                                                                 ++tmp_ptr_array_desireds_outputs)
    {
        tmp_output_value = *tmp_ptr_array_output_layer_outputs;

        tmp_desired_value = *tmp_ptr_array_desireds_outputs;

        tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
        
        Update_Error__Binary_Cross_Entropy(tmp_output_value,
                                                                tmp_desired_value,
                                                                ptr_array_loss_values_received + tmp_thread_global_index);
        
        Update_Accuracy(tmp_difference,
                                  accurancy_variance_received,
                                  ptr_array_accuracy_values_received + tmp_thread_global_index);
    }
}

template<typename T>
__global__ void kernel__cuModel__FF__Test__Binary_Cross_Entropy__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                        float *const ptr_array_accuracy_values_received,
                                                                                                                                                                        float *const ptr_array_loss_values_received,
                                                                                                                                                                        float const accurancy_variance_received,
                                                                                                                                                                        T const *const ptr_array_output_layer_outputs_received,
                                                                                                                                                                        T **const ptr_array_desired_outputs_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *tmp_ptr_array_output_layer_outputs,
                 *tmp_ptr_array_output_layer_outputs_end,
                 *tmp_ptr_array_desireds_outputs;
    T tmp_output_value,
       tmp_desired_value,
       tmp_difference;
    
    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_desireds_outputs = ptr_array_desired_outputs_received[tmp_thread_global_index];
        tmp_ptr_array_output_layer_outputs = ptr_array_output_layer_outputs_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_output_layer_outputs_end = tmp_ptr_array_output_layer_outputs + number_neurons_received - 1_UZ; // Subtract bias.

        for(; tmp_ptr_array_output_layer_outputs != tmp_ptr_array_output_layer_outputs_end; ++tmp_ptr_array_output_layer_outputs,
                                                                                                                                    ++tmp_ptr_array_desireds_outputs)
        {
            tmp_output_value = *tmp_ptr_array_output_layer_outputs;

            tmp_desired_value = *tmp_ptr_array_desireds_outputs;

            tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
            
            Update_Error__Binary_Cross_Entropy(tmp_output_value,
                                                                    tmp_desired_value,
                                                                    ptr_array_loss_values_received + tmp_thread_global_index);
            
            Update_Accuracy(tmp_difference,
                                      accurancy_variance_received,
                                      ptr_array_accuracy_values_received + tmp_thread_global_index);
        }
    }
}

template<typename T>
__global__ void kernel_while__cuModel__FF__Test__Binary_Cross_Entropy__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                                size_t const number_neurons_received,
                                                                                                                                                                                float *const ptr_array_accuracy_values_received,
                                                                                                                                                                                float *const ptr_array_loss_values_received,
                                                                                                                                                                                float const accurancy_variance_received,
                                                                                                                                                                                T const *const ptr_array_output_layer_outputs_received,
                                                                                                                                                                                T **const ptr_array_desired_outputs_received)
{
    size_t const tmp_thread_grid_index(blockIdx.x * blockDim.x + threadIdx.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *tmp_ptr_array_output_layer_outputs,
                 *tmp_ptr_array_output_layer_outputs_end,
                 *tmp_ptr_array_desireds_outputs;
    T tmp_output_value,
       tmp_desired_value,
       tmp_difference;
    
    // Loop through each sample.
    do
    {
        tmp_ptr_array_desireds_outputs = ptr_array_desired_outputs_received[tmp_thread_global_index];
        tmp_ptr_array_output_layer_outputs = ptr_array_output_layer_outputs_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_output_layer_outputs_end = tmp_ptr_array_output_layer_outputs + number_neurons_received - 1_UZ; // Subtract bias.

        for(; tmp_ptr_array_output_layer_outputs != tmp_ptr_array_output_layer_outputs_end; ++tmp_ptr_array_output_layer_outputs,
                                                                                                                                    ++tmp_ptr_array_desireds_outputs)
        {
            tmp_output_value = *tmp_ptr_array_output_layer_outputs;

            tmp_desired_value = *tmp_ptr_array_desireds_outputs;

            tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
            
            Update_Error__Binary_Cross_Entropy(tmp_output_value,
                                                                    tmp_desired_value,
                                                                    ptr_array_loss_values_received + tmp_thread_grid_index);
            
            Update_Accuracy(tmp_difference,
                                      accurancy_variance_received,
                                      ptr_array_accuracy_values_received + tmp_thread_grid_index);
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::FF__Test__Binary_Cross_Entropy(size_t const batch_size, var **const ptr_array_outputs_received)
{
    size_t tmp_data_index;

    struct cuLayer *const tmp_ptr_output_layer(this->ptr_last_layer - 1);
    
    struct cuNeuron const *const tmp_ptr_output_layer_first_neuron(tmp_ptr_output_layer->ptr_array_neuron_units);
    
    size_t const tmp_number_neuron_units(*tmp_ptr_output_layer->ptr_number_neurons);
    
    // Variable to cache optimal size to launch dynamic parallelisme through the GPU.
    struct dim3 tmp_dim3_grid,
                     tmp_dim3_block;

    // Condition to enter into dynamic parallelisme of each sample.
    if(USE_PARALLEL && batch_size >= warpSize)
    {
        size_t const tmp_batch_size_scale(std::min<size_t>(batch_size, this->number_threads));

        if(tmp_batch_size_scale == this->number_threads)
        {
            tmp_dim3_grid = this->ptr_array_dim3_grid[0];
            tmp_dim3_block = this->ptr_array_dim3_block[0];
        }
        else
        {
            this->ptr_array_layers->ptr_Class_Storage_Dim3_Batch->Get__Dim3_1D(tmp_batch_size_scale,
                                                                                                                    tmp_dim3_grid,
                                                                                                                    tmp_dim3_block,
                                                                                                                    this->Get__Class_Device_Information_Array()->Get__CUDA_Device());
        }
        
        // Condition to enter into dynamic parallelisme of each sample and neurons.
        if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
        {
            LAUNCH_KERNEL_1D(cuModel__FF__Test__Binary_Cross_Entropy__Parallel_Batch__Parallel_Neurons<var>,
                                              tmp_dim3_grid,
                                              tmp_dim3_block,
                                              0_UZ,
                                              batch_size,
                                              tmp_number_neuron_units,
                                              this->ptr_array_accuracy_values[0],
                                              this->ptr_array_loss_values,
                                              this->acc_var,
                                              tmp_ptr_output_layer_first_neuron->ptr_array_values,
                                              ptr_array_outputs_received,
                                              tmp_ptr_output_layer->ptr_dim3_grid_neurons,
                                              tmp_ptr_output_layer->ptr_dim3_block_neurons)
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            LAUNCH_KERNEL_1D(cuModel__FF__Test__Binary_Cross_Entropy__Parallel_Batch__Serialize_Neurons<var>,
                                            tmp_dim3_grid,
                                            tmp_dim3_block,
                                            0_UZ,
                                            batch_size,
                                            tmp_number_neuron_units,
                                            this->ptr_array_accuracy_values[0],
                                            this->ptr_array_loss_values,
                                            this->acc_var,
                                            tmp_ptr_output_layer_first_neuron->ptr_array_values,
                                            ptr_array_outputs_received)
        }
    }
    // Condition to enter into dynamic parallelisme of each neurons.
    else if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
    {
        // KERNEL LAUNCH
        //    1: Launching do-while elements.
        if(tmp_ptr_output_layer->ptr_dim3_grid_neurons_DP->x * tmp_ptr_output_layer->ptr_dim3_block_neurons_DP->x < tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel_while__cuModel__FF__Test__Binary_Cross_Entropy__Serialize_Batch__Parallel_Neurons<var> <<< *tmp_ptr_output_layer->ptr_dim3_grid_neurons_DP, *tmp_ptr_output_layer->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                          this->ptr_array_accuracy_values[0],
                                                                                                                                                                                                                                                                                                                                                          this->ptr_array_loss_values,
                                                                                                                                                                                                                                                                                                                                                          this->acc_var,
                                                                                                                                                                                                                                                                                                                                                          tmp_ptr_output_layer_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                          ptr_array_outputs_received[tmp_data_index]);
            }
        }
        //    2: Launching size condition.
        else if(tmp_ptr_output_layer->ptr_dim3_grid_neurons_DP->x * tmp_ptr_output_layer->ptr_dim3_block_neurons_DP->x > tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__cuModel__FF__Test__Binary_Cross_Entropy__Serialize_Batch__Parallel_Neurons<var> <<< *tmp_ptr_output_layer->ptr_dim3_grid_neurons_DP, *tmp_ptr_output_layer->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                 this->ptr_array_accuracy_values[0],
                                                                                                                                                                                                                                                                                                                                                 this->ptr_array_loss_values,
                                                                                                                                                                                                                                                                                                                                                 this->acc_var,
                                                                                                                                                                                                                                                                                                                                                 tmp_ptr_output_layer_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                 ptr_array_outputs_received[tmp_data_index]);
            }
        }
        //    3: Standard.
        else
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__cuModel__FF__Test__Binary_Cross_Entropy__Serialize_Batch__Parallel_Neurons<var> <<< *tmp_ptr_output_layer->ptr_dim3_grid_neurons_DP, *tmp_ptr_output_layer->ptr_dim3_block_neurons_DP >>> (this->ptr_array_accuracy_values[0],
                                                                                                                                                                                                                                                                                                                                                 this->ptr_array_loss_values,
                                                                                                                                                                                                                                                                                                                                                 this->acc_var,
                                                                                                                                                                                                                                                                                                                                                 tmp_ptr_output_layer_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                 ptr_array_outputs_received[tmp_data_index]);
            }
        }
        // |END| KERNEL LAUNCH |END|
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        var const *tmp_ptr_array_desireds_outputs,
                      *tmp_ptr_array_values,
                      *tmp_ptr_array_values_end;
        var tmp_output_value,
            tmp_desired_value,
            tmp_difference;
        
        // Loop through each sample.
        for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
        {
            tmp_ptr_array_desireds_outputs = ptr_array_outputs_received[tmp_data_index];
            tmp_ptr_array_values = tmp_ptr_output_layer_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units;
            tmp_ptr_array_values_end = tmp_ptr_array_values + tmp_number_neuron_units - 1_UZ; // Subtract bias.

            for(; tmp_ptr_array_values != tmp_ptr_array_values_end; ++tmp_ptr_array_values,
                                                                                              ++tmp_ptr_array_desireds_outputs)
            {
                tmp_output_value = *tmp_ptr_array_values;

                tmp_desired_value = *tmp_ptr_array_desireds_outputs;

                tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
                
                Update_Error__Binary_Cross_Entropy(tmp_output_value,
                                                                        tmp_desired_value,
                                                                        this->ptr_array_loss_values);
                
                Update_Accuracy(tmp_difference,
                                          this->acc_var,
                                          this->ptr_array_accuracy_values[0]);
            }
        }
    }
}
