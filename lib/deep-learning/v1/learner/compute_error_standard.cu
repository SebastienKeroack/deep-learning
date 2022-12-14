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
__global__ void kernel__cuModel__FF__Compute__Error__Standard__Serialize_Batch__Parallel_Neurons(float *const ptr_array_accuracy_values_received,
                                                                                                                                                                    float *const ptr_array_loss_values_received,
                                                                                                                                                                    float const accurancy_variance_received,
                                                                                                                                                                    T const *const ptr_array_output_layer_summations_received,
                                                                                                                                                                    T const *const ptr_array_output_layer_values_received,
                                                                                                                                                                    T *const ptr_array_output_layer_errors_received,
                                                                                                                                                                    T const *const ptr_array_desired_outputs_received,
                                                                                                                                                                    DL::LOSS_FN::TYPE const type_loss_function_received,
                                                                                                                                                                    enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *ptr_array_output_layer_type_activations_functions_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const tmp_output_value(ptr_array_output_layer_values_received[tmp_thread_global_index]),
                tmp_desired_value(ptr_array_desired_outputs_received[tmp_thread_global_index]),
                tmp_difference(tmp_output_value - tmp_desired_value);
    
    update_loss(tmp_output_value,
                        tmp_desired_value,
                        tmp_difference,
                        ptr_array_loss_values_received,
                        type_loss_function_received);
        
    Update_Accuracy(tmp_difference,
                            accurancy_variance_received,
                            ptr_array_accuracy_values_received);

    ptr_array_output_layer_errors_received[tmp_thread_global_index] = Activation_Derived(T(1),
                                                                                                                                    ptr_array_output_layer_summations_received[tmp_thread_global_index],
                                                                                                                                    tmp_output_value,
                                                                                                                                    ptr_array_output_layer_type_activations_functions_received[tmp_thread_global_index],
                                                                                                                                    type_loss_function_received) * tmp_difference;
}

template<typename T>
__global__ void kernel__cuModel__FF__Compute__Error__Standard__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                    float *const ptr_array_accuracy_values_received,
                                                                                                                                                                    float *const ptr_array_loss_values_received,
                                                                                                                                                                    float const accurancy_variance_received,
                                                                                                                                                                    T const *const ptr_array_output_layer_summations_received,
                                                                                                                                                                    T const *const ptr_array_output_layer_values_received,
                                                                                                                                                                    T *const ptr_array_output_layer_errors_received,
                                                                                                                                                                    T const *const ptr_array_desired_outputs_received,
                                                                                                                                                                    DL::LOSS_FN::TYPE const type_loss_function_received,
                                                                                                                                                                    enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *ptr_array_output_layer_type_activations_functions_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_output_value,
       tmp_desired_value,
       tmp_difference;
    
    if(tmp_thread_global_index < size_received)
    {
        tmp_output_value = ptr_array_output_layer_values_received[tmp_thread_global_index];

        tmp_desired_value = ptr_array_desired_outputs_received[tmp_thread_global_index];

        tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
        
        update_loss(tmp_output_value,
                            tmp_desired_value,
                            tmp_difference,
                            ptr_array_loss_values_received,
                            type_loss_function_received);
        
        Update_Accuracy(tmp_difference,
                                accurancy_variance_received,
                                ptr_array_accuracy_values_received);

        ptr_array_output_layer_errors_received[tmp_thread_global_index] = Activation_Derived(T(1),
                                                                                                                                     ptr_array_output_layer_summations_received[tmp_thread_global_index],
                                                                                                                                     tmp_output_value,
                                                                                                                                     ptr_array_output_layer_type_activations_functions_received[tmp_thread_global_index],
                                                                                                                                     type_loss_function_received) * tmp_difference;
    }
}

template<typename T>
__global__ void kernel_while__cuModel__FF__Compute__Error__Standard__Serialize_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                            float *const ptr_array_accuracy_values_received,
                                                                                                                                                                            float *const ptr_array_loss_values_received,
                                                                                                                                                                            float const accurancy_variance_received,
                                                                                                                                                                            T const *const ptr_array_output_layer_summations_received,
                                                                                                                                                                            T const *const ptr_array_output_layer_values_received,
                                                                                                                                                                            T *const ptr_array_output_layer_errors_received,
                                                                                                                                                                            T const *const ptr_array_desired_outputs_received,
                                                                                                                                                                            DL::LOSS_FN::TYPE const type_loss_function_received,
                                                                                                                                                                            enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *ptr_array_output_layer_type_activations_functions_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T tmp_output_value,
       tmp_desired_value,
       tmp_difference;
    
    do
    {
        tmp_output_value = ptr_array_output_layer_values_received[tmp_thread_global_index];

        tmp_desired_value = ptr_array_desired_outputs_received[tmp_thread_global_index];

        tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
        
        update_loss(tmp_output_value,
                            tmp_desired_value,
                            tmp_difference,
                            ptr_array_loss_values_received,
                            type_loss_function_received);
        
        Update_Accuracy(tmp_difference,
                                accurancy_variance_received,
                                ptr_array_accuracy_values_received);

        ptr_array_output_layer_errors_received[tmp_thread_global_index] = Activation_Derived(T(1),
                                                                                                                                     ptr_array_output_layer_summations_received[tmp_thread_global_index],
                                                                                                                                     tmp_output_value,
                                                                                                                                     ptr_array_output_layer_type_activations_functions_received[tmp_thread_global_index],
                                                                                                                                     type_loss_function_received) * tmp_difference;

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__cuModel__FF__Compute__Error__Standard__Parallel_Batch__Parallel_Neurons(size_t const number_neurons_received,
                                                                                                                                                                  float *const ptr_array_accuracy_values_received,
                                                                                                                                                                  float *const ptr_array_loss_values_received,
                                                                                                                                                                  float const accurancy_variance_received,
                                                                                                                                                                  T const *const ptr_array_output_layer_summations_received,
                                                                                                                                                                  T const *const ptr_array_output_layer_values_received,
                                                                                                                                                                  T *const ptr_array_output_layer_errors_received,
                                                                                                                                                                  T **const ptr_array_desired_outputs_received,
                                                                                                                                                                  DL::LOSS_FN::TYPE const type_loss_function_received,
                                                                                                                                                                  enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_output_layer_type_activations_functions_received,
                                                                                                                                                                  struct dim3 const *const ptr_array_dim3_grid_received,
                                                                                                                                                                  struct dim3 const *const ptr_array_dim3_block_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    LAUNCH_KERNEL_POINTER_1D(cuModel__FF__Compute__Error__Standard__Serialize_Batch__Parallel_Neurons<T>,
                                                        ptr_array_dim3_grid_received,
                                                        ptr_array_dim3_block_received,
                                                        0_UZ,
                                                        number_neurons_received - 1, // Subtract bias.
                                                        ptr_array_accuracy_values_received + blockIdx.x,
                                                        ptr_array_loss_values_received + blockIdx.x,
                                                        accurancy_variance_received,
                                                        ptr_array_output_layer_summations_received + tmp_thread_global_index * number_neurons_received,
                                                        ptr_array_output_layer_values_received + tmp_thread_global_index * number_neurons_received,
                                                        ptr_array_output_layer_errors_received + tmp_thread_global_index * number_neurons_received,
                                                        ptr_array_desired_outputs_received[tmp_thread_global_index],
                                                        type_loss_function_received,
                                                        ptr_array_output_layer_type_activations_functions_received)
}

template<typename T>
__global__ void kernel__cuModel__FF__Compute__Error__Standard__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                  size_t const number_neurons_received,
                                                                                                                                                                  float *const ptr_array_accuracy_values_received,
                                                                                                                                                                  float *const ptr_array_loss_values_received,
                                                                                                                                                                  float const accurancy_variance_received,
                                                                                                                                                                  T const *const ptr_array_output_layer_summations_received,
                                                                                                                                                                  T const *const ptr_array_output_layer_values_received,
                                                                                                                                                                  T *const ptr_array_output_layer_errors_received,
                                                                                                                                                                  T **const ptr_array_desired_outputs_received,
                                                                                                                                                                  DL::LOSS_FN::TYPE const type_loss_function_received,
                                                                                                                                                                  enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_output_layer_type_activations_functions_received,
                                                                                                                                                                  struct dim3 const *const ptr_array_dim3_grid_received,
                                                                                                                                                                  struct dim3 const *const ptr_array_dim3_block_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(cuModel__FF__Compute__Error__Standard__Serialize_Batch__Parallel_Neurons<T>,
                                                          ptr_array_dim3_grid_received,
                                                          ptr_array_dim3_block_received,
                                                          0_UZ,
                                                          number_neurons_received - 1, // Subtract bias.
                                                          ptr_array_accuracy_values_received + blockIdx.x,
                                                          ptr_array_loss_values_received + blockIdx.x,
                                                          accurancy_variance_received,
                                                          ptr_array_output_layer_summations_received + tmp_thread_global_index * number_neurons_received,
                                                          ptr_array_output_layer_values_received + tmp_thread_global_index * number_neurons_received,
                                                          ptr_array_output_layer_errors_received + tmp_thread_global_index * number_neurons_received,
                                                          ptr_array_desired_outputs_received[tmp_thread_global_index],
                                                          type_loss_function_received,
                                                          ptr_array_output_layer_type_activations_functions_received)
    }
}

template<typename T>
__global__ void kernel_while__cuModel__FF__Compute__Error__Standard__Parallel_Batch__Parallel_Neurons(size_t const size_received,
                                                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                                                            float *const ptr_array_accuracy_values_received,
                                                                                                                                                                            float *const ptr_array_loss_values_received,
                                                                                                                                                                            float const accurancy_variance_received,
                                                                                                                                                                            T const *const ptr_array_output_layer_summations_received,
                                                                                                                                                                            T const *const ptr_array_output_layer_values_received,
                                                                                                                                                                            T *const ptr_array_output_layer_errors_received,
                                                                                                                                                                            T **const ptr_array_desired_outputs_received,
                                                                                                                                                                            DL::LOSS_FN::TYPE const type_loss_function_received,
                                                                                                                                                                            enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_output_layer_type_activations_functions_received,
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
            kernel_while__cuModel__FF__Compute__Error__Standard__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_array_dim3_grid_received, *ptr_array_dim3_block_received >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                   ptr_array_accuracy_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                                   ptr_array_loss_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                                   accurancy_variance_received,
                                                                                                                                                                                                                                                                                   ptr_array_output_layer_summations_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                   ptr_array_output_layer_values_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                   ptr_array_output_layer_errors_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                   ptr_array_desired_outputs_received[tmp_thread_global_index],
                                                                                                                                                                                                                                                                                   type_loss_function_received,
                                                                                                                                                                                                                                                                                   ptr_array_output_layer_type_activations_functions_received);
        
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
            kernel__cuModel__FF__Compute__Error__Standard__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_array_dim3_grid_received, *ptr_array_dim3_block_received >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                          ptr_array_accuracy_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                          ptr_array_loss_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                          accurancy_variance_received,
                                                                                                                                                                                                                                                                          ptr_array_output_layer_summations_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                          ptr_array_output_layer_values_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                          ptr_array_output_layer_errors_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                          ptr_array_desired_outputs_received[tmp_thread_global_index],
                                                                                                                                                                                                                                                                          type_loss_function_received,
                                                                                                                                                                                                                                                                          ptr_array_output_layer_type_activations_functions_received);
        
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
            kernel__cuModel__FF__Compute__Error__Standard__Serialize_Batch__Parallel_Neurons<T> <<< *ptr_array_dim3_grid_received, *ptr_array_dim3_block_received >>> (ptr_array_accuracy_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                            ptr_array_loss_values_received + blockIdx.x,
                                                                                                                                                                                                                                                                            accurancy_variance_received,
                                                                                                                                                                                                                                                                            ptr_array_output_layer_summations_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                            ptr_array_output_layer_values_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                            ptr_array_output_layer_errors_received + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                            ptr_array_desired_outputs_received[tmp_thread_global_index],
                                                                                                                                                                                                                                                                            type_loss_function_received,
                                                                                                                                                                                                                                                                            ptr_array_output_layer_type_activations_functions_received);
        
            tmp_thread_global_index += tmp_grid_stride;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

template<typename T>
__global__ void kernel__cuModel__FF__Compute__Error__Standard__Parallel_Batch__Serialize_Neurons(size_t const number_neurons_received,
                                                                                                                                                                    float *const ptr_array_accuracy_values_received,
                                                                                                                                                                    float *const ptr_array_loss_values_received,
                                                                                                                                                                    float const accurancy_variance_received,
                                                                                                                                                                    T const *const ptr_array_output_layer_summatins_received,
                                                                                                                                                                    T const *const ptr_array_output_layer_values_received,
                                                                                                                                                                    T *const ptr_array_output_layer_errors_received,
                                                                                                                                                                    T **const ptr_array_desired_outputs_received,
                                                                                                                                                                    DL::LOSS_FN::TYPE const type_loss_function_received,
                                                                                                                                                                    enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *ptr_array_output_layer_type_activations_functions_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *tmp_ptr_array_desireds_outputs(ptr_array_desired_outputs_received[tmp_thread_global_index]),
                 *tmp_ptr_array_output_layer_summations(ptr_array_output_layer_summatins_received + tmp_thread_global_index * number_neurons_received),
                 *tmp_ptr_array_output_layer_values(ptr_array_output_layer_values_received + tmp_thread_global_index * number_neurons_received);
    T tmp_output_value,
       tmp_desired_value,
       tmp_difference,
       *tmp_ptr_array_output_layer_errors(ptr_array_output_layer_errors_received + tmp_thread_global_index * number_neurons_received);
    T const *const tmp_ptr_array_output_layer_errors_end(tmp_ptr_array_output_layer_errors + number_neurons_received - 1u); // Subtract bias.

    for(; tmp_ptr_array_output_layer_errors != tmp_ptr_array_output_layer_errors_end; ++tmp_ptr_array_output_layer_errors,
                                                                                                                            ++tmp_ptr_array_output_layer_summations,
                                                                                                                            ++tmp_ptr_array_output_layer_values,
                                                                                                                            ++ptr_array_output_layer_type_activations_functions_received,
                                                                                                                            ++tmp_ptr_array_desireds_outputs)
    {
        tmp_output_value = *tmp_ptr_array_output_layer_values;

        tmp_desired_value = *tmp_ptr_array_desireds_outputs;

        tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
            
        update_loss(tmp_output_value,
                            tmp_desired_value,
                            tmp_difference,
                            ptr_array_loss_values_received + tmp_thread_global_index,
                            type_loss_function_received);
            
        Update_Accuracy(tmp_difference,
                                  accurancy_variance_received,
                                  ptr_array_accuracy_values_received + tmp_thread_global_index);

        *tmp_ptr_array_output_layer_errors = Activation_Derived(T(1),
                                                                                          *tmp_ptr_array_output_layer_summations,
                                                                                          tmp_output_value,
                                                                                          *ptr_array_output_layer_type_activations_functions_received,
                                                                                          type_loss_function_received) * tmp_difference;
    }
}

template<typename T>
__global__ void kernel__cuModel__FF__Compute__Error__Standard__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                    float *const ptr_array_accuracy_values_received,
                                                                                                                                                                    float *const ptr_array_loss_values_received,
                                                                                                                                                                    float const accurancy_variance_received,
                                                                                                                                                                    T const *const ptr_array_output_layer_summatins_received,
                                                                                                                                                                    T const *const ptr_array_output_layer_values_received,
                                                                                                                                                                    T *const ptr_array_output_layer_errors_received,
                                                                                                                                                                    T **const ptr_array_desired_outputs_received,
                                                                                                                                                                    DL::LOSS_FN::TYPE const type_loss_function_received,
                                                                                                                                                                    enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *ptr_array_output_layer_type_activations_functions_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *tmp_ptr_array_output_layer_summations,
                 *tmp_ptr_array_output_layer_values,
                 *tmp_ptr_array_output_layer_errors_end,
                 *tmp_ptr_array_desireds_outputs;
    T *tmp_ptr_array_output_layer_errors,
       tmp_output_value,
       tmp_desired_value,
       tmp_difference;
    
    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_desireds_outputs = ptr_array_desired_outputs_received[tmp_thread_global_index];
        tmp_ptr_array_output_layer_summations = ptr_array_output_layer_summatins_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_output_layer_values = ptr_array_output_layer_values_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_output_layer_errors = ptr_array_output_layer_errors_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_output_layer_errors_end = tmp_ptr_array_output_layer_errors + number_neurons_received - 1_UZ; // Subtract bias.
        
        for(; tmp_ptr_array_output_layer_errors != tmp_ptr_array_output_layer_errors_end; ++tmp_ptr_array_output_layer_errors,
                                                                                                                                ++tmp_ptr_array_output_layer_summations,
                                                                                                                                ++tmp_ptr_array_output_layer_values,
                                                                                                                                ++ptr_array_output_layer_type_activations_functions_received,
                                                                                                                                ++tmp_ptr_array_desireds_outputs)
        {
            tmp_output_value = *tmp_ptr_array_output_layer_values;

            tmp_desired_value = *tmp_ptr_array_desireds_outputs;

            tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
            
            update_loss(tmp_output_value,
                                tmp_desired_value,
                                tmp_difference,
                                ptr_array_loss_values_received + tmp_thread_global_index,
                                type_loss_function_received);
            
            Update_Accuracy(tmp_difference,
                                      accurancy_variance_received,
                                      ptr_array_accuracy_values_received + tmp_thread_global_index);

            *tmp_ptr_array_output_layer_errors = Activation_Derived(T(1),
                                                                                              *tmp_ptr_array_output_layer_summations,
                                                                                              tmp_output_value,
                                                                                              *ptr_array_output_layer_type_activations_functions_received,
                                                                                              type_loss_function_received) * tmp_difference;
        }
    }
}

template<typename T>
__global__ void kernel_while__cuModel__FF__Compute__Error__Standard__Parallel_Batch__Serialize_Neurons(size_t const size_received,
                                                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                                                            float *const ptr_array_accuracy_values_received,
                                                                                                                                                                            float *const ptr_array_loss_values_received,
                                                                                                                                                                            float const accurancy_variance_received,
                                                                                                                                                                            T const *const ptr_array_output_layer_summatins_received,
                                                                                                                                                                            T const *const ptr_array_output_layer_values_received,
                                                                                                                                                                            T *const ptr_array_output_layer_errors_received,
                                                                                                                                                                            T **const ptr_array_desired_outputs_received,
                                                                                                                                                                            DL::LOSS_FN::TYPE const type_loss_function_received,
                                                                                                                                                                            enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_output_layer_type_activations_functions_received)
{
    size_t const tmp_thread_grid_index(blockIdx.x * blockDim.x + threadIdx.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    T const *tmp_ptr_array_output_layer_summations,
                 *tmp_ptr_array_output_layer_values,
                 *tmp_ptr_array_output_layer_errors_end,
                 *tmp_ptr_array_desireds_outputs;
    T *tmp_ptr_array_output_layer_errors,
       tmp_output_value,
       tmp_desired_value,
       tmp_difference;
    
    enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *tmp_ptr_array_output_layer_type_activations_functions;

    // Loop through each sample.
    do
    {
        tmp_ptr_array_desireds_outputs = ptr_array_desired_outputs_received[tmp_thread_global_index];
        tmp_ptr_array_output_layer_summations = ptr_array_output_layer_summatins_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_output_layer_values = ptr_array_output_layer_values_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_output_layer_errors = ptr_array_output_layer_errors_received + tmp_thread_global_index * number_neurons_received;
        tmp_ptr_array_output_layer_errors_end = tmp_ptr_array_output_layer_errors + number_neurons_received - 1_UZ; // Subtract bias.
        
        tmp_ptr_array_output_layer_type_activations_functions = ptr_array_output_layer_type_activations_functions_received;

        for(; tmp_ptr_array_output_layer_errors != tmp_ptr_array_output_layer_errors_end; ++tmp_ptr_array_output_layer_errors,
                                                                                                                                ++tmp_ptr_array_output_layer_summations,
                                                                                                                                ++tmp_ptr_array_output_layer_values,
                                                                                                                                ++tmp_ptr_array_output_layer_type_activations_functions,
                                                                                                                                ++tmp_ptr_array_desireds_outputs)
        {
            tmp_output_value = *tmp_ptr_array_output_layer_values;

            tmp_desired_value = *tmp_ptr_array_desireds_outputs;

            tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
            
            update_loss(tmp_output_value,
                                tmp_desired_value,
                                tmp_difference,
                                ptr_array_loss_values_received + tmp_thread_grid_index,
                                type_loss_function_received);
            
            Update_Accuracy(tmp_difference,
                                      accurancy_variance_received,
                                      ptr_array_accuracy_values_received + tmp_thread_grid_index);

            *tmp_ptr_array_output_layer_errors = Activation_Derived(T(1),
                                                                                              *tmp_ptr_array_output_layer_summations,
                                                                                              tmp_output_value,
                                                                                              *tmp_ptr_array_output_layer_type_activations_functions,
                                                                                              type_loss_function_received) * tmp_difference;
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::FF__Compute__Error__Standard(size_t const batch_size, var **const ptr_array_outputs_received)
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
            LAUNCH_KERNEL_1D(cuModel__FF__Compute__Error__Standard__Parallel_Batch__Parallel_Neurons<var>,
                                              tmp_dim3_grid,
                                              tmp_dim3_block,
                                              0_UZ,
                                              batch_size,
                                              tmp_number_neuron_units,
                                              this->ptr_array_accuracy_values[0],
                                              this->ptr_array_loss_values,
                                              this->acc_var,
                                              tmp_ptr_output_layer_first_neuron->ptr_array_summations,
                                              tmp_ptr_output_layer_first_neuron->ptr_array_values,
                                              tmp_ptr_output_layer_first_neuron->ptr_array_errors,
                                              ptr_array_outputs_received,
                                              this->type_loss_function,
                                              tmp_ptr_output_layer_first_neuron->ptr_type_activation_function,
                                              tmp_ptr_output_layer->ptr_dim3_grid_neurons,
                                              tmp_ptr_output_layer->ptr_dim3_block_neurons)
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            LAUNCH_KERNEL_1D(cuModel__FF__Compute__Error__Standard__Parallel_Batch__Serialize_Neurons<var>,
                                               tmp_dim3_grid,
                                               tmp_dim3_block,
                                               0_UZ,
                                               batch_size,
                                               tmp_number_neuron_units,
                                               this->ptr_array_accuracy_values[0],
                                               this->ptr_array_loss_values,
                                               this->acc_var,
                                               tmp_ptr_output_layer_first_neuron->ptr_array_summations,
                                               tmp_ptr_output_layer_first_neuron->ptr_array_values,
                                               tmp_ptr_output_layer_first_neuron->ptr_array_errors,
                                               ptr_array_outputs_received,
                                               this->type_loss_function,
                                               tmp_ptr_output_layer_first_neuron->ptr_type_activation_function)
        }

        // Synchronize before using the computed derivatives values.
        CUDA__Check_Error();
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
                kernel_while__cuModel__FF__Compute__Error__Standard__Serialize_Batch__Parallel_Neurons<var> <<< *tmp_ptr_output_layer->ptr_dim3_grid_neurons_DP, *tmp_ptr_output_layer->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                      this->ptr_array_accuracy_values[0],
                                                                                                                                                                                                                                                                                                                                                      this->ptr_array_loss_values,
                                                                                                                                                                                                                                                                                                                                                      this->acc_var,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_output_layer_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_output_layer_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_output_layer_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                      ptr_array_outputs_received[tmp_data_index],
                                                                                                                                                                                                                                                                                                                                                      this->type_loss_function,
                                                                                                                                                                                                                                                                                                                                                      tmp_ptr_output_layer_first_neuron->ptr_type_activation_function);
            }
        }
        //    2: Launching size condition.
        else if(tmp_ptr_output_layer->ptr_dim3_grid_neurons_DP->x * tmp_ptr_output_layer->ptr_dim3_block_neurons_DP->x > tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__cuModel__FF__Compute__Error__Standard__Serialize_Batch__Parallel_Neurons<var> <<< *tmp_ptr_output_layer->ptr_dim3_grid_neurons_DP, *tmp_ptr_output_layer->ptr_dim3_block_neurons_DP >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                             this->ptr_array_accuracy_values[0],
                                                                                                                                                                                                                                                                                                                                             this->ptr_array_loss_values,
                                                                                                                                                                                                                                                                                                                                             this->acc_var,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_output_layer_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_output_layer_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_output_layer_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                             ptr_array_outputs_received[tmp_data_index],
                                                                                                                                                                                                                                                                                                                                             this->type_loss_function,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_output_layer_first_neuron->ptr_type_activation_function);
            }
        }
        //    3: Standard.
        else
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a reduction of summation.
                kernel__cuModel__FF__Compute__Error__Standard__Serialize_Batch__Parallel_Neurons<var> <<< *tmp_ptr_output_layer->ptr_dim3_grid_neurons_DP, *tmp_ptr_output_layer->ptr_dim3_block_neurons_DP >>> (this->ptr_array_accuracy_values[0],
                                                                                                                                                                                                                                                                                                                                             this->ptr_array_loss_values,
                                                                                                                                                                                                                                                                                                                                             this->acc_var,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_output_layer_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_output_layer_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_output_layer_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                             ptr_array_outputs_received[tmp_data_index],
                                                                                                                                                                                                                                                                                                                                             this->type_loss_function,
                                                                                                                                                                                                                                                                                                                                             tmp_ptr_output_layer_first_neuron->ptr_type_activation_function);
            }
        }
        // |END| KERNEL LAUNCH |END|

        // Synchronize before using the computed derivatives values.
        CUDA__Check_Error();
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        var const *tmp_ptr_array_desireds_outputs,
                      *tmp_ptr_array_summations,
                      *tmp_ptr_array_values,
                      *tmp_ptr_array_errors_end;
        var *tmp_ptr_array_errors,
             tmp_difference,
             tmp_output_value,
             tmp_desired_value;
        
        enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *tmp_ptr_array_type_activations_functions;
        
        // Loop through each sample.
        for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
        {
            tmp_ptr_array_desireds_outputs = ptr_array_outputs_received[tmp_data_index];
            tmp_ptr_array_summations = tmp_ptr_output_layer_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units;
            tmp_ptr_array_values = tmp_ptr_output_layer_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units;
            tmp_ptr_array_errors = tmp_ptr_output_layer_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units;
            tmp_ptr_array_errors_end = tmp_ptr_array_errors + tmp_number_neuron_units - 1_UZ; // Subtract bias.

            tmp_ptr_array_type_activations_functions = tmp_ptr_output_layer_first_neuron->ptr_type_activation_function;

            for(; tmp_ptr_array_errors != tmp_ptr_array_errors_end; ++tmp_ptr_array_errors,
                                                                                            ++tmp_ptr_array_desireds_outputs,
                                                                                            ++tmp_ptr_array_summations,
                                                                                            ++tmp_ptr_array_values,
                                                                                            ++tmp_ptr_array_type_activations_functions)
            {
                tmp_output_value = *tmp_ptr_array_values;

                tmp_desired_value = *tmp_ptr_array_desireds_outputs;

                tmp_difference = tmp_output_value - tmp_desired_value; // Gradient descent
                
                update_loss(tmp_output_value,
                                    tmp_desired_value,
                                    tmp_difference,
                                    this->ptr_array_loss_values,
                                    this->type_loss_function);
                
                Update_Accuracy(tmp_difference,
                                           this->acc_var,
                                           this->ptr_array_accuracy_values[0]);
                
                *tmp_ptr_array_errors = Activation_Derived(1_r,
                                                                              *tmp_ptr_array_summations,
                                                                              tmp_output_value,
                                                                              *tmp_ptr_array_type_activations_functions,
                                                                              this->type_loss_function) * tmp_difference;
            }
        }
    }
}
