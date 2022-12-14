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
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative(size_t const data_index_received,
                                                                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                                                                    size_t const next_layer_number_neurons_received,
                                                                                                                                                                                                                    size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                    T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                    T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_values_hats_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_values_normalizes_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_values_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_errors_received,
                                                                                                                                                                                                                    T **const ptr_array_layer_it_reduce_errors_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_scales_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_scales_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_shifts_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_means_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_variances_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                                                    enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                                                                    struct dim3 const *const ptr_array_layer_it_dim3_grid_reduce_error_received,
                                                                                                                                                                                                                    struct dim3 const *const ptr_array_layer_it_dim3_block_reduce_error_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative_0[];
    /* Index map:
        0: error
        1: variance_b
        2: negate_r_correction */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative_0;
    T *const tmp_ptr_array_reduce_error(ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received);
    
    Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                        number_neurons_received,
                                        tmp_ptr_array_reduce_error,
                                        ptr_array_next_layer_parameters_received + tmp_thread_global_index * next_layer_number_neurons_received,
                                        ptr_array_next_layer_errors_received,
                                        ptr_array_layer_it_dim3_grid_reduce_error_received + tmp_thread_global_index,
                                        ptr_array_layer_it_dim3_block_reduce_error_received + tmp_thread_global_index);
    
    tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_variances_received[tmp_thread_global_index];
    tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -ptr_array_layer_it_r_correction_received[tmp_thread_global_index];

    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    tmp_ptr_array_smem[threadIdx.x] = *tmp_ptr_array_reduce_error;
    
    // Derivative function.
    // dY *= dAF(value_normalize)
    tmp_ptr_array_smem[threadIdx.x] *= Activation_Derived(T(1),
                                                                                     ptr_array_layer_it_values_normalizes_received[tmp_thread_global_index],
                                                                                     ptr_array_layer_it_values_received[tmp_thread_global_index],
                                                                                     ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);

    // Derivative scale.
    // dScale += dY * value_hat
    ptr_array_layer_it_derivatives_scales_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x] * ptr_array_layer_it_values_hats_received[tmp_thread_global_index];
    
    // Derivative shift.
    // dShift += dY
    ptr_array_layer_it_derivatives_shifts_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x];

    // Derivative value hat.
    // dX_h = dY * scale
    tmp_ptr_array_smem[threadIdx.x] *= ptr_array_layer_it_scales_received[tmp_thread_global_index];
    
    // dMean_b += dX_h * ( -r_correction / variance_b )
    ptr_array_layer_it_derivatives_means_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x] * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + blockDim.x] );

    // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
    ptr_array_layer_it_derivatives_variances_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x] * (ptr_array_layer_it_summations_received[tmp_thread_global_index] - ptr_array_layer_it_means_received[tmp_thread_global_index]) * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / (tmp_ptr_array_smem[threadIdx.x + blockDim.x] * tmp_ptr_array_smem[threadIdx.x + blockDim.x]) );

    ptr_array_layer_it_errors_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x];
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative(size_t const size_received,
                                                                                                                                                                                                                    size_t const data_index_received,
                                                                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                                                                    size_t const next_layer_number_neurons_received,
                                                                                                                                                                                                                    size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                    T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                    T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_values_hats_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_values_normalizes_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_values_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_errors_received,
                                                                                                                                                                                                                    T **const ptr_array_layer_it_reduce_errors_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_scales_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_scales_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_shifts_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_means_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_variances_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                                                    enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                                                                    struct dim3 const *const ptr_array_layer_it_dim3_grid_reduce_error_received,
                                                                                                                                                                                                                    struct dim3 const *const ptr_array_layer_it_dim3_block_reduce_error_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative_1[];
    /* Index map:
        0: error
        1: variance_b
        2: negate_r_correction */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative_1;
    T *tmp_ptr_array_reduce_error;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_reduce_error = ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received;
        
        Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                            number_neurons_received,
                                            tmp_ptr_array_reduce_error,
                                            ptr_array_next_layer_parameters_received + tmp_thread_global_index * next_layer_number_neurons_received,
                                            ptr_array_next_layer_errors_received,
                                            ptr_array_layer_it_dim3_grid_reduce_error_received + tmp_thread_global_index,
                                            ptr_array_layer_it_dim3_block_reduce_error_received + tmp_thread_global_index);
        
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_variances_received[tmp_thread_global_index];
        tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -ptr_array_layer_it_r_correction_received[tmp_thread_global_index];
    }
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_smem[threadIdx.x] = *tmp_ptr_array_reduce_error;
        
        // Derivative function.
        // dY *= dAF(value_normalize)
        tmp_ptr_array_smem[threadIdx.x] *= Activation_Derived(T(1),
                                                                                         ptr_array_layer_it_values_normalizes_received[tmp_thread_global_index],
                                                                                         ptr_array_layer_it_values_received[tmp_thread_global_index],
                                                                                         ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);

        // Derivative scale.
        // dScale += dY * value_hat
        ptr_array_layer_it_derivatives_scales_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x] * ptr_array_layer_it_values_hats_received[tmp_thread_global_index];
        
        // Derivative shift.
        // dShift += dY
        ptr_array_layer_it_derivatives_shifts_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x];

        // Derivative value hat.
        // dX_h = dY * scale
        tmp_ptr_array_smem[threadIdx.x] *= ptr_array_layer_it_scales_received[tmp_thread_global_index];
        
        // dMean_b += dX_h * ( -r_correction / variance_b )
        ptr_array_layer_it_derivatives_means_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x] * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + blockDim.x] );

        // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
        ptr_array_layer_it_derivatives_variances_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x] * (ptr_array_layer_it_summations_received[tmp_thread_global_index] - ptr_array_layer_it_means_received[tmp_thread_global_index]) * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / (tmp_ptr_array_smem[threadIdx.x + blockDim.x] * tmp_ptr_array_smem[threadIdx.x + blockDim.x]) );

        ptr_array_layer_it_errors_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x];
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative(size_t const size_received,
                                                                                                                                                                                                                            size_t const data_index_received,
                                                                                                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                                                                                                            size_t const next_layer_number_neurons_received,
                                                                                                                                                                                                                            size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                            T const *const ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                            T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                                                                                            T const *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                                            T const *const ptr_array_layer_it_values_hats_received,
                                                                                                                                                                                                                            T const *const ptr_array_layer_it_values_normalizes_received,
                                                                                                                                                                                                                            T const *const ptr_array_layer_it_values_received,
                                                                                                                                                                                                                            T *const ptr_array_layer_it_errors_received,
                                                                                                                                                                                                                            T **const ptr_array_layer_it_reduce_errors_received,
                                                                                                                                                                                                                            T const *const ptr_array_layer_it_scales_received,
                                                                                                                                                                                                                            T *const ptr_array_layer_it_derivatives_scales_received,
                                                                                                                                                                                                                            T *const ptr_array_layer_it_derivatives_shifts_received,
                                                                                                                                                                                                                            T const *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                                            T const *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                                            T *const ptr_array_layer_it_derivatives_means_received,
                                                                                                                                                                                                                            T *const ptr_array_layer_it_derivatives_variances_received,
                                                                                                                                                                                                                            T const *const ptr_array_layer_it_r_correction_received,
                                                                                                                                                                                                                            enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_layer_it_type_activations_functions_received,
                                                                                                                                                                                                                            struct dim3 const *const ptr_array_layer_it_dim3_grid_reduce_error_received,
                                                                                                                                                                                                                            struct dim3 const *const ptr_array_layer_it_dim3_block_reduce_error_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative[];
    /* Index map:
        0: error
        1: variance_b
        2: negate_r_correction */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative;
    
    // Loop through each neurons.
    do
    {
        Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                            number_neurons_received,
                                            ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received,
                                            ptr_array_next_layer_parameters_received + tmp_thread_global_index * next_layer_number_neurons_received,
                                            ptr_array_next_layer_errors_received,
                                            ptr_array_layer_it_dim3_grid_reduce_error_received + tmp_thread_global_index,
                                            ptr_array_layer_it_dim3_block_reduce_error_received + tmp_thread_global_index);

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce_XX" Function.
    // => Synchronisation before using the reduced summation of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop through each neurons.
    do
    {
        tmp_ptr_array_smem[threadIdx.x] = *(ptr_array_layer_it_reduce_errors_received[tmp_thread_global_index] + data_index_received * neurons_total_reduce_error_size_received);
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_variances_received[tmp_thread_global_index];
        tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -ptr_array_layer_it_r_correction_received[tmp_thread_global_index];

        // Derivative function.
        // dY *= dAF(value_normalize)
        tmp_ptr_array_smem[threadIdx.x] *= Activation_Derived(T(1),
                                                                                         ptr_array_layer_it_values_normalizes_received[tmp_thread_global_index],
                                                                                         ptr_array_layer_it_values_received[tmp_thread_global_index],
                                                                                         ptr_array_layer_it_type_activations_functions_received[tmp_thread_global_index]);

        // Derivative scale.
        // dScale += dY * value_hat
        ptr_array_layer_it_derivatives_scales_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x] * ptr_array_layer_it_values_hats_received[tmp_thread_global_index];
        
        // Derivative shift.
        // dShift += dY
        ptr_array_layer_it_derivatives_shifts_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x];

        // Derivative value hat.
        // dX_h = dY * scale
        tmp_ptr_array_smem[threadIdx.x] *= ptr_array_layer_it_scales_received[tmp_thread_global_index];
        
        // dMean_b += dX_h * ( -r_correction / variance_b )
        ptr_array_layer_it_derivatives_means_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x] * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + blockDim.x] );

        // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
        ptr_array_layer_it_derivatives_variances_received[tmp_thread_global_index] += tmp_ptr_array_smem[threadIdx.x] * (ptr_array_layer_it_summations_received[tmp_thread_global_index] - ptr_array_layer_it_means_received[tmp_thread_global_index]) * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / (tmp_ptr_array_smem[threadIdx.x + blockDim.x] * tmp_ptr_array_smem[threadIdx.x + blockDim.x]) );

        ptr_array_layer_it_errors_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x];
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative(size_t const data_index_received,
                                                                                                                                                                                                           T const T_batch_size_received,
                                                                                                                                                                                                           T *const ptr_array_layer_it_errors_received,
                                                                                                                                                                                                           T const *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                           T const *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                           T const *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                           T const *const ptr_array_layer_it_derivatives_means_received,
                                                                                                                                                                                                           T const *const ptr_array_layer_it_derivatives_variances_received,
                                                                                                                                                                                                           T const *const ptr_array_layer_it_r_corrections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative_0[];
    /* Index map:
        0: error
        1: variance_b */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative_0;

    tmp_ptr_array_smem[threadIdx.x] = ptr_array_layer_it_errors_received[tmp_thread_global_index];
    tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_variances_received[tmp_thread_global_index];

    // First
    // dX_h *= r_correction / variance_b
    tmp_ptr_array_smem[threadIdx.x] *= ptr_array_layer_it_r_corrections_received[tmp_thread_global_index] / tmp_ptr_array_smem[threadIdx.x + blockDim.x];
        
    // Middle
    // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
    tmp_ptr_array_smem[threadIdx.x] += ptr_array_layer_it_derivatives_variances_received[tmp_thread_global_index] * ( (ptr_array_layer_it_summations_received[tmp_thread_global_index] - ptr_array_layer_it_means_received[tmp_thread_global_index]) / (T_batch_size_received * tmp_ptr_array_smem[threadIdx.x + blockDim.x]) );

    // Last
    // dX_h += dMean_b * 1 / N
    // dX_h += dMean_b / N
    tmp_ptr_array_smem[threadIdx.x] += ptr_array_layer_it_derivatives_means_received[tmp_thread_global_index] / T_batch_size_received;

    // dX = dX_h
    ptr_array_layer_it_errors_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x];
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative(size_t const size_received,
                                                                                                                                                                                                          size_t const data_index_received,
                                                                                                                                                                                                          T const T_batch_size_received,
                                                                                                                                                                                                        T *const ptr_array_layer_it_errors_received,
                                                                                                                                                                                                        T const *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                        T const *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                        T const *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                        T const *const ptr_array_layer_it_derivatives_means_received,
                                                                                                                                                                                                        T const *const ptr_array_layer_it_derivatives_variances_received,
                                                                                                                                                                                                        T const *const ptr_array_layer_it_r_corrections_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative_1[];
    /* Index map:
        0: error
        1: variance_b */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative_1;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_smem[threadIdx.x] = ptr_array_layer_it_errors_received[tmp_thread_global_index];
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_variances_received[tmp_thread_global_index];

        // First
        // dX_h *= r_correction / variance_b
        tmp_ptr_array_smem[threadIdx.x] *= ptr_array_layer_it_r_corrections_received[tmp_thread_global_index] / tmp_ptr_array_smem[threadIdx.x + blockDim.x];
        
        // Middle
        // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
        tmp_ptr_array_smem[threadIdx.x] += ptr_array_layer_it_derivatives_variances_received[tmp_thread_global_index] * ( (ptr_array_layer_it_summations_received[tmp_thread_global_index] - ptr_array_layer_it_means_received[tmp_thread_global_index]) / (T_batch_size_received * tmp_ptr_array_smem[threadIdx.x + blockDim.x]) );

        // Last
        // dX_h += dMean_b * 1 / N
        // dX_h += dMean_b / N
        tmp_ptr_array_smem[threadIdx.x] += ptr_array_layer_it_derivatives_means_received[tmp_thread_global_index] / T_batch_size_received;

        // dX = dX_h
        ptr_array_layer_it_errors_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x];
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative(size_t const size_received,
                                                                                                                                                                                                                    size_t const data_index_received,
                                                                                                                                                                                                                    T const T_batch_size_received,
                                                                                                                                                                                                                    T *const ptr_array_layer_it_errors_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_summations_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_means_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_variances_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_derivatives_means_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_derivatives_variances_received,
                                                                                                                                                                                                                    T const *const ptr_array_layer_it_r_corrections_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    extern __shared__ T tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative[];
    /* Index map:
        0: error
        1: variance_b */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative;

    // Loop through each neurons.
    do
    {
        tmp_ptr_array_smem[threadIdx.x] = ptr_array_layer_it_errors_received[tmp_thread_global_index];
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = ptr_array_layer_it_variances_received[tmp_thread_global_index];

        // First
        // dX_h *= r_correction / variance_b
        tmp_ptr_array_smem[threadIdx.x] *= ptr_array_layer_it_r_corrections_received[tmp_thread_global_index] / tmp_ptr_array_smem[threadIdx.x + blockDim.x];
        
        // Middle
        // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
        tmp_ptr_array_smem[threadIdx.x] += ptr_array_layer_it_derivatives_variances_received[tmp_thread_global_index] * ( (ptr_array_layer_it_summations_received[tmp_thread_global_index] - ptr_array_layer_it_means_received[tmp_thread_global_index]) / (T_batch_size_received * tmp_ptr_array_smem[threadIdx.x + blockDim.x]) );

        // Last
        // dX_h += dMean_b * 1 / N
        // dX_h += dMean_b / N
        tmp_ptr_array_smem[threadIdx.x] += ptr_array_layer_it_derivatives_means_received[tmp_thread_global_index] / T_batch_size_received;

        // dX = dX_h
        ptr_array_layer_it_errors_received[tmp_thread_global_index] = tmp_ptr_array_smem[threadIdx.x];
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative(size_t const total_parameters_received,
                                                                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                                                                    size_t const next_layer_number_neurons_received,
                                                                                                                                                                                                                    size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                    T const *ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                    T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                                                                                    struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                                    struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;
    
    extern __shared__ T tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative_0[];
    /* Index map:
        0: error
        1: variance_b
        2: negate_r_correction */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative_0;
    T const *tmp_ptr_array_next_layer_errors(ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u)); // Add bias.

    // Loop through each neurons for doing a reduction of summation.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                         ptr_array_next_layer_parameters_received += next_layer_number_neurons_received)
    {
        Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                             number_neurons_received,
                                             *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received,
                                             ptr_array_next_layer_parameters_received,
                                             tmp_ptr_array_next_layer_errors,
                                             tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                             tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
    }

    // Do we need to synchronise? Based on "Reduce::Reduce_XX" Function.
    // => Synchronisation before using the summed error of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // Loop through each neurons for doing a reduction of summation.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
    {
        tmp_ptr_array_smem[threadIdx.x] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received);
        tmp_ptr_array_smem[threadIdx.x + blockDim.x] = *tmp_ptr_neuron_unit_it->ptr_array_variances;
        tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -*tmp_ptr_neuron_unit_it->ptr_d_correction; // Negate.

        // Derivative function.
        // dY *= dAF(value_normalize)
        tmp_ptr_array_smem[threadIdx.x] *= Activation_Derived(T(1),
                                                        tmp_ptr_neuron_unit_it->ptr_array_values_normalizes[tmp_thread_global_index * number_neurons_received],
                                                        tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                                        *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

        // Derivative scale.
        // dScale += dY * value_hat
        tmp_ptr_neuron_unit_it->ptr_array_derivatives_scales[tmp_thread_global_index * total_parameters_received] += tmp_ptr_array_smem[threadIdx.x] * tmp_ptr_neuron_unit_it->ptr_array_values_hats[tmp_thread_global_index * number_neurons_received];
        
        // Derivative shift.
        // dShift += dY
        tmp_ptr_neuron_unit_it->ptr_array_derivatives_shifts[tmp_thread_global_index * total_parameters_received] += tmp_ptr_array_smem[threadIdx.x];

        // Derivative value hat.
        // dX_h = dY * scale
        tmp_ptr_array_smem[threadIdx.x] *= *tmp_ptr_neuron_unit_it->ptr_scale;
            
        // dMean_b += dX_h * ( -r_correction / variance_b )
        tmp_ptr_neuron_unit_it->ptr_array_derivatives_means[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_smem[threadIdx.x] * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + blockDim.x] );

        // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
        tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_smem[threadIdx.x] * (tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] - *tmp_ptr_neuron_unit_it->ptr_array_means) * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / (tmp_ptr_array_smem[threadIdx.x + blockDim.x] * tmp_ptr_array_smem[threadIdx.x + blockDim.x]) );

        tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_smem[threadIdx.x];
    }
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative(size_t const size_received,
                                                                                                                                                                                                                    size_t const total_parameters_received,
                                                                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                                                                    size_t const next_layer_number_neurons_received,
                                                                                                                                                                                                                    size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                    T const *ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                    T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                                                                                    struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                                    struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;
    
    extern __shared__ T tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative_1[];
    /* Index map:
        0: error
        1: variance_b
        2: negate_r_correction */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative_1;
    T const *tmp_ptr_array_next_layer_errors;

    if(tmp_thread_global_index < size_received)
    {
        tmp_ptr_array_next_layer_errors = ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u); // Add bias.

        // Loop through each neurons for doing a reduction of summation.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                              ptr_array_next_layer_parameters_received += next_layer_number_neurons_received)
        {
            Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                                 number_neurons_received,
                                                 *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received,
                                                 ptr_array_next_layer_parameters_received,
                                                 tmp_ptr_array_next_layer_errors,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
        }
    }

    // Do we need to synchronise? Based on "Reduce::Reduce_XX" Function.
    // => Synchronisation before using the summed error of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons for doing a reduction of summation.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_ptr_array_smem[threadIdx.x] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received);
            tmp_ptr_array_smem[threadIdx.x + blockDim.x] = *tmp_ptr_neuron_unit_it->ptr_array_variances;
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -*tmp_ptr_neuron_unit_it->ptr_d_correction; // Negate.

            // Derivative function.
            // dY *= dAF(value_normalize)
            tmp_ptr_array_smem[threadIdx.x] *= Activation_Derived(T(1),
                                                            tmp_ptr_neuron_unit_it->ptr_array_values_normalizes[tmp_thread_global_index * number_neurons_received],
                                                            tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                                            *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

            // Derivative scale.
            // dScale += dY * value_hat
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_scales[tmp_thread_global_index * total_parameters_received] += tmp_ptr_array_smem[threadIdx.x] * tmp_ptr_neuron_unit_it->ptr_array_values_hats[tmp_thread_global_index * number_neurons_received];
        
            // Derivative shift.
            // dShift += dY
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_shifts[tmp_thread_global_index * total_parameters_received] += tmp_ptr_array_smem[threadIdx.x];

            // Derivative value hat.
            // dX_h = dY * scale
            tmp_ptr_array_smem[threadIdx.x] *= *tmp_ptr_neuron_unit_it->ptr_scale;
            
            // dMean_b += dX_h * ( -r_correction / variance_b )
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_means[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_smem[threadIdx.x] * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + blockDim.x] );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_smem[threadIdx.x] * (tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] - *tmp_ptr_neuron_unit_it->ptr_array_means) * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / (tmp_ptr_array_smem[threadIdx.x + blockDim.x] * tmp_ptr_array_smem[threadIdx.x + blockDim.x]) );

            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_smem[threadIdx.x];
        }
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative(size_t const size_received,
                                                                                                                                                                                                                            size_t const total_parameters_received,
                                                                                                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                                                                                                            size_t const next_layer_number_neurons_received,
                                                                                                                                                                                                                            size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                            T const *ptr_array_next_layer_parameters_received,
                                                                                                                                                                                                                            T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                                                                                            struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                                            struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_grid_index(blockIdx.x * blockDim.x + threadIdx.x);
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;
    
    extern __shared__ T tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative[];
    /* Index map:
        0: error
        1: variance_b
        2: negate_r_correction */
    T (&tmp_ptr_array_smem)[] = tmp_shared_T__kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative;
    T const *tmp_ptr_array_next_layer_parameters,
                *tmp_ptr_array_next_layer_errors;

    // Loop through each sample.
    do
    {
        tmp_ptr_array_next_layer_errors = ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u); // Add bias.

        // Loop through each neurons for doing a reduction of summation.
        for(tmp_ptr_array_next_layer_parameters = ptr_array_next_layer_parameters_received,
            tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                              tmp_ptr_array_next_layer_parameters += next_layer_number_neurons_received)
        {
            Reduce::Reduce_XX<T>(next_layer_number_neurons_received,
                                                 number_neurons_received,
                                                 *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received,
                                                 tmp_ptr_array_next_layer_parameters,
                                                 tmp_ptr_array_next_layer_errors,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                                 tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);

    // Do we need to synchronise? Based on "Reduce::Reduce_XX" Function.
    // => Synchronisation before using the summed error of the layer.
    if(next_layer_number_neurons_received >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }
    
    // reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Loop through each sample.
    do
    {
        // Loop through each neurons for doing a reduction of summation.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_ptr_array_smem[threadIdx.x] = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_thread_global_index * neurons_total_reduce_error_size_received);
            tmp_ptr_array_smem[threadIdx.x + blockDim.x] = *tmp_ptr_neuron_unit_it->ptr_array_variances;
            tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] = -*tmp_ptr_neuron_unit_it->ptr_d_correction; // Negate.

            // Derivative function.
            // dY *= dAF(value_normalize)
            tmp_ptr_array_smem[threadIdx.x] *= Activation_Derived(T(1),
                                                            tmp_ptr_neuron_unit_it->ptr_array_values_normalizes[tmp_thread_global_index * number_neurons_received],
                                                            tmp_ptr_neuron_unit_it->ptr_array_values[tmp_thread_global_index * number_neurons_received],
                                                            *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

            // Derivative scale.
            // dScale += dY * value_hat
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_scales[tmp_thread_grid_index * total_parameters_received] += tmp_ptr_array_smem[threadIdx.x] * tmp_ptr_neuron_unit_it->ptr_array_values_hats[tmp_thread_global_index * number_neurons_received];
        
            // Derivative shift.
            // dShift += dY
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_shifts[tmp_thread_grid_index * total_parameters_received] += tmp_ptr_array_smem[threadIdx.x];

            // Derivative value hat.
            // dX_h = dY * scale
            tmp_ptr_array_smem[threadIdx.x] *= *tmp_ptr_neuron_unit_it->ptr_scale;
            
            // dMean_b += dX_h * ( -r_correction / variance_b )
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_means[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_smem[threadIdx.x] * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / tmp_ptr_array_smem[threadIdx.x + blockDim.x] );

            // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
            tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances[tmp_thread_global_index * number_neurons_received] += tmp_ptr_array_smem[threadIdx.x] * (tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] - *tmp_ptr_neuron_unit_it->ptr_array_means) * ( tmp_ptr_array_smem[threadIdx.x + 2u * blockDim.x] / (tmp_ptr_array_smem[threadIdx.x + blockDim.x] * tmp_ptr_array_smem[threadIdx.x + blockDim.x]) );

            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = tmp_ptr_array_smem[threadIdx.x];
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Final_Derivative(size_t const number_neurons_received,
                                                                                                                                                                                                          T const T_batch_size_received,
                                                                                                                                                                                                          struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                          struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;
    
    T tmp_error,
       tmp_variance_b;

    // Loop through each neurons for doing a reduction of summation.
    for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
    {
        tmp_error = tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received];
        tmp_variance_b = *tmp_ptr_neuron_unit_it->ptr_array_variances;

        // First
        // dX_h *= r_correction / variance_b
        tmp_error *= *tmp_ptr_neuron_unit_it->ptr_d_correction / tmp_variance_b;
            
        // Middle
        // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
        tmp_error += *tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances * ( (tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] - *tmp_ptr_neuron_unit_it->ptr_array_means) / (T_batch_size_received * tmp_variance_b) );

        // Last
        // dX_h += dMean_b * 1 / N
        // dX_h += dMean_b / N
        tmp_error += *tmp_ptr_neuron_unit_it->ptr_array_derivatives_means / T_batch_size_received;

        // dX = dX_h
        tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = tmp_error;
    }
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Final_Derivative(size_t const size_received,
                                                                                                                                                                                                          size_t const number_neurons_received,
                                                                                                                                                                                                          T const T_batch_size_received,
                                                                                                                                                                                                          struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                          struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;
    
    T tmp_error,
       tmp_variance_b;

    if(tmp_thread_global_index < size_received)
    {
        // Loop through each neurons for doing a reduction of summation.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_error = tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received];
            tmp_variance_b = *tmp_ptr_neuron_unit_it->ptr_array_variances;

            // First
            // dX_h *= r_correction / variance_b
            tmp_error *= *tmp_ptr_neuron_unit_it->ptr_d_correction / tmp_variance_b;
            
            // Middle
            // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
            tmp_error += *tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances * ( (tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] - *tmp_ptr_neuron_unit_it->ptr_array_means) / (T_batch_size_received * tmp_variance_b) );

            // Last
            // dX_h += dMean_b * 1 / N
            // dX_h += dMean_b / N
            tmp_error += *tmp_ptr_neuron_unit_it->ptr_array_derivatives_means / T_batch_size_received;

            // dX = dX_h
            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = tmp_error;
        }
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Final_Derivative(size_t const size_received,
                                                                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                                                                    T const T_batch_size_received,
                                                                                                                                                                                                                    struct cuNeuron *const ptr_layer_it_first_neuron_received,
                                                                                                                                                                                                                    struct cuNeuron const *const ptr_layer_it_last_neuron_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *tmp_ptr_neuron_unit_it;
    
    T tmp_error,
       tmp_variance_b;

    // Loop through each sample.
    do
    {
        // Loop through each neurons for doing a reduction of summation.
        for(tmp_ptr_neuron_unit_it = ptr_layer_it_first_neuron_received; tmp_ptr_neuron_unit_it != ptr_layer_it_last_neuron_received; ++tmp_ptr_neuron_unit_it)
        {
            tmp_error = tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received];
            tmp_variance_b = *tmp_ptr_neuron_unit_it->ptr_array_variances;

            // First
            // dX_h *= r_correction / variance_b
            tmp_error *= *tmp_ptr_neuron_unit_it->ptr_d_correction / tmp_variance_b;
            
            // Middle
            // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
            tmp_error += *tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances * ( (tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_thread_global_index * number_neurons_received] - *tmp_ptr_neuron_unit_it->ptr_array_means) / (T_batch_size_received * tmp_variance_b) );

            // Last
            // dX_h += dMean_b * 1 / N
            // dX_h += dMean_b / N
            tmp_error += *tmp_ptr_neuron_unit_it->ptr_array_derivatives_means / T_batch_size_received;

            // dX = dX_h
            tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_thread_global_index * number_neurons_received] = tmp_error;
        }
        
        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Summation_Derivative(size_t const total_parameters_received,
                                                                                                                                                                                                                  size_t const number_neurons_received,
                                                                                                                                                                                                                  size_t const next_layer_number_neurons_received,
                                                                                                                                                                                                                  size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                  T const *const ptr_array_next_parameters_received,
                                                                                                                                                                                                                  T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                                                                                  struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative<T>,
                                                        layer_it->ptr_dim3_grid_neurons,
                                                        layer_it->ptr_dim3_block_neurons,
                                                        layer_it->ptr_dim3_block_neurons->x * 3u * sizeof(T),
                                                        number_neurons_received - 1, // Subtract bias.
                                                        tmp_thread_global_index,
                                                        number_neurons_received,
                                                        next_layer_number_neurons_received,
                                                        neurons_total_reduce_error_size_received,
                                                        ptr_array_next_parameters_received,
                                                        ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                        tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_scales + blockIdx.x * total_parameters_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_shifts + blockIdx.x * total_parameters_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances + tmp_thread_global_index * number_neurons_received,
                                                        tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                        tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                        tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error)
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Summation_Derivative(size_t const size_received,
                                                                                                                                                                                                                  size_t const total_parameters_received,
                                                                                                                                                                                                                  size_t const number_neurons_received,
                                                                                                                                                                                                                  size_t const next_layer_number_neurons_received,
                                                                                                                                                                                                                  size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                  T const *const ptr_array_next_parameters_received,
                                                                                                                                                                                                                  T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                                                                                  struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative<T>,
                                                            layer_it->ptr_dim3_grid_neurons,
                                                            layer_it->ptr_dim3_block_neurons,
                                                            layer_it->ptr_dim3_block_neurons->x * 3u * sizeof(T),
                                                            number_neurons_received - 1, // Subtract bias.
                                                            tmp_thread_global_index,
                                                            number_neurons_received,
                                                            next_layer_number_neurons_received,
                                                            neurons_total_reduce_error_size_received,
                                                            ptr_array_next_parameters_received,
                                                            ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                            tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_scales + blockIdx.x * total_parameters_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_shifts + blockIdx.x * total_parameters_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                            tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error)
    }
    // |END| KERNEL LAUNCH |END|
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Summation_Derivative(size_t const size_received,
                                                                                                                                                                                                                            size_t const total_parameters_received,
                                                                                                                                                                                                                            size_t const number_neurons_received,
                                                                                                                                                                                                                            size_t const next_layer_number_neurons_received,
                                                                                                                                                                                                                            size_t const neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                            T const *const ptr_array_next_parameters_received,
                                                                                                                                                                                                                            T const *const ptr_array_next_layer_errors_received,
                                                                                                                                                                                                                            struct cuLayer *const layer_it)
{
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
            kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative<var> <<< *layer_it->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                                  *layer_it->ptr_dim3_block_neurons,
                                                                                                                                                                                                                                  layer_it->ptr_dim3_block_neurons->x * 3u * sizeof(T) >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                          tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                                          number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          next_layer_number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                                                                                                                                          ptr_array_next_parameters_received,
                                                                                                                                                                                                                                                                                                                                          ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_scales + blockIdx.x * total_parameters_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_shifts + blockIdx.x * total_parameters_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                                                          tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
        
            tmp_thread_global_index += gridDim.x * blockDim.x;
        } while(tmp_thread_global_index < size_received);
    }
    //    2: Launching size condition.
    else if(layer_it->ptr_dim3_grid_neurons->x * layer_it->ptr_dim3_block_neurons->x > number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative<var> <<< *layer_it->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                        *layer_it->ptr_dim3_block_neurons,
                                                                                                                                                                                                                        layer_it->ptr_dim3_block_neurons->x * 3u * sizeof(T) >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                                number_neurons_received,
                                                                                                                                                                                                                                                                                                                                next_layer_number_neurons_received,
                                                                                                                                                                                                                                                                                                                                neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                                                                                                                                ptr_array_next_parameters_received,
                                                                                                                                                                                                                                                                                                                                ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_scales + blockIdx.x * total_parameters_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_shifts + blockIdx.x * total_parameters_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
        
            tmp_thread_global_index += gridDim.x * blockDim.x;
        } while(tmp_thread_global_index < size_received);
    }
    //    3: Standard.
    else
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative<var> <<< *layer_it->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                        *layer_it->ptr_dim3_block_neurons,
                                                                                                                                                                                                                        layer_it->ptr_dim3_block_neurons->x * 3u * sizeof(T) >>> (tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                                number_neurons_received,
                                                                                                                                                                                                                                                                                                                                next_layer_number_neurons_received,
                                                                                                                                                                                                                                                                                                                                neurons_total_reduce_error_size_received,
                                                                                                                                                                                                                                                                                                                                ptr_array_next_parameters_received,
                                                                                                                                                                                                                                                                                                                                ptr_array_next_layer_errors_received + tmp_thread_global_index * (next_layer_number_neurons_received + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_scales + blockIdx.x * total_parameters_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_shifts + blockIdx.x * total_parameters_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
        
            tmp_thread_global_index += gridDim.x * blockDim.x;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Final_Derivative(size_t const number_neurons_received,
                                                                                                                                                                                                        T const T_batch_size_received,
                                                                                                                                                                                                        struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron const *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative<T>,
                                                      layer_it->ptr_dim3_grid_neurons,
                                                      layer_it->ptr_dim3_block_neurons,
                                                      layer_it->ptr_dim3_block_neurons->x * 2u * sizeof(T),
                                                      number_neurons_received - 1, // Subtract bias.
                                                      tmp_thread_global_index,
                                                      T_batch_size_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                      tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                      tmp_ptr_layer_it_first_neuron->ptr_r_correction)
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Final_Derivative(size_t const size_received,
                                                                                                                                                                                                        size_t const number_neurons_received,
                                                                                                                                                                                                        T const T_batch_size_received,
                                                                                                                                                                                                        struct cuLayer *const layer_it)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron const *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    if(tmp_thread_global_index < size_received)
    {
        LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative<T>,
                                                            layer_it->ptr_dim3_grid_neurons,
                                                            layer_it->ptr_dim3_block_neurons,
                                                            layer_it->ptr_dim3_block_neurons->x * 2u * sizeof(T),
                                                            number_neurons_received - 1, // Subtract bias.
                                                            tmp_thread_global_index,
                                                            T_batch_size_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                            tmp_ptr_layer_it_first_neuron->ptr_r_correction)
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Final_Derivative(size_t const size_received,
                                                                                                                                                                                                                 size_t const number_neurons_received,
                                                                                                                                                                                                                 T const T_batch_size_received,
                                                                                                                                                                                                                 struct cuLayer *const layer_it)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    struct cuNeuron const *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units);

    // KERNEL LAUNCH
    //    1: Launching do-while elements.
    if(layer_it->ptr_dim3_grid_neurons->x * layer_it->ptr_dim3_block_neurons->x < number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative<T> <<< *layer_it->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                     *layer_it->ptr_dim3_block_neurons,
                                                                                                                                                                                                                     layer_it->ptr_dim3_block_neurons->x * 2u * sizeof(T) >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                            tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                            T_batch_size_received,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_r_correction);
            
            tmp_thread_global_index += gridDim.x * blockDim.x;
        } while(tmp_thread_global_index < size_received);
    }
    //    2: Launching size condition.
    else if(layer_it->ptr_dim3_grid_neurons->x * layer_it->ptr_dim3_block_neurons->x > number_neurons_received - 1u) // Subtract bias.
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative<T> <<< *layer_it->ptr_dim3_grid_neurons,
                                                                                                                                                                                                            *layer_it->ptr_dim3_block_neurons,
                                                                                                                                                                                                            layer_it->ptr_dim3_block_neurons->x * 2u * sizeof(T) >>> (number_neurons_received - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                T_batch_size_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_r_correction);
            
            tmp_thread_global_index += gridDim.x * blockDim.x;
        } while(tmp_thread_global_index < size_received);
    }
    //    3: Standard.
    else
    {
        // Loop through each sample.
        do
        {
            // Parallel each neurons for doing a reduction of summation.
            kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative<T> <<< *layer_it->ptr_dim3_grid_neurons,
                                                                                                                                                                                                            *layer_it->ptr_dim3_block_neurons,
                                                                                                                                                                                                            layer_it->ptr_dim3_block_neurons->x * 2u * sizeof(T) >>> (tmp_thread_global_index,
                                                                                                                                                                                                                                                                                                                T_batch_size_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_thread_global_index * number_neurons_received,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron->ptr_r_correction);
            
            tmp_thread_global_index += gridDim.x * blockDim.x;
        } while(tmp_thread_global_index < size_received);
    }
    // |END| KERNEL LAUNCH |END|
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Reduce(size_t const batch_size,
                                                                                                                                                                     size_t const total_data_batch_received,
                                                                                                                                                                     size_t const number_neurons_received,
                                                                                                                                                                     T *const ptr_array_layer_it_derivatives_means_received,
                                                                                                                                                                     T *const ptr_array_layer_it_derivatives_transposed_means_received,
                                                                                                                                                                     T **const ptr_array_layer_it_reduce_means_received,
                                                                                                                                                                     T *const ptr_array_layer_it_derivatives_variances_received,
                                                                                                                                                                     T *const ptr_array_layer_it_derivatives_transposed_variances_received,
                                                                                                                                                                     T **const ptr_array_layer_it_reduce_variances_received,
                                                                                                                                                                     struct dim3 const *const ptr_array_dim3_grid_reduce_batch_received,
                                                                                                                                                                     struct dim3 const *const ptr_array_dim3_block_reduce_batch_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    Reduce::Reduce<T>(batch_size,
                                    number_neurons_received,
                                    ptr_array_layer_it_reduce_means_received[tmp_thread_global_index],
                                    ptr_array_layer_it_derivatives_transposed_means_received + tmp_thread_global_index * total_data_batch_received,
                                    ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                    ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);

    Reduce::Reduce<T>(batch_size,
                                    number_neurons_received,
                                    ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index],
                                    ptr_array_layer_it_derivatives_transposed_variances_received + tmp_thread_global_index * total_data_batch_received,
                                    ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                    ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);
    
    // Do we need to synchronise? Based on "Reduce" Function.
    // => Synchronize to see the mean and variance reduced of the layer.
    if(batch_size >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    ptr_array_layer_it_derivatives_means_received[tmp_thread_global_index] = *(ptr_array_layer_it_reduce_means_received[tmp_thread_global_index]);
    ptr_array_layer_it_derivatives_variances_received[tmp_thread_global_index] = *(ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index]);
}

template<typename T>
__global__ void kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Reduce(size_t const size_received,
                                                                                                                                                                    size_t const batch_size,
                                                                                                                                                                    size_t const total_data_batch_received,
                                                                                                                                                                    size_t const number_neurons_received,
                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_means_received,
                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_transposed_means_received,
                                                                                                                                                                    T **const ptr_array_layer_it_reduce_means_received,
                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_variances_received,
                                                                                                                                                                    T *const ptr_array_layer_it_derivatives_transposed_variances_received,
                                                                                                                                                                    T **const ptr_array_layer_it_reduce_variances_received,
                                                                                                                                                                    struct dim3 const *const ptr_array_dim3_grid_reduce_batch_received,
                                                                                                                                                                    struct dim3 const *const ptr_array_dim3_block_reduce_batch_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(tmp_thread_global_index < size_received)
    {
        Reduce::Reduce<T>(batch_size,
                                        number_neurons_received,
                                        ptr_array_layer_it_reduce_means_received[tmp_thread_global_index],
                                        ptr_array_layer_it_derivatives_transposed_means_received + tmp_thread_global_index * total_data_batch_received,
                                        ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);

        Reduce::Reduce<T>(batch_size,
                                        number_neurons_received,
                                        ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index],
                                        ptr_array_layer_it_derivatives_transposed_variances_received + tmp_thread_global_index * total_data_batch_received,
                                        ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);
    }
    
    // Do we need to synchronise? Based on "Reduce" Function.
    // => Synchronize to see the mean and variance reduced of the layer.
    if(batch_size >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_layer_it_derivatives_means_received[tmp_thread_global_index] = *(ptr_array_layer_it_reduce_means_received[tmp_thread_global_index]);
        ptr_array_layer_it_derivatives_variances_received[tmp_thread_global_index] = *(ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index]);
    }
}

template<typename T>
__global__ void kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Reduce(size_t const size_received,
                                                                                                                                                                              size_t const batch_size,
                                                                                                                                                                              size_t const total_data_batch_received,
                                                                                                                                                                              size_t const number_neurons_received,
                                                                                                                                                                              T *const ptr_array_layer_it_derivatives_means_received,
                                                                                                                                                                              T *const ptr_array_layer_it_derivatives_transposed_means_received,
                                                                                                                                                                              T **const ptr_array_layer_it_reduce_means_received,
                                                                                                                                                                              T *const ptr_array_layer_it_derivatives_variances_received,
                                                                                                                                                                              T *const ptr_array_layer_it_derivatives_transposed_variances_received,
                                                                                                                                                                              T **const ptr_array_layer_it_reduce_variances_received,
                                                                                                                                                                              struct dim3 const *const ptr_array_dim3_grid_reduce_batch_received,
                                                                                                                                                                              struct dim3 const *const ptr_array_dim3_block_reduce_batch_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    
    do
    {
        Reduce::Reduce<T>(batch_size,
                                        number_neurons_received,
                                        ptr_array_layer_it_reduce_means_received[tmp_thread_global_index],
                                        ptr_array_layer_it_derivatives_transposed_means_received + tmp_thread_global_index * total_data_batch_received,
                                        ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);

        Reduce::Reduce<T>(batch_size,
                                        number_neurons_received,
                                        ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index],
                                        ptr_array_layer_it_derivatives_transposed_variances_received + tmp_thread_global_index * total_data_batch_received,
                                        ptr_array_dim3_grid_reduce_batch_received + tmp_thread_global_index,
                                        ptr_array_dim3_block_reduce_batch_received + tmp_thread_global_index);

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
    
    // Do we need to synchronise? Based on "Reduce" Function.
    // => Synchronize to see the mean and variance reduced of the layer.
    if(batch_size >= warpSize * 2u) { CUDA__ThreadBlockSynchronize(); }

    // reset index to the initial state.
    tmp_thread_global_index = blockIdx.x * blockDim.x + threadIdx.x;

    do
    {
        ptr_array_layer_it_derivatives_means_received[tmp_thread_global_index] = *(ptr_array_layer_it_reduce_means_received[tmp_thread_global_index]);
        ptr_array_layer_it_derivatives_variances_received[tmp_thread_global_index] = *(ptr_array_layer_it_reduce_variances_received[tmp_thread_global_index]);

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::Backward_Pass__FC_to_FC__Batch_Renormalization(bool &ref_synchronized_received,
                                                                                                                                            size_t const batch_size,
                                                                                                                                            struct cuLayer *const layer_it,
                                                                                                                                            struct cuLayer *const ptr_next_layer_received,
                                                                                                                                            struct dim3 const *const ptr_dim3_batch_size_grid_received,
                                                                                                                                            struct dim3 const *const ptr_dim3_batch_size_block_received)
{
    size_t const tmp_number_neuron_units(*layer_it->ptr_number_neurons),
                                tmp_next_layer_number_neurons(*ptr_next_layer_received->ptr_number_neurons - 1u); // Subtract bias.
    size_t tmp_data_index;

    var const *tmp_ptr_next_layer_error,
                 *tmp_ptr_next_layer_parameters;
    var tmp_error,
         tmp_variance_b,
         tmp_negate_r_correction;
    
    struct cuNeuron const *const tmp_ptr_layer_it_last_neuron(layer_it->ptr_last_neuron_unit - 1), // Subtract bias.
                                                      *tmp_ptr_next_layer_neuron_it(ptr_next_layer_received->ptr_array_neuron_units);
    struct cuNeuron *const tmp_ptr_layer_it_first_neuron(layer_it->ptr_array_neuron_units),
                                             *tmp_ptr_neuron_unit_it;
    
    // Condition to enter into dynamic parallelisme of each sample.
    if(USE_PARALLEL && batch_size >= warpSize)
    {
        // Set the synchronisation state to false. Because we launch a kernel.
        ref_synchronized_received = false;
        
        // Condition to enter into dynamic parallelisme of each sample and neurons.
        if(USE_PARALLEL && tmp_number_neuron_units - 1u >= warpSize)
        {
            // KERNEL LAUNCH
            //    1: Launching do-while elements.
            if(ptr_dim3_batch_size_grid_received->x * ptr_dim3_batch_size_block_received->x < batch_size)
            {
                kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Summation_Derivative<var> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (batch_size,
                                                                                                                                                                                                                                                                                                                                                        this->total_parameters_allocated,
                                                                                                                                                                                                                                                                                                                                                        tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                        tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                                                        this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                                                        this->ptr_array_transposed_weights + *tmp_ptr_next_layer_neuron_it->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                        tmp_ptr_next_layer_neuron_it->ptr_array_errors,
                                                                                                                                                                                                                                                                                                                                                        layer_it);
                
                this->Transpose_Layer_Backward__Batch_Normalization(layer_it);

                LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Reduce<var>,
                                                                  layer_it->ptr_dim3_grid_neurons_DP,
                                                                  layer_it->ptr_dim3_block_neurons_DP,
                                                                  0_UZ,
                                                                  tmp_number_neuron_units - 1, // Subtract bias.
                                                                  batch_size,
                                                                  this->batch_size,
                                                                  tmp_number_neuron_units,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)

                kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Final_Derivative<var> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (batch_size,
                                                                                                                                                                                                                                                                                                                                              *layer_it->ptr_number_neurons,
                                                                                                                                                                                                                                                                                                                                              static_cast<var>(batch_size),
                                                                                                                                                                                                                                                                                                                                              layer_it);
            }
            //    2: Launching size condition.
            else if(ptr_dim3_batch_size_grid_received->x * ptr_dim3_batch_size_block_received->x > batch_size)
            {
                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Summation_Derivative<var> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (batch_size,
                                                                                                                                                                                                                                                                                                                                            this->total_parameters_allocated,
                                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                                            this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                                            this->ptr_array_transposed_weights + *tmp_ptr_next_layer_neuron_it->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_next_layer_neuron_it->ptr_array_errors,
                                                                                                                                                                                                                                                                                                                                            layer_it);
                
                this->Transpose_Layer_Backward__Batch_Normalization(layer_it);

                LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Reduce<var>,
                                                                  layer_it->ptr_dim3_grid_neurons_DP,
                                                                  layer_it->ptr_dim3_block_neurons_DP,
                                                                  0_UZ,
                                                                  tmp_number_neuron_units - 1, // Subtract bias.
                                                                  batch_size,
                                                                  this->batch_size,
                                                                  tmp_number_neuron_units,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)

                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Final_Derivative<var> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (batch_size,
                                                                                                                                                                                                                                                                                                                                    *layer_it->ptr_number_neurons,
                                                                                                                                                                                                                                                                                                                                    static_cast<var>(batch_size),
                                                                                                                                                                                                                                                                                                                                    layer_it);
            }
            //    3: Standard.
            else
            {
                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Summation_Derivative<var> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (this->total_parameters_allocated,
                                                                                                                                                                                                                                                                                                                                              tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                              tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                                              this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                                              this->ptr_array_transposed_weights + *tmp_ptr_next_layer_neuron_it->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                              tmp_ptr_next_layer_neuron_it->ptr_array_errors,
                                                                                                                                                                                                                                                                                                                                              layer_it);
                
                this->Transpose_Layer_Backward__Batch_Normalization(layer_it);
                
                LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Reduce<var>,
                                                                  layer_it->ptr_dim3_grid_neurons_DP,
                                                                  layer_it->ptr_dim3_block_neurons_DP,
                                                                  0_UZ,
                                                                  tmp_number_neuron_units - 1, // Subtract bias.
                                                                  batch_size,
                                                                  this->batch_size,
                                                                  tmp_number_neuron_units,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)
                    
                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Parallel_Final_Derivative<var> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (*layer_it->ptr_number_neurons,
                                                                                                                                                                                                                                                                                                                                    static_cast<var>(batch_size),
                                                                                                                                                                                                                                                                                                                                    layer_it);
            }
            // |END| KERNEL LAUNCH |END|
        }
        // Condition to enter into dynamic parallelisme of each sample.
        else
        {
            // KERNEL LAUNCH
            //    1: Launching do-while elements.
            if(ptr_dim3_batch_size_grid_received->x * ptr_dim3_batch_size_block_received->x < batch_size)
            {
                kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative<var> <<< *ptr_dim3_batch_size_grid_received,
                                                                                                                                                                                                                                    *ptr_dim3_batch_size_block_received,
                                                                                                                                                                                                                                    ptr_dim3_batch_size_block_received->x * 3u * sizeof(var) >>> (batch_size,
                                                                                                                                                                                                                                                                                                                            this->total_parameters_allocated,
                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                            tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                            this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                            this->ptr_array_transposed_weights + *tmp_ptr_next_layer_neuron_it->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_next_layer_neuron_it->ptr_array_errors,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_last_neuron);
                
                this->Transpose_Layer_Backward__Batch_Normalization(layer_it);
                
                LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Reduce<var>,
                                                                  layer_it->ptr_dim3_grid_neurons_DP,
                                                                  layer_it->ptr_dim3_block_neurons_DP,
                                                                  0_UZ,
                                                                  tmp_number_neuron_units - 1, // Subtract bias.
                                                                  batch_size,
                                                                  this->batch_size,
                                                                  tmp_number_neuron_units,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)

                kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Final_Derivative<var> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (batch_size,
                                                                                                                                                                                                                                                                                                                                                tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                static_cast<var>(batch_size),
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                                                                tmp_ptr_layer_it_last_neuron);
            }
            //    2: Launching size condition.
            else if(ptr_dim3_batch_size_grid_received->x * ptr_dim3_batch_size_block_received->x > batch_size)
            {
                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative<var> <<< *ptr_dim3_batch_size_grid_received,
                                                                                                                                                                                                                            *ptr_dim3_batch_size_block_received,
                                                                                                                                                                                                                            ptr_dim3_batch_size_block_received->x * 3u * sizeof(var) >>> (batch_size,
                                                                                                                                                                                                                                                                                                                    this->total_parameters_allocated,
                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                    this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                    this->ptr_array_transposed_weights + *tmp_ptr_next_layer_neuron_it->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_next_layer_neuron_it->ptr_array_errors,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_last_neuron);
                
                this->Transpose_Layer_Backward__Batch_Normalization(layer_it);
                
                LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Reduce<var>,
                                                                  layer_it->ptr_dim3_grid_neurons_DP,
                                                                  layer_it->ptr_dim3_block_neurons_DP,
                                                                  0_UZ,
                                                                  tmp_number_neuron_units - 1, // Subtract bias.
                                                                  batch_size,
                                                                  this->batch_size,
                                                                  tmp_number_neuron_units,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)

                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Final_Derivative<var> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (batch_size,
                                                                                                                                                                                                                                                                                                                                       tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                       static_cast<var>(batch_size),
                                                                                                                                                                                                                                                                                                                                       tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                                                       tmp_ptr_layer_it_last_neuron);
            }
            //    3: Standard.
            else
            {
                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Summation_Derivative<var> <<< *ptr_dim3_batch_size_grid_received,
                                                                                                                                                                                                                            *ptr_dim3_batch_size_block_received,
                                                                                                                                                                                                                            ptr_dim3_batch_size_block_received->x * 3u * sizeof(var) >>> (this->total_parameters_allocated,
                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                    tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                    this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                    this->ptr_array_transposed_weights + *tmp_ptr_next_layer_neuron_it->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_next_layer_neuron_it->ptr_array_errors,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_last_neuron);
                
                this->Transpose_Layer_Backward__Batch_Normalization(layer_it);
                
                LAUNCH_KERNEL_POINTER_1D(Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Reduce<var>,
                                                                  layer_it->ptr_dim3_grid_neurons_DP,
                                                                  layer_it->ptr_dim3_block_neurons_DP,
                                                                  0_UZ,
                                                                  tmp_number_neuron_units - 1, // Subtract bias.
                                                                  batch_size,
                                                                  this->batch_size,
                                                                  tmp_number_neuron_units,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_mean,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_transposed_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_reduce_variance,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_threads,
                                                                  tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_threads)

                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Parallel_Batch__Serialize_Final_Derivative<var> <<< *ptr_dim3_batch_size_grid_received, *ptr_dim3_batch_size_block_received >>> (tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                       static_cast<var>(batch_size),
                                                                                                                                                                                                                                                                                                                                       tmp_ptr_layer_it_first_neuron,
                                                                                                                                                                                                                                                                                                                                       tmp_ptr_layer_it_last_neuron);
                
            }
            // |END| KERNEL LAUNCH |END|
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
                kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative<var> <<< *layer_it->ptr_dim3_grid_neurons_DP,
                                                                                                                                                                                                                                    *layer_it->ptr_dim3_block_neurons_DP,
                                                                                                                                                                                                                                    layer_it->ptr_dim3_block_neurons_DP->x * 3u * sizeof(var) >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                                    tmp_data_index,
                                                                                                                                                                                                                                                                                                                                                    tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                    tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                                                    this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                                                    this->ptr_array_transposed_weights + *tmp_ptr_next_layer_neuron_it->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_next_layer_neuron_it->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_scales,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_shifts,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                                                                                                                                                                                                                                                                                                    tmp_ptr_layer_it_first_neuron->ptr_r_correction,
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
                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative<var> <<< *layer_it->ptr_dim3_grid_neurons_DP,
                                                                                                                                                                                                                            *layer_it->ptr_dim3_block_neurons_DP,
                                                                                                                                                                                                                            layer_it->ptr_dim3_block_neurons_DP->x * 3u * sizeof(var) >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                            tmp_data_index,
                                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                                            this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                                            this->ptr_array_transposed_weights + *tmp_ptr_next_layer_neuron_it->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_next_layer_neuron_it->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_scales,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_shifts,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_r_correction,
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
                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Summation_Derivative<var> <<< *layer_it->ptr_dim3_grid_neurons_DP,
                                                                                                                                                                                                                            *layer_it->ptr_dim3_block_neurons_DP,
                                                                                                                                                                                                                            layer_it->ptr_dim3_block_neurons_DP->x * 3u * sizeof(var) >>> (tmp_data_index,
                                                                                                                                                                                                                                                                                                                                            tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_next_layer_number_neurons,
                                                                                                                                                                                                                                                                                                                                            this->neurons_total_reduce_error_size,
                                                                                                                                                                                                                                                                                                                                            this->ptr_array_transposed_weights + *tmp_ptr_next_layer_neuron_it->ptr_first_forward_connection_index,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_next_layer_neuron_it->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u), // Add bias.
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values_hats + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values_normalizes + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_values + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_reduce_error,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_scale,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_scales,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_shifts,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_r_correction,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_type_activation_function,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_grid_reduce_error,
                                                                                                                                                                                                                                                                                                                                            tmp_ptr_layer_it_first_neuron->ptr_array_dim3_block_reduce_error);
            }
        }
        // |END| KERNEL LAUNCH |END|
        
        // KERNEL LAUNCH
        //    1: Launching do-while elements.
        if(layer_it->ptr_dim3_grid_neurons->x * layer_it->ptr_dim3_block_neurons->x < tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a activation function.
                kernel_while__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative<var> <<< *layer_it->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                           *layer_it->ptr_dim3_block_neurons,
                                                                                                                                                                                                                           layer_it->ptr_dim3_block_neurons->x * 2u * sizeof(var) >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                                        tmp_data_index,
                                                                                                                                                                                                                                                                                                                                        static_cast<var>(batch_size),
                                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_r_correction);
            }
        }
        //    2: Launching size condition.
        else if(layer_it->ptr_dim3_grid_neurons->x * layer_it->ptr_dim3_block_neurons->x > tmp_number_neuron_units - 1u) // Subtract bias.
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a activation function.
                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative<var> <<< *layer_it->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                  *layer_it->ptr_dim3_block_neurons,
                                                                                                                                                                                                                  layer_it->ptr_dim3_block_neurons->x * 2u * sizeof(var) >>> (tmp_number_neuron_units - 1, // Subtract bias.
                                                                                                                                                                                                                                                                                                                        tmp_data_index,
                                                                                                                                                                                                                                                                                                                        static_cast<var>(batch_size),
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_r_correction);
            }
        }
        //    3: Standard.
        else
        {
            // Loop through each sample.
            for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
            {
                // Parallel each neurons for doing a activation function.
                kernel__Backward_Pass__FC_to_FC__Batch_Renormalization__Training__Serialize_Batch__Parallel_Final_Derivative<var> <<< *layer_it->ptr_dim3_grid_neurons,
                                                                                                                                                                                                                  *layer_it->ptr_dim3_block_neurons,
                                                                                                                                                                                                                  layer_it->ptr_dim3_block_neurons->x * 2u * sizeof(var) >>> (tmp_data_index,
                                                                                                                                                                                                                                                                                                                        static_cast<var>(batch_size),
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_errors + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_summations + tmp_data_index * tmp_number_neuron_units,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_means,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_variances,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_means,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_array_derivatives_variances,
                                                                                                                                                                                                                                                                                                                        tmp_ptr_layer_it_first_neuron->ptr_r_correction);
            }
        }
        // |END| KERNEL LAUNCH |END|
    }
    // If we don't enter into dynamic parallelisme, we serialize the computation.
    else
    {
        // Synchronize if needed to see the derivative output of the next layer.
        CUDA__Device_Synchronise(ref_synchronized_received, DL::ENUM_TYPE_DEVICE_SYNCHRONIZED::TYPE_DEVICE_SYNCHRONIZED_THREAD);

        // Loop through each sample.
        for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
        {
            // Loop through each neurons in the layer and do a reduction.
            for(tmp_ptr_next_layer_parameters = this->ptr_array_transposed_weights + *tmp_ptr_next_layer_neuron_it->ptr_first_forward_connection_index,
                tmp_ptr_next_layer_error = tmp_ptr_next_layer_neuron_it->ptr_array_errors + tmp_data_index * (tmp_next_layer_number_neurons + 1u), // Add bias.
                tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_layer_it_last_neuron; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                        tmp_ptr_next_layer_parameters += tmp_next_layer_number_neurons)
            {
                Reduce::Reduce_XX<var>(tmp_next_layer_number_neurons,
                                                      tmp_number_neuron_units,
                                                      *tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_data_index * this->neurons_total_reduce_error_size,
                                                      tmp_ptr_next_layer_parameters,
                                                      tmp_ptr_next_layer_error,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error,
                                                      tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error);
            }
            
            // Do we need to synchronise? Based on "Reduce::Reduce_XX" Function.
            // => Synchronisation before using the summed error of the layer.
            if(tmp_next_layer_number_neurons >= warpSize * 2u) { CUDA__Check_Error(); }

            // Loop through each neurons in the layer and do the begining derivation with a summation.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_layer_it_last_neuron; ++tmp_ptr_neuron_unit_it)
            {
                tmp_error = *(*tmp_ptr_neuron_unit_it->ptr_array_reduce_error + tmp_data_index * this->neurons_total_reduce_error_size);
                tmp_negate_r_correction = -*tmp_ptr_neuron_unit_it->ptr_d_correction; // Negate.
                tmp_variance_b = *tmp_ptr_neuron_unit_it->ptr_array_variances;

                // Derivative function.
                // dY *= dAF(value_normalize)
                tmp_error *= Activation_Derived(tmp_ptr_neuron_unit_it->ptr_array_values_normalizes[tmp_data_index * tmp_number_neuron_units],
                                                              tmp_ptr_neuron_unit_it->ptr_array_values[tmp_data_index * tmp_number_neuron_units],
                                                              *tmp_ptr_neuron_unit_it->ptr_type_activation_function);

                // Derivative scale.
                // dScale += dY * value_hat
                *tmp_ptr_neuron_unit_it->ptr_array_derivatives_scales += tmp_error * tmp_ptr_neuron_unit_it->ptr_array_values_hats[tmp_data_index * tmp_number_neuron_units];
            
                // Derivative shift.
                // dShift += dY
                *tmp_ptr_neuron_unit_it->ptr_array_derivatives_shifts += tmp_error;

                // Derivative value hat.
                // dX_h = dY * scale
                tmp_error *= *tmp_ptr_neuron_unit_it->ptr_scale;
            
                // dMean_b += dX_h * ( -r_correction / variance_b )
                *tmp_ptr_neuron_unit_it->ptr_array_derivatives_means += tmp_error * ( tmp_negate_r_correction / tmp_variance_b );

                // dVariance_b += dX_h * (summation - mean_b) * ( -r_correction / pow(variance_b, 2) )
                *tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances += tmp_error * (tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units] - *tmp_ptr_neuron_unit_it->ptr_array_means) * ( tmp_negate_r_correction / (tmp_variance_b * tmp_variance_b) );

                tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_data_index * tmp_number_neuron_units] = tmp_error;
            }
        }
    
        // Loop through each sample.
        for(tmp_data_index = 0_UZ; tmp_data_index != batch_size; ++tmp_data_index)
        {
            // Loop through each neurons in the layer and do the final derivation.
            for(tmp_ptr_neuron_unit_it = tmp_ptr_layer_it_first_neuron; tmp_ptr_neuron_unit_it != tmp_ptr_layer_it_last_neuron; ++tmp_ptr_neuron_unit_it)
            {
                tmp_error = tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_data_index * tmp_number_neuron_units];
                tmp_variance_b = *tmp_ptr_neuron_unit_it->ptr_array_variances;

                // First
                // dX_h *= r_correction / variance_b
                tmp_error *= *tmp_ptr_neuron_unit_it->ptr_d_correction / tmp_variance_b;
                
                // Middle
                // dX_h += dVariance_b * ( (summation - mean_b) / (N * variance_b) )
                tmp_error += *tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances * ( (tmp_ptr_neuron_unit_it->ptr_array_summations[tmp_data_index * tmp_number_neuron_units] - *tmp_ptr_neuron_unit_it->ptr_array_means) / (static_cast<var>(batch_size) * tmp_variance_b) );

                // Last
                // dX_h += dMean_b * 1 / N
                // dX_h += dMean_b / N
                tmp_error += *tmp_ptr_neuron_unit_it->ptr_array_derivatives_means / static_cast<var>(batch_size);

                // dX = dX_h
                tmp_ptr_neuron_unit_it->ptr_array_errors[tmp_data_index * tmp_number_neuron_units] = tmp_error;
            }
        }
    }
}
