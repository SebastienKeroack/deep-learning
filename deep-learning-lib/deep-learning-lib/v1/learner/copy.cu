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

#include <chrono>

template<typename T>
__global__ void kernel__cuModel__Copy_Neurons(size_t *const ptr_array_neuron_units_first_connection_index_destination_received,
                                                                                      size_t *const ptr_array_neuron_units_last_connection_index_destination_received,
                                                                                      size_t *const ptr_array_neuron_units_number_connections_destination_received,
                                                                                      size_t const *const ptr_array_neuron_units_first_connection_index_source_received,
                                                                                      size_t const *const ptr_array_neuron_units_last_connection_index_source_received,
                                                                                      enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS *const ptr_array_neuron_units_activation_function_destination_received,
                                                                                      enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_neuron_units_activation_function_source_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    size_t tmp_number_connections;

    ptr_array_neuron_units_last_connection_index_destination_received[tmp_thread_global_index] = tmp_number_connections = ptr_array_neuron_units_last_connection_index_source_received[tmp_thread_global_index];
        
    tmp_number_connections -= ptr_array_neuron_units_first_connection_index_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_first_connection_index_source_received[tmp_thread_global_index];
        
    ptr_array_neuron_units_number_connections_destination_received[tmp_thread_global_index] = tmp_number_connections;

    ptr_array_neuron_units_activation_function_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_activation_function_source_received[tmp_thread_global_index];
}

template<typename T>
__global__ void kernel__cuModel__Copy_Neurons(size_t const size_received,
                                                                                      size_t *const ptr_array_neuron_units_first_connection_index_destination_received,
                                                                                      size_t *const ptr_array_neuron_units_last_connection_index_destination_received,
                                                                                      size_t *const ptr_array_neuron_units_number_connections_destination_received,
                                                                                      size_t const *const ptr_array_neuron_units_first_connection_index_source_received,
                                                                                      size_t const *const ptr_array_neuron_units_last_connection_index_source_received,
                                                                                      enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS *const ptr_array_neuron_units_activation_function_destination_received,
                                                                                      enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_neuron_units_activation_function_source_received)
{
    size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);
    size_t tmp_number_connections;

    if(tmp_thread_global_index < size_received)
    {
        ptr_array_neuron_units_last_connection_index_destination_received[tmp_thread_global_index] = tmp_number_connections = ptr_array_neuron_units_last_connection_index_source_received[tmp_thread_global_index];
        
        tmp_number_connections -= ptr_array_neuron_units_first_connection_index_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_first_connection_index_source_received[tmp_thread_global_index];
        
        ptr_array_neuron_units_number_connections_destination_received[tmp_thread_global_index] = tmp_number_connections;

        ptr_array_neuron_units_activation_function_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_activation_function_source_received[tmp_thread_global_index];
    }
}

template<typename T>
__global__ void kernel_while__cuModel__Copy_Neurons(size_t const size_received,
                                                                                               size_t *const ptr_array_neuron_units_first_connection_index_destination_received,
                                                                                               size_t *const ptr_array_neuron_units_last_connection_index_destination_received,
                                                                                               size_t *const ptr_array_neuron_units_number_connections_destination_received,
                                                                                               size_t const *const ptr_array_neuron_units_first_connection_index_source_received,
                                                                                               size_t const *const ptr_array_neuron_units_last_connection_index_source_received,
                                                                                               enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS *const ptr_array_neuron_units_activation_function_destination_received,
                                                                                               enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *const ptr_array_neuron_units_activation_function_source_received)
{
    size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
                      tmp_number_connections;

    do
    {
        ptr_array_neuron_units_last_connection_index_destination_received[tmp_thread_global_index] = tmp_number_connections = ptr_array_neuron_units_last_connection_index_source_received[tmp_thread_global_index];
        
        tmp_number_connections -= ptr_array_neuron_units_first_connection_index_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_first_connection_index_source_received[tmp_thread_global_index];
        
        ptr_array_neuron_units_number_connections_destination_received[tmp_thread_global_index] = tmp_number_connections;

        ptr_array_neuron_units_activation_function_destination_received[tmp_thread_global_index] = ptr_array_neuron_units_activation_function_source_received[tmp_thread_global_index];

        tmp_thread_global_index += gridDim.x * blockDim.x;
    } while(tmp_thread_global_index < size_received);
}

__device__ void cuModel::Copy_Neurons(size_t const *ptr_array_neuron_units_first_connection_index_received,
                                                                                  size_t const *ptr_array_neuron_units_last_connection_index_received,
                                                                                  enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *ptr_array_neuron_units_activation_function_received,
                                                                                  struct cuNeuron *const ptr_array_copy_first_neuron_received,
                                                                                  struct cuNeuron *const ptr_array_copy_last_neuron_received)
{
    size_t const tmp_size(static_cast<size_t>(ptr_array_copy_last_neuron_received - ptr_array_copy_first_neuron_received));

    if(USE_PARALLEL && tmp_size >= warpSize)
    {
        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;
        
        this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(tmp_size,
                                                                                                                                                0,
                                                                                                                                                tmp_dim3_grid,
                                                                                                                                                tmp_dim3_block);

        LAUNCH_KERNEL_1D(cuModel__Copy_Neurons<var>,
                                          tmp_dim3_grid,
                                          tmp_dim3_block,
                                          0_UZ,
                                          tmp_size,
                                          ptr_array_copy_first_neuron_received->ptr_first_forward_connection_index,
                                          ptr_array_copy_first_neuron_received->ptr_last_forward_connection_index,
                                          ptr_array_copy_first_neuron_received->ptr_number_forward_connections,
                                          ptr_array_neuron_units_first_connection_index_received,
                                          ptr_array_neuron_units_last_connection_index_received,
                                          ptr_array_copy_first_neuron_received->ptr_type_activation_function,
                                          ptr_array_neuron_units_activation_function_received)

        CUDA__Check_Error();
    }
    else
    {
        for(struct cuNeuron *tmp_ptr_neuron_unit_it(ptr_array_copy_first_neuron_received); tmp_ptr_neuron_unit_it != ptr_array_copy_last_neuron_received; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                                                                                 ++ptr_array_neuron_units_first_connection_index_received,
                                                                                                                                                                                                                                 ++ptr_array_neuron_units_last_connection_index_received,
                                                                                                                                                                                                                                 ++ptr_array_neuron_units_activation_function_received)
        {
            this->Copy__Neuron_Unit(tmp_ptr_neuron_unit_it,
                                         *ptr_array_neuron_units_first_connection_index_received,
                                         *ptr_array_neuron_units_last_connection_index_received,
                                         *ptr_array_neuron_units_activation_function_received);
        }
    }

    // Prepare grids and blocks dimensions.
    this->Prepare__Layers__Grids_Blocks_Dimensions();
    this->Prepare__Neurons__Grids_Blocks_Dimensions();

    this->Prepare__Batch_Layers__Grids_Blocks_Dimensions(this->batch_size);
    // |END| Prepare grids and blocks dimensions. |END|
}

__device__ void inline cuModel::Copy__Neuron_Unit(struct cuNeuron *const ptr_copy_neuron_received,
                                                                                size_t const neuron_first_connection_index_received,
                                                                                size_t const neuron_last_connection_index_received,
                                                                                enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const neuron_activation_function_received)
{
    *ptr_copy_neuron_received->ptr_first_forward_connection_index = neuron_first_connection_index_received;
    *ptr_copy_neuron_received->ptr_last_forward_connection_index = neuron_last_connection_index_received;
    *ptr_copy_neuron_received->ptr_number_forward_connections = neuron_last_connection_index_received - neuron_first_connection_index_received;

    *ptr_copy_neuron_received->ptr_type_activation_function = neuron_activation_function_received;
}

__device__ void cuModel::Copy__FC_to_FC(struct cuNeuron *ptr_copy_neuron_it_received,
                                                                                        struct cuNeuron const *const ptr_copy_last_neuron_received,
                                                                                        struct cuNeuron *const ptr_copy_first_neuron_received,
                                                                                        size_t const *&ptr_array_neuron_units_first_connection_index_received,
                                                                                        size_t const *&ptr_array_neuron_units_last_connection_index_received,
                                                                                        enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *&ptr_array_neuron_units_activation_function_received)
{
    for(; ptr_copy_neuron_it_received != ptr_copy_last_neuron_received; ++ptr_copy_neuron_it_received)
    {
        this->Copy__Neuron_Unit(ptr_copy_neuron_it_received,
                                     *ptr_array_neuron_units_first_connection_index_received++,
                                     *ptr_array_neuron_units_last_connection_index_received++,
                                     *ptr_array_neuron_units_activation_function_received++);
    }
}

template<typename T>
__global__ void kernel__cuModel__Copy__Optimizer_Gradient_Descent__Host_To_Device(T const optimizer_time_step_received,
                                                                                                                                                T const warm_restarts_maximum_learning_rate_received,
                                                                                                                                                T const warm_restarts_T_i_received,
                                                                                                                                                T *const ptr_array_previous_delta_parameters_received,
                                                                                                                                                class cuModel *const ptr_cuModel_received)
{
    ptr_cuModel_received->optimizer_time_step = optimizer_time_step_received;
    ptr_cuModel_received->warm_restarts_maximum_learning_rate = warm_restarts_maximum_learning_rate_received;
    ptr_cuModel_received->warm_restarts_T_i = warm_restarts_T_i_received;
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                                   ptr_cuModel_received->ptr_array_previous_delta_parameters,
                                                   ptr_array_previous_delta_parameters_received,
                                                   ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                                   ptr_cuModel_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Model::Copy__Optimizer_Gradient_Descent__Host_To_Device(T const optimizer_time_step_received,
                                                                                                                T const warm_restarts_maximum_learning_rate_received,
                                                                                                                T const warm_restarts_T_i_received,
                                                                                                                T const *const ptr_array_previous_delta_parameters_received)
{
    T *tmp_ptr_device_array_previous_delta_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_delta_parameters, this->total_parameters * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_delta_parameters,
                                                    ptr_array_previous_delta_parameters_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__cuModel__Copy__Optimizer_Gradient_Descent__Host_To_Device<T> <<< 1, 1 >>> (optimizer_time_step_received,
                                                                                                                                                            warm_restarts_maximum_learning_rate_received,
                                                                                                                                                            warm_restarts_T_i_received,
                                                                                                                                                            tmp_ptr_device_array_previous_delta_parameters,
                                                                                                                                                            this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_delta_parameters)); // T

    return true;
}
template bool Model::Copy__Optimizer_Gradient_Descent__Host_To_Device(var const,
                                                                                                                              var const,
                                                                                                                              var const,
                                                                                                                              var const *const);

template<typename T>
__global__ void kernel__cuModel__Copy__Optimizer_RPROP_minus__Host_To_Device(T *const ptr_array_previous_steps_received,
                                                                                                                                            T *const ptr_array_previous_derivates_parameters_received,
                                                                                                                                            class cuModel *const ptr_cuModel_received)
{
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_cuModel_received->ptr_array_previous_steps,
                                             ptr_array_previous_steps_received,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_cuModel_received->ptr_array_previous_derivatives_parameters,
                                             ptr_array_previous_derivates_parameters_received,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Model::Copy__Optimizer_RPROP_minus__Host_To_Device(T const *const ptr_array_previous_steps_received, T const *const ptr_array_previous_derivates_parameters_received)
{
    T *tmp_ptr_device_array_previous_steps,
        *tmp_ptr_device_array_previous_derivatives_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_steps, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_derivatives_parameters, this->total_parameters * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_steps,
                                                    ptr_array_previous_steps_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_derivatives_parameters,
                                                    ptr_array_previous_derivates_parameters_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__cuModel__Copy__Optimizer_RPROP_minus__Host_To_Device<T> <<< 1, 1 >>> (tmp_ptr_device_array_previous_steps,
                                                                                                                                                tmp_ptr_device_array_previous_derivatives_parameters,
                                                                                                                                                this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_steps)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_derivatives_parameters)); // T

    return true;
}
template bool Model::Copy__Optimizer_RPROP_minus__Host_To_Device(var const *const, var const *const);

template<typename T>
__global__ void kernel__cuModel__Copy__Optimizer_RPROP_plus__Host_To_Device(T const loss_received,
                                                                                                                                        T const previous_loss_received,
                                                                                                                                        T *const ptr_array_previous_steps_received,
                                                                                                                                        T *const ptr_array_previous_derivates_parameters_received,
                                                                                                                                        T *const ptr_array_previous_delta_parameters_received,
                                                                                                                                        class cuModel *const ptr_cuModel_received)
{
    ptr_cuModel_received->loss_rprop = loss_received;
    ptr_cuModel_received->loss_rprop_tm1 = previous_loss_received;
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_cuModel_received->ptr_array_previous_steps,
                                             ptr_array_previous_steps_received,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);

    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_cuModel_received->ptr_array_previous_derivatives_parameters,
                                             ptr_array_previous_derivates_parameters_received,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);

    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_cuModel_received->ptr_array_previous_delta_parameters,
                                             ptr_array_previous_delta_parameters_received,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Model::Copy__Optimizer_RPROP_plus__Host_To_Device(T const loss_received,
                                                                                                                    T const previous_loss_received,
                                                                                                                    T const *const ptr_array_previous_steps_received,
                                                                                                                    T const *const ptr_array_previous_derivates_parameters_received,
                                                                                                                    T const *const ptr_array_previous_delta_parameters_received)
{
    T *tmp_ptr_device_array_previous_steps,
        *tmp_ptr_device_array_previous_derivatives_parameters,
        *tmp_ptr_device_array_previous_delta_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_steps, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_derivatives_parameters, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_delta_parameters, this->total_parameters * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_steps,
                                                    ptr_array_previous_steps_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_derivatives_parameters,
                                                    ptr_array_previous_derivates_parameters_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_delta_parameters,
                                                    ptr_array_previous_delta_parameters_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__cuModel__Copy__Optimizer_RPROP_plus__Host_To_Device<T> <<< 1, 1 >>> (loss_received,
                                                                                                                                                previous_loss_received,
                                                                                                                                                tmp_ptr_device_array_previous_steps,
                                                                                                                                                tmp_ptr_device_array_previous_derivatives_parameters,
                                                                                                                                                tmp_ptr_device_array_previous_delta_parameters,
                                                                                                                                                this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_steps)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_derivatives_parameters)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_delta_parameters)); // T

    return true;
}
template bool Model::Copy__Optimizer_RPROP_plus__Host_To_Device(var const,
                                                                                                                                var const,
                                                                                                                                var const *const,
                                                                                                                                var const *const,
                                                                                                                                var const *const);

template<typename T>
__global__ void kernel__cuModel__Copy__Optimizer_Adam__Host_To_Device(T const optimizer_time_step_received,
                                                                                                                               T const warm_restarts_maximum_learning_rate_received,
                                                                                                                               T const warm_restarts_T_i_received,
                                                                                                                               T *const ptr_array_previous_biased_first_moment_received,
                                                                                                                               T *const ptr_array_previous_biased_second_moment_received,
                                                                                                                               class cuModel *const ptr_cuModel_received)
{
    ptr_cuModel_received->optimizer_time_step = optimizer_time_step_received;
    ptr_cuModel_received->warm_restarts_maximum_learning_rate = warm_restarts_maximum_learning_rate_received;
    ptr_cuModel_received->warm_restarts_T_i = warm_restarts_T_i_received;

    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_cuModel_received->ptr_array_previous_biased_first_moment,
                                             ptr_array_previous_biased_first_moment_received,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);

    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_cuModel_received->ptr_array_previous_biased_second_moment,
                                             ptr_array_previous_biased_second_moment_received,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Model::Copy__Optimizer_Adam__Host_To_Device(T const optimizer_time_step_received,
                                                                                               T const warm_restarts_maximum_learning_rate_received,
                                                                                               T const warm_restarts_T_i_received,
                                                                                               T const *const ptr_array_previous_biased_first_moment_received,
                                                                                               T const *const ptr_array_previous_biased_second_moment_received)
{
    T *tmp_ptr_device_array_previous_biased_first_moment,
        *tmp_ptr_device_array_previous_biased_second_moment;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_first_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_moment, this->total_parameters * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_biased_first_moment,
                                                    ptr_array_previous_biased_first_moment_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_biased_second_moment,
                                                    ptr_array_previous_biased_second_moment_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__cuModel__Copy__Optimizer_Adam__Host_To_Device<T> <<< 1, 1 >>> (optimizer_time_step_received,
                                                                                                                                            warm_restarts_maximum_learning_rate_received,
                                                                                                                                            warm_restarts_T_i_received,
                                                                                                                                            tmp_ptr_device_array_previous_biased_first_moment,
                                                                                                                                            tmp_ptr_device_array_previous_biased_second_moment,
                                                                                                                                            this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_first_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_moment)); // T

    return true;
}
template bool Model::Copy__Optimizer_Adam__Host_To_Device(var const,
                                                                                                            var const,
                                                                                                            var const,
                                                                                                            var const *const,
                                                                                                            var const *const);

template<typename T>
__global__ void kernel__cuModel__Copy__Optimizer_AMSGrad__Host_To_Device(T const optimizer_time_step_received,
                                                                                                                                    T const warm_restarts_maximum_learning_rate_received,
                                                                                                                                    T const warm_restarts_T_i_received,
                                                                                                                                    T *const ptr_array_previous_biased_first_moment_received,
                                                                                                                                    T *const ptr_array_previous_biased_second_moment_received,
                                                                                                                                    T *const ptr_array_previous_biased_second_moment_hat_received,
                                                                                                                                    class cuModel *const ptr_cuModel_received)
{
    ptr_cuModel_received->optimizer_time_step = optimizer_time_step_received;
    ptr_cuModel_received->warm_restarts_maximum_learning_rate = warm_restarts_maximum_learning_rate_received;
    ptr_cuModel_received->warm_restarts_T_i = warm_restarts_T_i_received;

    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_cuModel_received->ptr_array_previous_biased_first_moment,
                                             ptr_array_previous_biased_first_moment_received,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);

    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_cuModel_received->ptr_array_previous_biased_second_moment,
                                             ptr_array_previous_biased_second_moment_received,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);

    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_cuModel_received->ptr_array_previous_biased_second_moment_hat,
                                             ptr_array_previous_biased_second_moment_hat_received,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Model::Copy__Optimizer_AMSGrad__Host_To_Device(T const optimizer_time_step_received,
                                                                                                    T const warm_restarts_maximum_learning_rate_received,
                                                                                                    T const warm_restarts_T_i_received,
                                                                                                    T const *const ptr_array_previous_biased_first_moment_received,
                                                                                                    T const *const ptr_array_previous_biased_second_moment_received,
                                                                                                    T const *const ptr_array_previous_biased_second_moment_hat_received)
{
    T *tmp_ptr_device_array_previous_biased_first_moment,
        *tmp_ptr_device_array_previous_biased_second_moment,
        *tmp_ptr_device_array_previous_biased_second_hat_moment;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_first_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_hat_moment, this->total_parameters * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_biased_first_moment,
                                                    ptr_array_previous_biased_first_moment_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_biased_second_moment,
                                                    ptr_array_previous_biased_second_moment_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_previous_biased_second_hat_moment,
                                                    ptr_array_previous_biased_second_moment_hat_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__cuModel__Copy__Optimizer_AMSGrad__Host_To_Device<T> <<< 1, 1 >>> (optimizer_time_step_received,
                                                                                                                                                warm_restarts_maximum_learning_rate_received,
                                                                                                                                                warm_restarts_T_i_received,
                                                                                                                                                tmp_ptr_device_array_previous_biased_first_moment,
                                                                                                                                                tmp_ptr_device_array_previous_biased_second_moment,
                                                                                                                                                tmp_ptr_device_array_previous_biased_second_hat_moment,
                                                                                                                                                this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_first_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_hat_moment)); // T

    return true;
}
template bool Model::Copy__Optimizer_AMSGrad__Host_To_Device(var const,
                                                                                                                  var const,
                                                                                                                  var const,
                                                                                                                  var const *const,
                                                                                                                  var const *const,
                                                                                                                  var const *const);

template<typename T>
__global__ void kernel__cuModel__Copy__Batch_Normalization_Neurons__Host_To_Device(T *const ptr_array_neuron_units_scale_received,
                                                                                                                                                  T *const ptr_array_neuron_units_shift_received,
                                                                                                                                                  T *const ptr_array_neuron_units_mean_average_received,
                                                                                                                                                  T *const ptr_array_neuron_units_variance_average_received,
                                                                                                                                                  class cuModel *const ptr_cuModel_received)
{
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_neuron_units,
                                                 ptr_cuModel_received->ptr_array_normalized_batch_units_scales,
                                                 ptr_array_neuron_units_scale_received,
                                                 ptr_cuModel_received->ptr_array_dim3_grid + 3,
                                                 ptr_cuModel_received->ptr_array_dim3_block + 3);

    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_neuron_units,
                                                 ptr_cuModel_received->ptr_array_normalized_batch_units_shifts,
                                                 ptr_array_neuron_units_shift_received,
                                                 ptr_cuModel_received->ptr_array_dim3_grid + 3,
                                                 ptr_cuModel_received->ptr_array_dim3_block + 3);

    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_neuron_units,
                                                 ptr_cuModel_received->ptr_array_normalized_batch_units_means_averages,
                                                 ptr_array_neuron_units_mean_average_received,
                                                 ptr_cuModel_received->ptr_array_dim3_grid + 3,
                                                 ptr_cuModel_received->ptr_array_dim3_block + 3);

    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_neuron_units,
                                                 ptr_cuModel_received->ptr_array_normalized_batch_units_variances_averages,
                                                 ptr_array_neuron_units_variance_average_received,
                                                 ptr_cuModel_received->ptr_array_dim3_grid + 3,
                                                 ptr_cuModel_received->ptr_array_dim3_block + 3);
}

template<typename T>
bool Model::Copy__Batch_Normalization_Neurons__Host_To_Device(T const *const ptr_array_neuron_units_scale_received,
                                                                                                                            T const *const ptr_array_neuron_units_shift_received,
                                                                                                                            T const *const ptr_array_neuron_units_mean_average_received,
                                                                                                                            T const *const ptr_array_neuron_units_variance_average_received) const
{
    T *tmp_ptr_device_array_neurons_scale(NULL),
       *tmp_ptr_device_array_neurons_shift(NULL),
       *tmp_ptr_device_array_neurons_mean_average(NULL),
       *tmp_ptr_device_array_neurons_variance_average(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_scale, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_shift, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_mean_average, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_variance_average, this->total_neuron_units * sizeof(T)));
        
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_neurons_scale,
                                                    ptr_array_neuron_units_scale_received,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_neurons_shift,
                                                    ptr_array_neuron_units_shift_received,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_neurons_mean_average,
                                                    ptr_array_neuron_units_mean_average_received,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_neurons_variance_average,
                                                    ptr_array_neuron_units_variance_average_received,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__cuModel__Copy__Batch_Normalization_Neurons__Host_To_Device<T> <<< 1, 1 >>> (tmp_ptr_device_array_neurons_scale,
                                                                                                                                                              tmp_ptr_device_array_neurons_shift,
                                                                                                                                                              tmp_ptr_device_array_neurons_mean_average,
                                                                                                                                                              tmp_ptr_device_array_neurons_variance_average,
                                                                                                                                                              this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_scale)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_shift)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_mean_average)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_variance_average)); // T

    return true;
}
template bool Model::Copy__Batch_Normalization_Neurons__Host_To_Device(var const *const,
                                                                                                                                        var const *const,
                                                                                                                                        var const *const,
                                                                                                                                        var const *const) const;

__Lch_Bds__(MAXIMUM_THREADS_PER_BLOCK, 1)
__global__ void kernel__cuModel__Copy__Host_To_Device(size_t const *ptr_array_number_neurons_by_layer_received,
                                                                                                   size_t const *ptr_array_neuron_units_first_connection_index_received,
                                                                                                   size_t const *ptr_array_neuron_units_last_connection_index_received,
                                                                                                   size_t const *ptr_array_neuron_units_bias_index_received,
                                                                                                   size_t const number_loss_received,
                                                                                                   size_t const number_bit_fail_received,
                                                                                                   var const loss_values_received,
                                                                                                   var const *ptr_array_accuracy_value_received,
                                                                                                   var const *ptr_array_dropout_value_by_layer_received,
                                                                                                   var const *ptr_array_parameters_received,
                                                                                                   DL::LAYER::TYPE const *ptr_array_type_layer_received,
                                                                                                   DL::LAYER_ACTIVATION::TYPE const *ptr_array_type_activation_received,
                                                                                                   DL::LAYER_DROPOUT::TYPE const *ptr_array_type_dropout_by_layer_received,
                                                                                                   DL::LAYER_NORM::TYPE const *ptr_array_tpye_normalization_by_layer_received,
                                                                                                   enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *ptr_array_neuron_units_type_activation_function_received,
                                                                                                   class Model const *const model,
                                                                                                   class cuModel *const ptr_cuModel_received)
{

    size_t const *tmp_ptr_array_number_neurons(ptr_array_number_neurons_by_layer_received),
                                *tmp_ptr_array_neuron_units_first_connection_index(ptr_array_neuron_units_first_connection_index_received),
                                *tmp_ptr_array_neuron_units_last_connection_index(ptr_array_neuron_units_last_connection_index_received);

    DL::LAYER::TYPE const *tmp_ptr_array_type_layer(ptr_array_type_layer_received);
    DL::LAYER_ACTIVATION::TYPE const *tmp_ptr_array_type_activation(ptr_array_type_activation_received);
    enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const *tmp_ptr_array_neuron_units_activation_function(ptr_array_neuron_units_type_activation_function_received);
    DL::LAYER_DROPOUT::TYPE const *tmp_ptr_array_type_dropout_by_layer_received(ptr_array_type_dropout_by_layer_received);
    DL::LAYER_NORM::TYPE const *tmp_ptr_array_type_normalization_by_layer_received(ptr_array_tpye_normalization_by_layer_received);

    var const *tmp_ptr_array_dropout_value_by_layers(
        ptr_array_dropout_value_by_layer_received);
        
    // General parameters.
    ptr_cuModel_received->type = model->type;
    ptr_cuModel_received->connection_rate = model->connection_rate;
    ptr_cuModel_received->seq_w = model->seq_w;
    ptr_cuModel_received->n_time_delay = model->n_time_delay;
    // |END| General parameters. |END|

    // Gradient descent parameters.
    ptr_cuModel_received->learning_rate = model->learning_rate;
    ptr_cuModel_received->learning_momentum = model->learning_momentum;
    ptr_cuModel_received->use_nesterov = model->use_nesterov;
    // |END| Gradient descent parameters. |END|
        
    // Quickprop parameters.
    ptr_cuModel_received->quickprop_decay = model->quickprop_decay;
    ptr_cuModel_received->quickprop_mu = model->quickprop_mu;
    // |END| Quickprop parameters. |END|

    // Resillent propagation parameters.
    ptr_cuModel_received->rprop_increase_factor = model->rprop_increase_factor;
    ptr_cuModel_received->rprop_decrease_factor = model->rprop_decrease_factor;
    ptr_cuModel_received->rprop_delta_min = model->rprop_delta_min;
    ptr_cuModel_received->rprop_delta_max = model->rprop_delta_max;
    ptr_cuModel_received->rprop_delta_zero = model->rprop_delta_zero;
    ptr_cuModel_received->loss_rprop = model->loss_rprop;
    ptr_cuModel_received->loss_rprop_tm1 = model->loss_rprop_tm1;
    // |END| Resillent propagation parameters. |END|
        
    // SARProp parameters.
    ptr_cuModel_received->sarprop_weight_decay_shift = model->sarprop_weight_decay_shift;
    ptr_cuModel_received->sarprop_step_error_threshold_factor = model->sarprop_step_error_threshold_factor;
    ptr_cuModel_received->sarprop_step_error_shift = model->sarprop_step_error_shift;
    ptr_cuModel_received->sarprop_temperature = model->sarprop_temperature;
    ptr_cuModel_received->sarprop_epoch = model->sarprop_epoch;
    // |END| SARProp parameters. |END|
        
    // Adam parameters.
     ptr_cuModel_received->adam_learning_rate = model->adam_learning_rate;
     ptr_cuModel_received->adam_beta1 = model->adam_beta1;
     ptr_cuModel_received->adam_beta2 = model->adam_beta2;
     ptr_cuModel_received->adam_epsilon = model->adam_epsilon;
     ptr_cuModel_received->use_adam_bias_correction = model->use_adam_bias_correction;
     ptr_cuModel_received->adam_gamma = model->adam_gamma;
    // |END| Adam parameters. |END|

    // Loss parameters.
    *ptr_cuModel_received->ptr_array_number_loss = number_loss_received;
    *ptr_cuModel_received->ptr_array_number_bit_fail = number_bit_fail_received;
    *ptr_cuModel_received->ptr_array_loss_values = loss_values_received;
    ptr_cuModel_received->loss_train = model->loss_train;
    ptr_cuModel_received->loss_valid = model->loss_valid;
    ptr_cuModel_received->loss_testg = model->loss_testg;
    // |END| Loss parameters. |END|
        
    // Accuracy parameters.
    *ptr_cuModel_received->ptr_array_accuracy_values[0] = ptr_array_accuracy_value_received[0];
    *ptr_cuModel_received->ptr_array_accuracy_values[1] = ptr_array_accuracy_value_received[1];
    *ptr_cuModel_received->ptr_array_accuracy_values[2] = ptr_array_accuracy_value_received[2];
    *ptr_cuModel_received->ptr_array_accuracy_values[3] = ptr_array_accuracy_value_received[3];
    *ptr_cuModel_received->ptr_array_accuracy_values[4] = ptr_array_accuracy_value_received[4];
    ptr_cuModel_received->n_acc_trial = model->n_acc_trial;
    ptr_cuModel_received->acc_var = model->acc_var;
    ptr_cuModel_received->acc_train = model->acc_train;
    ptr_cuModel_received->acc_valid = model->acc_valid;
    ptr_cuModel_received->acc_testg = model->acc_testg;
    // |END| Accuracy parameters. |END|

    // Dimension.
    ptr_cuModel_received->n_inp = model->n_inp;
    ptr_cuModel_received->n_out = model->n_out;
    ptr_cuModel_received->total_neuron_units = model->total_neuron_units;
    ptr_cuModel_received->total_block_units = model->total_block_units;
    ptr_cuModel_received->total_cell_units = model->total_cell_units;
    ptr_cuModel_received->total_parameters = ptr_cuModel_received->total_weights = model->total_weights;
    
    // Prepare grids and blocks dimensions.
    ptr_cuModel_received->Prepare__Global__Grids_Blocks_Dimensions();
    // |END| Prepare grids and blocks dimensions. |END|

    struct cuLayer const *const last_layer(ptr_cuModel_received->ptr_last_layer);
    struct cuLayer *layer_it(ptr_cuModel_received->ptr_array_layers);

    for(; layer_it != last_layer; ++layer_it,
                                                                    ++tmp_ptr_array_type_layer,
                                                                    ++tmp_ptr_array_type_activation,
                                                                    ++tmp_ptr_array_number_neurons)
    {
        layer_it->type_layer = *tmp_ptr_array_type_layer;
        layer_it->type_activation = *tmp_ptr_array_type_activation;
        
        // Neuron_unit.
        *layer_it->ptr_number_neurons = *tmp_ptr_array_number_neurons;

        layer_it->ptr_last_neuron_unit = layer_it->ptr_array_neuron_units + *tmp_ptr_array_number_neurons;
        // |END| Neuron_unit. |END|

        // LSTM block.
        //layer_it->ptr_last_block_unit = layer_it->ptr_array_block_units + static_cast<size_t>(tmp_ptr_original_layer_it->ptr_last_block_unit - tmp_ptr_original_layer_it->ptr_array_block_units);
        // |END| LSTM block. |END|
        
        // LSTM cell.
        //layer_it->ptr_last_cell_unit = layer_it->ptr_array_cell_units + static_cast<size_t>(tmp_ptr_original_layer_it->ptr_last_cell_unit - tmp_ptr_original_layer_it->ptr_array_cell_units);
        // |END| LSTM cell. |END|
    }
    // |END| Dimension. |END|
    
    // allocate reduce batch.
    if(ptr_cuModel_received->Allocate_Reduce_Threads() == false)
    {
        ERR(L"From \"Allocate_Reduce_Threads\".",);

        return;
    }
    // |END| allocate reduce batch. |END|

    // allocate reduce cost.
    if(ptr_cuModel_received->Allocate_Reduce_Cost() == false)
    {
        ERR(L"From \"Allocate_Reduce_Cost\".",);

        return;
    }
    // |END| allocate reduce cost. |END|

    // allocate neurons.
    if(ptr_cuModel_received->Allocate__Neuron_Units() == false)
    {
        ERR(L"From \"Allocate__Neuron_Units\".",);

        return;
    }
    // |END| allocate neurons. |END|
    
    // allocate connections.
    if(ptr_cuModel_received->Allocate__Parameter() == false)
    {
        ERR(L"From \"Allocate__Parameter\".",);

        return;
    }
    // |END| allocate connections. |END|

    // copy connections.
    struct cuLayer *const tmp_ptr_first_layer(ptr_cuModel_received->ptr_array_layers);

    ptr_cuModel_received->Copy_Neurons(tmp_ptr_array_neuron_units_first_connection_index,
                                                                         tmp_ptr_array_neuron_units_last_connection_index,
                                                                         tmp_ptr_array_neuron_units_activation_function,
                                                                         tmp_ptr_first_layer->ptr_array_neuron_units,
                                                                         tmp_ptr_first_layer->ptr_array_neuron_units + ptr_cuModel_received->total_neuron_units);
    // |END| copy dimension. |END|    
    
    // allocate neurons reduce summation.
    if(ptr_cuModel_received->Allocate__Neurons_Reduce_Summation() == false)
    {
        ERR(L"From \"Allocate__Neurons_Reduce_Summation\".",);

        return;
    }
    // |END| allocate neurons reduce summation. |END|
    
    // Dropout.
    for(layer_it = tmp_ptr_first_layer; layer_it != last_layer; ++layer_it,
                                                                                                                         ++tmp_ptr_array_dropout_value_by_layers,
                                                                                                                         ++tmp_ptr_array_type_dropout_by_layer_received)
    {
        ptr_cuModel_received->Set__Probability_Retained_Unit(layer_it,
                                                                                                   *tmp_ptr_array_type_dropout_by_layer_received == DL::LAYER_DROPOUT::BERNOULLI ? *tmp_ptr_array_dropout_value_by_layers : 1_r,
                                                                                                   false);
    }
    // |END| Dropout. |END|

    // Batch renormalization.
    for(++tmp_ptr_array_type_normalization_by_layer_received, // Skip input layer.
        layer_it = tmp_ptr_first_layer + 1; layer_it != last_layer - 1; ++layer_it,
                                                                                                                                    ++tmp_ptr_array_type_normalization_by_layer_received)
    { ptr_cuModel_received->Set__Batch_Renormalization(layer_it, *tmp_ptr_array_type_normalization_by_layer_received == DL::LAYER_NORM::BATCH_RENORMALIZATION); }
    // |END| Batch renormalization. |END|
    
    // Assign connections.
    Memory::Memory_Copy_1D<var>(ptr_cuModel_received->total_parameters,
                                                     ptr_cuModel_received->ptr_array_parameters,
                                                     ptr_array_parameters_received,
                                                     ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                                     ptr_cuModel_received->ptr_array_dim3_block + 1);
    // |END| Assign connections. |END|

    // allocate transposed weights.
    // TODO: allocate only at training.
    if(ptr_cuModel_received->Allocate_Weights_Transposed() == false)
    {
        ERR(L"From \"Allocate_Weights_Transposed\".",);

        return;
    }
    // |END| allocate transposed weights. |END|

    // allocate derivative parameters.
    // TODO: allocate only at training.
    ptr_cuModel_received->device__Clear_Train_Arrays();
    // |END| allocate derivative parameters. |END|

    // allocate neurons reduce error.
    // TODO: allocate only at training.
    if(ptr_cuModel_received->Allocate__Neurons_Reduce_Error() == false)
    {
        ERR(L"From \"Allocate__Neurons_Reduce_Error\".",);

        return;
    }
    // |END| allocate neurons reduce error. |END|
        
    // Warm restarts parameters.
    ptr_cuModel_received->use_warm_restarts = model->use_warm_restarts;
    ptr_cuModel_received->warm_restarts_decay_learning_rate = model->warm_restarts_decay_learning_rate;
    ptr_cuModel_received->warm_restarts_maximum_learning_rate = ptr_cuModel_received->warm_restarts_initial_maximum_learning_rate = model->warm_restarts_initial_maximum_learning_rate;
    ptr_cuModel_received->warm_restarts_minimum_learning_rate = model->warm_restarts_minimum_learning_rate;
    ptr_cuModel_received->warm_restarts_T_i = ptr_cuModel_received->warm_restarts_initial_T_i = model->warm_restarts_initial_T_i;
    ptr_cuModel_received->warm_restarts_multiplier = model->warm_restarts_multiplier;
    // |END| Warm restarts parameters. |END|

    // Training parameters.
    ptr_cuModel_received->set_optimizer(model->type_optimizer_function);
    ptr_cuModel_received->set_loss_fn(model->type_loss_function);
    ptr_cuModel_received->bit_fail_limit = model->bit_fail_limit;
    ptr_cuModel_received->optimizer_time_step = model->optimizer_time_step;
    ptr_cuModel_received->epoch_time_step = model->epoch_time_step;
    // |END| Training parameters. |END|

    // Regularization parameters.
    ptr_cuModel_received->Set__Regularization__Max_Norm_Constraints(model->regularization__max_norm_constraints);
    ptr_cuModel_received->Set__Regularization__L1(model->regularization__l1);
    ptr_cuModel_received->Set__Regularization__L2(model->regularization__l2);
    ptr_cuModel_received->Set__Regularization__Weight_Decay(model->weight_decay);
    // |END| Regularization parameters. |END|

    // Regularization parameters.
    ptr_cuModel_received->Set__Normalization_Momentum_Average(model->normalization_momentum_average);
    ptr_cuModel_received->Set__Normalization_Epsilon(model->normalization_epsilon);
    ptr_cuModel_received->Set__Batch_Renormalization_r_Correction_Maximum(model->batch_renormalization_r_correction_maximum);
    ptr_cuModel_received->Set__Batch_Renormalization_d_Correction_Maximum(model->batch_renormalization_d_correction_maximum);
    // |END| Regularization parameters. |END|

    // TODO: Transpose only on allocation of \"Allocate_Weights_Transposed\".
    ptr_cuModel_received->Transpose_Weights();
}

__host__ bool cuModel::Copy__Host_To_Device(class Model const *const ptr_host_Neural_Network_received, size_t const allowable_memory)
{
    if(ptr_host_Neural_Network_received == NULL)
    {
        ERR(L"Host pointer source is a nullptr.",);

        return false;
    }

    if(this->Allocate__Structure(ptr_host_Neural_Network_received->total_layers, allowable_memory) == false)
    {
        ERR(L"An error has been triggered from the \"Allocate__Structure(%zu)\" function.",
                                 allowable_memory);

        return false;
    }

    size_t *tmp_ptr_device_array_number_neurons_by_layer,
              *tmp_ptr_device_array_neurons_first_connection_index,
              *tmp_ptr_device_array_neurons_last_connection_index,
              *tmp_ptr_device_array_neurons_bias_index;
        
    DL::LAYER::TYPE *tmp_ptr_device_array_type_layer;
    DL::LAYER_ACTIVATION::TYPE *tmp_ptr_device_array_type_activation;
    DL::LAYER_NORM::TYPE *tmp_ptr_device_array_type_normalization_by_layer;
    DL::LAYER_DROPOUT::TYPE *tmp_ptr_device_array_type_dropout_by_layer;
    enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS *tmp_ptr_device_array_neurons_type_activation_function;

    var *tmp_ptr_device_array_accuracy_values,
         *tmp_ptr_device_array_value_dropout_by_layer,
         *tmp_ptr_device_array_parameters;
        
    class Model *tmp_ptr_device_original_Neural_Network;

    // allocate layers variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_number_neurons_by_layer, ptr_host_Neural_Network_received->total_layers * sizeof(size_t)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_value_dropout_by_layer, ptr_host_Neural_Network_received->total_layers * sizeof(var)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_type_layer, ptr_host_Neural_Network_received->total_layers * sizeof(DL::LAYER::TYPE)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_type_activation, ptr_host_Neural_Network_received->total_layers * sizeof(DL::LAYER_ACTIVATION::TYPE)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_type_dropout_by_layer, ptr_host_Neural_Network_received->total_layers * sizeof(DL::LAYER_DROPOUT::TYPE)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_type_normalization_by_layer, ptr_host_Neural_Network_received->total_layers * sizeof(DL::LAYER_NORM::TYPE)));
    // |END| allocate layers variable. |END|

    // allocate neurons variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_first_connection_index, ptr_host_Neural_Network_received->total_neuron_units * sizeof(size_t)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_last_connection_index, ptr_host_Neural_Network_received->total_neuron_units * sizeof(size_t)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_bias_index, ptr_host_Neural_Network_received->total_neuron_units * sizeof(size_t)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_type_activation_function, ptr_host_Neural_Network_received->total_neuron_units * sizeof(enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS)));
    // |END| allocate neurons variable. |END|

    // allocate connections.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_parameters, ptr_host_Neural_Network_received->total_parameters * sizeof(var)));
    // |END| allocate connections. |END|
        
    // allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Model)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_accuracy_values, 5_UZ * sizeof(var)));
    // |END| allocate structure neural network global variable. |END|
        
    struct Neuron_unit const *tmp_ptr_host_neuron_it(ptr_host_Neural_Network_received->ptr_array_layers->ptr_array_neuron_units);

    struct Layer const *tmp_ptr_host_layer_it(ptr_host_Neural_Network_received->ptr_array_layers);

    for(size_t tmp_index_neuron,
                            tmp_number_neurons_in_layer,
                            tmp_index_neuron_so_far(0u),
                            tmp_index_layer(0u); tmp_index_layer != ptr_host_Neural_Network_received->total_layers; ++tmp_index_layer,
                                                                                                                                                                ++tmp_ptr_host_layer_it)
    {
        // Assign layers variable.
        tmp_number_neurons_in_layer = static_cast<size_t>(tmp_ptr_host_layer_it->ptr_last_neuron_unit - tmp_ptr_host_layer_it->ptr_array_neuron_units);
        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_number_neurons_by_layer[tmp_index_layer],
                                                        &tmp_number_neurons_in_layer,
                                                        sizeof(size_t),
                                                        cudaMemcpyHostToDevice));
        
        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_value_dropout_by_layer[tmp_index_layer],
                                                        &tmp_ptr_host_layer_it->dropout_values[0],
                                                        sizeof(var),
                                                        cudaMemcpyHostToDevice));

        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_type_layer[tmp_index_layer],
                                                        &tmp_ptr_host_layer_it->type_layer,
                                                        sizeof(DL::LAYER::TYPE),
                                                        cudaMemcpyHostToDevice));
            
        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_type_activation[tmp_index_layer],
                                                        &tmp_ptr_host_layer_it->type_activation,
                                                        sizeof(DL::LAYER_ACTIVATION::TYPE),
                                                        cudaMemcpyHostToDevice));

        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_type_dropout_by_layer[tmp_index_layer],
                                                        &tmp_ptr_host_layer_it->dropout_values[0],
                                                        sizeof(DL::LAYER_DROPOUT::TYPE),
                                                        cudaMemcpyHostToDevice));

        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_type_normalization_by_layer[tmp_index_layer],
                                                        &tmp_ptr_host_layer_it->type_normalization,
                                                        sizeof(DL::LAYER_NORM::TYPE),
                                                        cudaMemcpyHostToDevice));
        // |END| Assign layers variable. |END|

        // Assign neurons variable.
        for(tmp_index_neuron = 0u; tmp_index_neuron != tmp_number_neurons_in_layer; ++tmp_index_neuron,
                                                                                                                              ++tmp_index_neuron_so_far,
                                                                                                                              ++tmp_ptr_host_neuron_it)
        {
            CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_neurons_first_connection_index[tmp_index_neuron_so_far],
                                                            tmp_ptr_host_neuron_it->ptr_first_forward_connection_index,
                                                            sizeof(size_t),
                                                            cudaMemcpyHostToDevice));
            CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_neurons_last_connection_index[tmp_index_neuron_so_far],
                                                            tmp_ptr_host_neuron_it->ptr_last_forward_connection_index,
                                                            sizeof(size_t),
                                                            cudaMemcpyHostToDevice));
                
            CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_neurons_type_activation_function[tmp_index_neuron_so_far],
                                                            tmp_ptr_host_neuron_it->ptr_type_activation_function,
                                                            sizeof(enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS),
                                                            cudaMemcpyHostToDevice));
        }
        // |END| Assign neurons variable. |END|
    }
        
    // Assign connections.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_parameters,
                                                    ptr_host_Neural_Network_received->ptr_array_parameters,
                                                    ptr_host_Neural_Network_received->total_parameters * sizeof(var),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign connections. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                            ptr_host_Neural_Network_received,
                                                            sizeof(class Model),
                                                            cudaMemcpyHostToDevice));

    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_accuracy_values,
                                                             ptr_host_Neural_Network_received->ptr_array_accuracy_values,
                                                             5_UZ * sizeof(var),
                                                             cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__cuModel__Copy__Host_To_Device <<< 1, 1 >>> (tmp_ptr_device_array_number_neurons_by_layer, // size_t
                                                                                                        tmp_ptr_device_array_neurons_first_connection_index, // size_t
                                                                                                        tmp_ptr_device_array_neurons_last_connection_index, // size_t
                                                                                                        tmp_ptr_device_array_neurons_bias_index, // size_t
                                                                                                        *ptr_host_Neural_Network_received->ptr_array_number_loss,
                                                                                                        *ptr_host_Neural_Network_received->ptr_array_number_bit_fail,
                                                                                                        *ptr_host_Neural_Network_received->ptr_array_loss_values,
                                                                                                        tmp_ptr_device_array_accuracy_values,
                                                                                                        tmp_ptr_device_array_value_dropout_by_layer, // var
                                                                                                        tmp_ptr_device_array_parameters, // var
                                                                                                        tmp_ptr_device_array_type_layer, // enum
                                                                                                        tmp_ptr_device_array_type_activation, // enum
                                                                                                        tmp_ptr_device_array_type_dropout_by_layer, // var
                                                                                                        tmp_ptr_device_array_type_normalization_by_layer, // enum
                                                                                                        tmp_ptr_device_array_neurons_type_activation_function, // enum
                                                                                                        tmp_ptr_device_original_Neural_Network, // struct
                                                                                                        this); // class
        
    CUDA__Check_Error();

    // Delete layers variable.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_number_neurons_by_layer)); // size_t
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_value_dropout_by_layer)); // var
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_type_layer)); // enum
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_type_activation)); // enum
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_type_dropout_by_layer)); // var
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_type_normalization_by_layer)); // bool
    // |END| Delete layers variable. |END|

    // Delete neurons variable.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_first_connection_index)); // size_t
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_last_connection_index)); // size_t
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_bias_index)); // size_t
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_type_activation_function)); // enum
    // |END| Delete neurons variable. |END|
    
    // Delete connections.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_parameters)); // var
    // |END| Delete connections. |END|
    
    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_accuracy_values));
    // |END| Delete neural network. |END|

    if(ptr_host_Neural_Network_received->Use__Normalization() && ptr_host_Neural_Network_received->Copy__Batch_Normalization_Neurons__Host_To_Device(ptr_host_Neural_Network_received->ptr_array_normalized_batch_units_scales,
                                                                                                                                                                                                                                      ptr_host_Neural_Network_received->ptr_array_normalized_batch_units_shifts,
                                                                                                                                                                                                                                      ptr_host_Neural_Network_received->ptr_array_normalized_batch_units_means_averages,
                                                                                                                                                                                                                                      ptr_host_Neural_Network_received->ptr_array_normalized_batch_units_variances_averages) == false)
    {
        ERR(L"An error has been triggered from the \"Copy__Batch_Normalization_Neurons__Host_To_Device()\" function.",);

        this->Deallocate();

        return false;
    }

    if(this->Initialize_cuRAND(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count())) == false)
    {
        ERR(L"An error has been triggered from the \"Initialize_cuRAND()\" function.",);
        
        this->Deallocate();

        return false;
    }

    return true;
}

bool Model::Copy__Optimizer_Paramaters__Device_To_Host(void)
{
    switch(this->type_optimizer_function)
    {
        case DL::OPTIMIZER::GD: return(this->Copy__Optimizer_Gradient_Descent__Device_To_Host());
        case DL::OPTIMIZER::IRPROP_MINUS: return(this->Copy__Optimizer_RPROP_minus__Device_To_Host());
        case DL::OPTIMIZER::IRPROP_PLUS: return(this->Copy__Optimizer_RPROP_plus__Device_To_Host());
        case DL::OPTIMIZER::ADAM:
        case DL::OPTIMIZER::ADAMAX:
        case DL::OPTIMIZER::NOSADAM: return(this->Copy__Optimizer_Adam__Device_To_Host());
        case DL::OPTIMIZER::AMSGRAD: return(this->Copy__Optimizer_AMSGrad__Device_To_Host());
        default:
            ERR(L"Can not copy parameters of the optimizer (%u | %ls).",
                                    this->type_optimizer_function,
                                    DL::OPTIMIZER_NAME[this->type_optimizer_function].c_str());
                return false;
    }
}

template<typename T>
__global__ void kernel__cuModel__Copy__Optimizer_Gradient_Descent__Device_To_Host(T *const ptr_optimizer_time_step_received,
                                                                                                                                                T *const ptr_warm_restarts_maximum_learning_rate_received,
                                                                                                                                                T *const ptr_warm_T_i_received,
                                                                                                                                                T *const ptr_array_previous_delta_parameters_received,
                                                                                                                                                class cuModel const *const ptr_cuModel_received)
{
    *ptr_optimizer_time_step_received = ptr_cuModel_received->optimizer_time_step;
    *ptr_warm_restarts_maximum_learning_rate_received = ptr_cuModel_received->warm_restarts_maximum_learning_rate;
    *ptr_warm_T_i_received = ptr_cuModel_received->warm_restarts_T_i;
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_array_previous_delta_parameters_received,
                                             ptr_cuModel_received->ptr_array_previous_delta_parameters,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Model::Copy__Optimizer_Gradient_Descent__Device_To_Host(T &ref_optimizer_time_step_received,
                                                                                                                T &ref_warm_maximum_learning_rate_received,
                                                                                                                T &ref_warm_T_i_received,
                                                                                                                T *const ptr_array_previous_delta_parameters_received) const
{
    T *tmp_ptr_device_optimizer_time_step,
        *tmp_ptr_device_warm_maximum_learning_rate,
        *tmp_ptr_device_warm_T_i,
        *tmp_ptr_device_array_previous_delta_weights_received;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_optimizer_time_step, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_maximum_learning_rate, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_T_i, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_delta_weights_received, this->total_parameters * sizeof(T)));

    kernel__cuModel__Copy__Optimizer_Gradient_Descent__Device_To_Host<T> <<< 1, 1 >>> (tmp_ptr_device_optimizer_time_step,
                                                                                                                                                            tmp_ptr_device_warm_maximum_learning_rate,
                                                                                                                                                            tmp_ptr_device_warm_T_i,
                                                                                                                                                            tmp_ptr_device_array_previous_delta_weights_received,
                                                                                                                                                            this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaMemcpy(&ref_optimizer_time_step_received,
                                                    tmp_ptr_device_optimizer_time_step,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_maximum_learning_rate_received,
                                                    tmp_ptr_device_warm_maximum_learning_rate,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_T_i_received,
                                                    tmp_ptr_device_warm_T_i,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_delta_parameters_received,
                                                    tmp_ptr_device_array_previous_delta_weights_received,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_optimizer_time_step)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_maximum_learning_rate)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_T_i)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_delta_weights_received)); // T

    return true;
}

bool Model::Copy__Optimizer_Gradient_Descent__Device_To_Host(void)
{
    return(this->Copy__Optimizer_Gradient_Descent__Device_To_Host<var>(this->optimizer_time_step,
                                                                                                             this->warm_restarts_maximum_learning_rate,
                                                                                                             this->warm_restarts_T_i,
                                                                                                             this->ptr_array_previous_delta_parameters));
}

template<typename T>
__global__ void kernel__cuModel__Copy__Optimizer_RPROP_minus__Device_To_Host(T *const ptr_array_previous_steps_received,
                                                                                                                                            T *const ptr_array_previous_derivates_parameters_received,
                                                                                                                                            class cuModel const *const ptr_cuModel_received)
{
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_array_previous_steps_received,
                                             ptr_cuModel_received->ptr_array_previous_steps,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_array_previous_derivates_parameters_received,
                                             ptr_cuModel_received->ptr_array_previous_derivatives_parameters,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Model::Copy__Optimizer_RPROP_minus__Device_To_Host(T *const ptr_array_previous_steps_received, T *const ptr_array_previous_derivates_parameters_received) const
{
    T *tmp_ptr_device_array_previous_steps,
        *tmp_ptr_device_array_previous_derivatives_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_steps, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_derivatives_parameters, this->total_parameters * sizeof(T)));

    kernel__cuModel__Copy__Optimizer_RPROP_minus__Device_To_Host<T> <<< 1, 1 >>> (tmp_ptr_device_array_previous_steps,
                                                                                                                                                tmp_ptr_device_array_previous_derivatives_parameters,
                                                                                                                                                this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_steps_received,
                                                    tmp_ptr_device_array_previous_steps,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_derivates_parameters_received,
                                                    tmp_ptr_device_array_previous_derivatives_parameters,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_steps)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_derivatives_parameters)); // T

    return true;
}

bool Model::Copy__Optimizer_RPROP_minus__Device_To_Host(void)
{ return(this->Copy__Optimizer_RPROP_minus__Device_To_Host<var>(this->ptr_array_previous_steps, this->ptr_array_previous_derivatives_parameters)); }

template<typename T>
__global__ void kernel__cuModel__Copy__Optimizer_RPROP_plus__Device_To_Host(T *ptr_loss_received,
                                                                                                                                        T *ptr_previous_loss_received,
                                                                                                                                        T *const ptr_array_previous_steps_received,
                                                                                                                                        T *const ptr_array_previous_derivates_parameters_received,
                                                                                                                                        T *const ptr_array_previous_delta_parameters_received,
                                                                                                                                        class cuModel const *const ptr_cuModel_received)
{
    *ptr_loss_received = ptr_cuModel_received->loss_rprop;
    *ptr_previous_loss_received = ptr_cuModel_received->loss_rprop_tm1;
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_array_previous_steps_received,
                                             ptr_cuModel_received->ptr_array_previous_steps,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_array_previous_derivates_parameters_received,
                                             ptr_cuModel_received->ptr_array_previous_derivatives_parameters,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_array_previous_delta_parameters_received,
                                             ptr_cuModel_received->ptr_array_previous_delta_parameters,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Model::Copy__Optimizer_RPROP_plus__Device_To_Host(T &ref_loss_received,
                                                                                                                    T &ref_previous_loss_received,
                                                                                                                    T *const ptr_array_previous_steps_received,
                                                                                                                    T *const ptr_array_previous_derivates_parameters_received,
                                                                                                                    T *const ptr_array_previous_delta_parameters_received) const
{
    T *tmp_ptr_device_loss,
        *tmp_ptr_device_previous_loss,
        *tmp_ptr_device_array_previous_steps,
        *tmp_ptr_device_array_previous_derivatives_parameters,
        *tmp_ptr_device_array_previous_delta_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_loss, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_previous_loss, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_steps, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_derivatives_parameters, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_delta_parameters, this->total_parameters * sizeof(T)));

    kernel__cuModel__Copy__Optimizer_RPROP_plus__Device_To_Host<T> <<< 1, 1 >>> (tmp_ptr_device_loss,
                                                                                                                                                tmp_ptr_device_previous_loss,
                                                                                                                                                tmp_ptr_device_array_previous_steps,
                                                                                                                                                tmp_ptr_device_array_previous_derivatives_parameters,
                                                                                                                                                tmp_ptr_device_array_previous_delta_parameters,
                                                                                                                                                this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaMemcpy(&ref_loss_received,
                                                    tmp_ptr_device_loss,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_previous_loss_received,
                                                    tmp_ptr_device_previous_loss,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_steps_received,
                                                    tmp_ptr_device_array_previous_steps,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_derivates_parameters_received,
                                                    tmp_ptr_device_array_previous_derivatives_parameters,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_delta_parameters_received,
                                                    tmp_ptr_device_array_previous_delta_parameters,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_previous_loss)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_steps)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_derivatives_parameters)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_delta_parameters)); // T

    return true;
}

bool Model::Copy__Optimizer_RPROP_plus__Device_To_Host(void)
{
    return(this->Copy__Optimizer_RPROP_plus__Device_To_Host<var>(this->loss_rprop,
                                                                                            this->loss_rprop_tm1,
                                                                                            this->ptr_array_previous_steps,
                                                                                            this->ptr_array_previous_derivatives_parameters,
                                                                                            this->ptr_array_previous_delta_parameters));
}

template<typename T>
__global__ void kernel__cuModel__Copy__Optimizer_Adam__Device_To_Host(T *const ptr_optimizer_time_step_received,
                                                                                                                                T *const ptr_warm_restarts_maximum_learning_rate_received,
                                                                                                                                T *const ptr_warm_T_i_received,
                                                                                                                                T *const ptr_array_previous_biased_first_moment_received,
                                                                                                                                T *const ptr_array_previous_biased_second_moment_received,
                                                                                                                                class cuModel const *const ptr_cuModel_received)
{
    *ptr_optimizer_time_step_received = ptr_cuModel_received->optimizer_time_step;
    *ptr_warm_restarts_maximum_learning_rate_received = ptr_cuModel_received->warm_restarts_maximum_learning_rate;
    *ptr_warm_T_i_received = ptr_cuModel_received->warm_restarts_T_i;

    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_array_previous_biased_first_moment_received,
                                             ptr_cuModel_received->ptr_array_previous_biased_first_moment,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_array_previous_biased_second_moment_received,
                                             ptr_cuModel_received->ptr_array_previous_biased_second_moment,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Model::Copy__Optimizer_Adam__Device_To_Host(T &ref_optimizer_time_step_received,
                                                                                                        T &ref_warm_maximum_learning_rate_received,
                                                                                                        T &ref_warm_T_i_received,
                                                                                                        T *const ptr_array_previous_biased_first_moment_received,
                                                                                                        T *const ptr_array_previous_biased_second_moment_received) const
{
    T *tmp_ptr_device_optimizer_time_step,
        *tmp_ptr_device_warm_maximum_learning_rate,
        *tmp_ptr_device_warm_T_i,
        *tmp_ptr_device_array_previous_biased_first_moment,
        *tmp_ptr_device_array_previous_biased_second_moment;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_optimizer_time_step, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_maximum_learning_rate, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_T_i, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_first_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_moment, this->total_parameters * sizeof(T)));

    kernel__cuModel__Copy__Optimizer_Adam__Device_To_Host<T> <<< 1, 1 >>> (tmp_ptr_device_optimizer_time_step,
                                                                                                                                            tmp_ptr_device_warm_maximum_learning_rate,
                                                                                                                                            tmp_ptr_device_warm_T_i,
                                                                                                                                            tmp_ptr_device_array_previous_biased_first_moment,
                                                                                                                                            tmp_ptr_device_array_previous_biased_second_moment,
                                                                                                                                            this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaMemcpy(&ref_optimizer_time_step_received,
                                                    tmp_ptr_device_optimizer_time_step,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_maximum_learning_rate_received,
                                                    tmp_ptr_device_warm_maximum_learning_rate,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_T_i_received,
                                                    tmp_ptr_device_warm_T_i,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_biased_first_moment_received,
                                                    tmp_ptr_device_array_previous_biased_first_moment,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_biased_second_moment_received,
                                                    tmp_ptr_device_array_previous_biased_second_moment,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_optimizer_time_step)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_maximum_learning_rate)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_T_i)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_first_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_moment)); // T

    return true;
}

bool Model::Copy__Optimizer_Adam__Device_To_Host(void)
{
    return(this->Copy__Optimizer_Adam__Device_To_Host<var>(this->optimizer_time_step,
                                                                                            this->warm_restarts_maximum_learning_rate,
                                                                                            this->warm_restarts_T_i,
                                                                                            this->ptr_array_previous_biased_first_moment,
                                                                                            this->ptr_array_previous_biased_second_moment));
}

template<typename T>
__global__ void kernel__cuModel__Copy__Optimizer_AMSGrad__Device_To_Host(T *const ptr_optimizer_time_step_received,
                                                                                                                                    T *const ptr_warm_restarts_maximum_learning_rate_received,
                                                                                                                                    T *const ptr_warm_T_i_received,
                                                                                                                                    T *const ptr_array_previous_biased_first_moment_received,
                                                                                                                                    T *const ptr_array_previous_biased_second_moment_received,
                                                                                                                                    T *const ptr_array_previous_biased_second_moment_hat_received,
                                                                                                                                    class cuModel const *const ptr_cuModel_received)
{
    *ptr_optimizer_time_step_received = ptr_cuModel_received->optimizer_time_step;
    *ptr_warm_restarts_maximum_learning_rate_received = ptr_cuModel_received->warm_restarts_maximum_learning_rate;
    *ptr_warm_T_i_received = ptr_cuModel_received->warm_restarts_T_i;
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_array_previous_biased_first_moment_received,
                                             ptr_cuModel_received->ptr_array_previous_biased_first_moment,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_array_previous_biased_second_moment_received,
                                             ptr_cuModel_received->ptr_array_previous_biased_second_moment,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                             ptr_array_previous_biased_second_moment_hat_received,
                                             ptr_cuModel_received->ptr_array_previous_biased_second_moment_hat,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                             ptr_cuModel_received->ptr_array_dim3_block + 1);
}

template<typename T>
bool Model::Copy__Optimizer_AMSGrad__Device_To_Host(T &ref_optimizer_time_step_received,
                                                                                                              T &ref_warm_maximum_learning_rate_received,
                                                                                                              T &ref_warm_T_i_received,
                                                                                                              T *const ptr_array_previous_biased_first_moment_received,
                                                                                                              T *const ptr_array_previous_biased_second_moment_received,
                                                                                                              T *const ptr_array_previous_biased_second_moment_hat_received) const
{
    T *tmp_ptr_device_optimizer_time_step,
        *tmp_ptr_device_warm_maximum_learning_rate,
        *tmp_ptr_device_warm_T_i,
        *tmp_ptr_device_array_previous_biased_first_moment,
        *tmp_ptr_device_array_previous_biased_second_moment,
        *tmp_ptr_device_array_previous_biased_second_moment_hat;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_optimizer_time_step, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_maximum_learning_rate, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_warm_T_i, sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_first_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_moment, this->total_parameters * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_previous_biased_second_moment_hat, this->total_parameters * sizeof(T)));

    kernel__cuModel__Copy__Optimizer_AMSGrad__Device_To_Host<T> <<< 1, 1 >>> (tmp_ptr_device_optimizer_time_step,
                                                                                                                                                tmp_ptr_device_warm_maximum_learning_rate,
                                                                                                                                                tmp_ptr_device_warm_T_i,
                                                                                                                                                tmp_ptr_device_array_previous_biased_first_moment,
                                                                                                                                                tmp_ptr_device_array_previous_biased_second_moment,
                                                                                                                                                tmp_ptr_device_array_previous_biased_second_moment_hat,
                                                                                                                                                this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaMemcpy(&ref_optimizer_time_step_received,
                                                    tmp_ptr_device_optimizer_time_step,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_maximum_learning_rate_received,
                                                    tmp_ptr_device_warm_maximum_learning_rate,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(&ref_warm_T_i_received,
                                                    tmp_ptr_device_warm_T_i,
                                                    sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_biased_first_moment_received,
                                                    tmp_ptr_device_array_previous_biased_first_moment,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_biased_second_moment_received,
                                                    tmp_ptr_device_array_previous_biased_second_moment,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_previous_biased_second_moment_hat_received,
                                                    tmp_ptr_device_array_previous_biased_second_moment_hat,
                                                    this->total_parameters * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_optimizer_time_step)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_maximum_learning_rate)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_warm_T_i)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_first_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_moment)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_previous_biased_second_moment_hat)); // T

    return true;
}

bool Model::Copy__Optimizer_AMSGrad__Device_To_Host(void)
{
    return(this->Copy__Optimizer_AMSGrad__Device_To_Host<var>(this->optimizer_time_step,
                                                                                                 this->warm_restarts_maximum_learning_rate,
                                                                                                 this->warm_restarts_T_i,
                                                                                                 this->ptr_array_previous_biased_first_moment,
                                                                                                 this->ptr_array_previous_biased_second_moment,
                                                                                                 this->ptr_array_previous_biased_second_moment_hat));
}

template<typename T>
__global__ void kernel__cuModel__Copy__Batch_Normalization_Neurons__Device_To_Host(T *const ptr_array_neuron_units_scale_received,
                                                                                                                                                T *const ptr_array_neuron_units_shift_received,
                                                                                                                                                T *const ptr_array_neuron_units_mean_average_received,
                                                                                                                                                T *const ptr_array_neuron_units_variance_average_received,
                                                                                                                                                class cuModel const *const ptr_cuModel_received)
{
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_neuron_units,
                                             ptr_array_neuron_units_scale_received,
                                             ptr_cuModel_received->ptr_array_normalized_batch_units_scales,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 3,
                                             ptr_cuModel_received->ptr_array_dim3_block + 3);
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_neuron_units,
                                             ptr_array_neuron_units_shift_received,
                                             ptr_cuModel_received->ptr_array_normalized_batch_units_shifts,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 3,
                                             ptr_cuModel_received->ptr_array_dim3_block + 3);
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_neuron_units,
                                             ptr_array_neuron_units_mean_average_received,
                                             ptr_cuModel_received->ptr_array_normalized_batch_units_means_averages,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 3,
                                             ptr_cuModel_received->ptr_array_dim3_block + 3);
    
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_neuron_units,
                                             ptr_array_neuron_units_variance_average_received,
                                             ptr_cuModel_received->ptr_array_normalized_batch_units_variances_averages,
                                             ptr_cuModel_received->ptr_array_dim3_grid + 3,
                                             ptr_cuModel_received->ptr_array_dim3_block + 3);
}

template<typename T>
bool Model::Copy__Batch_Normalization_Neurons__Device_To_Host(T *const ptr_array_neuron_units_scale_received,
                                                                                                                            T *const ptr_array_neuron_units_shift_received,
                                                                                                                            T *const ptr_array_neuron_units_mean_average_received,
                                                                                                                            T *const ptr_array_neuron_units_variance_average_received) const
{
    T *tmp_ptr_device_array_neurons_scale,
        *tmp_ptr_device_array_neurons_shift,
        *tmp_ptr_device_array_neurons_mean_average,
        *tmp_ptr_device_array_neurons_variance_average;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_scale, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_shift, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_mean_average, this->total_neuron_units * sizeof(T)));
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_neurons_variance_average, this->total_neuron_units * sizeof(T)));

    kernel__cuModel__Copy__Batch_Normalization_Neurons__Device_To_Host<T> <<< 1, 1 >>> (tmp_ptr_device_array_neurons_scale,
                                                                                                                                                tmp_ptr_device_array_neurons_shift,
                                                                                                                                                tmp_ptr_device_array_neurons_mean_average,
                                                                                                                                                tmp_ptr_device_array_neurons_variance_average,
                                                                                                                                                this->cumodel);
        
    CUDA__Check_Error();
        
    CUDA__Safe_Call(cudaMemcpy(ptr_array_neuron_units_scale_received,
                                                    tmp_ptr_device_array_neurons_scale,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_neuron_units_shift_received,
                                                    tmp_ptr_device_array_neurons_shift,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_neuron_units_mean_average_received,
                                                    tmp_ptr_device_array_neurons_mean_average,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyDeviceToHost));
    CUDA__Safe_Call(cudaMemcpy(ptr_array_neuron_units_variance_average_received,
                                                    tmp_ptr_device_array_neurons_variance_average,
                                                    this->total_neuron_units * sizeof(T),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_scale)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_shift)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_mean_average)); // T
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_neurons_variance_average)); // T

    return true;
}

template<typename T>
__global__ void kernel__cuModel__Copy__Parameters__Host_To_Device(T *const ptr_array_parameters_received, class cuModel *const ptr_cuModel_received)
{
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                                   ptr_cuModel_received->ptr_array_parameters,
                                                   ptr_array_parameters_received,
                                                   ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                                   ptr_cuModel_received->ptr_array_dim3_block + 1);
}

void Model::Copy__Parameters__Host_To_Device(void)
{
    var *tmp_ptr_device_array_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_parameters, this->total_parameters * sizeof(var)));

    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_array_parameters,
                                                    this->ptr_array_parameters,
                                                    this->total_parameters * sizeof(var),
                                                    cudaMemcpyHostToDevice));

    kernel__cuModel__Copy__Parameters__Host_To_Device<var> <<< 1, 1 >>> (tmp_ptr_device_array_parameters, this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_parameters)); // var
}

bool Model::Copy__Batch_Normalization_Neurons__Device_To_Host(void)
{
    return(this->Copy__Batch_Normalization_Neurons__Device_To_Host<var>(this->ptr_array_normalized_batch_units_scales,
                                                                                                               this->ptr_array_normalized_batch_units_shifts,
                                                                                                               this->ptr_array_normalized_batch_units_means_averages,
                                                                                                               this->ptr_array_normalized_batch_units_variances_averages));
}

template<typename T>
__global__ void kernel__cuModel__Copy__Parameters__Device_To_Host(T *const ptr_array_parameters_received, class cuModel const *const ptr_cuModel_received)
{
    Memory::Memory_Copy_1D<T>(ptr_cuModel_received->total_parameters,
                                                   ptr_array_parameters_received,
                                                   ptr_cuModel_received->ptr_array_parameters,
                                                   ptr_cuModel_received->ptr_array_dim3_grid + 1,
                                                   ptr_cuModel_received->ptr_array_dim3_block + 1);
}

bool Model::Copy__Parameters__Device_To_Host(void)
{
    var *tmp_ptr_device_array_parameters;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_parameters, this->total_parameters * sizeof(var)));

    kernel__cuModel__Copy__Parameters__Device_To_Host<var> <<< 1, 1 >>> (tmp_ptr_device_array_parameters, this->cumodel);
        
    CUDA__Check_Error();

    CUDA__Safe_Call(cudaMemcpy(this->ptr_array_parameters,
                                                    tmp_ptr_device_array_parameters,
                                                    this->total_parameters * sizeof(var),
                                                    cudaMemcpyDeviceToHost));

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_parameters)); // var
    
    this->is_update_from_device = true;

    return true;
}

[[deprecated("Not properly implemented.")]] __global__ void kernel__cuModel__Copy_Device_To_Host(size_t *const ptr_array_number_neurons_by_layer_received,
                                                                                                 size_t *const ptr_array_neuron_units_first_connection_index_received,
                                                                                                 size_t *const ptr_array_neuron_units_last_connection_index_received,
                                                                                                 DL::LAYER::TYPE *const ptr_array_type_layer_received,
                                                                                                 DL::LAYER_ACTIVATION::TYPE *const ptr_array_type_activation_received,
                                                                                                 enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS *const ptr_array_neuron_units_type_activation_function_received,
                                                                                                 var *const ptr_array_weigth_received,
                                                                                                 var *const ptr_array_neuron_sum_received,
                                                                                                 var *const ptr_array_neuron_value_received,
                                                                                                 class cuModel *const ptr_cuModel_received) {
  // NotImplementedError
  // ...
  // ...
  ERR(L"NotImplementedError");
}

[[deprecated("Not properly implemented.")]] bool Model::Copy_Device_To_Host(bool const refresh_from_genetic_algorithm_received)
{
    if(refresh_from_genetic_algorithm_received)
    {
        var *tmp_ptr_device_array_parameters;

        CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_parameters, this->total_parameters * sizeof(var)));

        kernel__cuModel__Copy__Parameters__Device_To_Host<var> <<< 1, 1 >>> (tmp_ptr_device_array_parameters, this->cumodel);
            
        CUDA__Check_Error();

        CUDA__Safe_Call(cudaMemcpy(this->ptr_array_parameters, tmp_ptr_device_array_parameters, this->total_parameters * sizeof(var), cudaMemcpyDeviceToHost));

        CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_parameters)); // var
    }
    else
    {
      // NotImplementedError
      // ...
      // ...
      ERR(L"NotImplementedError");
    }

    return true;
}

__global__ void kernel__cuModel__Copy_Warm_Restarts_Parameters(class Model const *const model, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Copy_Warm_Restarts_Parameters(model); }

__host__ __device__ void cuModel::Copy_Warm_Restarts_Parameters(class Model const *const model) {
#ifndef COMPILE_CUDA
    class Model *tmp_ptr_device_original_Neural_Network;

    // allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Model)));
    // |END| allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    model,
                                                    sizeof(class Model),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__cuModel__Copy_Warm_Restarts_Parameters <<< 1, 1 >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    this->use_warm_restarts = model->use_warm_restarts;
    this->warm_restarts_decay_learning_rate = model->warm_restarts_decay_learning_rate;
    this->warm_restarts_initial_maximum_learning_rate = model->warm_restarts_initial_maximum_learning_rate;
    this->warm_restarts_maximum_learning_rate = model->warm_restarts_maximum_learning_rate;
    this->warm_restarts_minimum_learning_rate = model->warm_restarts_minimum_learning_rate;
    this->warm_restarts_initial_T_i = model->warm_restarts_initial_T_i;
    this->warm_restarts_T_i = model->warm_restarts_T_i;
    this->warm_restarts_multiplier = model->warm_restarts_multiplier;
#endif
}
    
__global__ void kernel__cuModel__Copy_Optimizer_Parameters(class Model const *const model, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Copy__Optimizer_Parameters(model); }

__host__ __device__ void cuModel::Copy__Optimizer_Parameters(class Model const *const model) {
#ifndef COMPILE_CUDA
    class Model *tmp_ptr_device_original_Neural_Network;

    // allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Model)));
    // |END| allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    model,
                                                    sizeof(class Model),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__cuModel__Copy_Optimizer_Parameters <<< 1, 1 >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    switch(this->type_optimizer_function)
    {
        case DL::OPTIMIZER::GD: this->Copy__Gradient_Descent_Parameters(model); break;
        case DL::OPTIMIZER::IRPROP_MINUS: this->Copy_RPROP_minus_Parameters(model); break;
        case DL::OPTIMIZER::IRPROP_PLUS: this->Copy_RPROP_plus_Parameters(model); break;
        case DL::OPTIMIZER::SARPROP: this->Copy_SARProp_Parameters(model); break;
        case DL::OPTIMIZER::QUICKPROP: this->Copy_QuickProp_Parameters(model); break;
        case DL::OPTIMIZER::ADAM:
        case DL::OPTIMIZER::ADAMAX:
        case DL::OPTIMIZER::AMSGRAD: this->Copy_Adam_Parameters(model); break;
        case DL::OPTIMIZER::NOSADAM: this->Copy_NosAdam_Parameters(model); break;
        default:
            ERR(L"Can not copy parameters of the optimizer (%u).",
                        this->type_optimizer_function);
                break;
    }

    this->Copy_Warm_Restarts_Parameters(model);

    this->optimizer_time_step = 0_r;
    this->epoch_time_step = 1_r;
#endif
}
    
__global__ void kernel__cuModel__Copy__Gradient_Descent_Parameters(class Model const *const model, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Copy__Gradient_Descent_Parameters(model); }

__host__ __device__ void cuModel::Copy__Gradient_Descent_Parameters(class Model const *const model) {
#ifndef COMPILE_CUDA
    class Model *tmp_ptr_device_original_Neural_Network;

    // allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Model)));
    // |END| allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    model,
                                                    sizeof(class Model),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__cuModel__Copy__Gradient_Descent_Parameters <<< 1, 1 >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Gradient descent parameters.
    var const lr_mom(this->learning_momentum);

    this->learning_rate = model->learning_rate;
    this->learning_momentum = model->learning_momentum;
    this->use_nesterov = model->use_nesterov;
        
    if(lr_mom == 0_r)
    { this->Allocate__Parameter__Gradient_Descent(); }
    else if(this->learning_momentum == 0_r)
    { this->Deallocate__Parameter__Gradient_Descent(); }
    // |END| Gradient descent parameters. |END|
#endif
}

__global__ void kernel__cuModel__Copy_QuickProp_Parameters(class Model const *const model, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Copy_QuickProp_Parameters(model); }

__host__ __device__ void cuModel::Copy_QuickProp_Parameters(class Model const *const model) {
#ifndef COMPILE_CUDA
    class Model *tmp_ptr_device_original_Neural_Network;

    // allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Model)));
    // |END| allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    model,
                                                    sizeof(class Model),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__cuModel__Copy_QuickProp_Parameters <<< 1, 1 >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Quickprop parameters.
    this->quickprop_decay = model->quickprop_decay;
    this->quickprop_mu = model->quickprop_mu;
    // |END| Quickprop parameters. |END|
#endif
}

__global__ void kernel__cuModel__Copy_RPROP_minus_Parameters(class Model const *const model, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Copy_RPROP_minus_Parameters(model); }

__host__ __device__ void cuModel::Copy_RPROP_minus_Parameters(class Model const *const model) {
#ifndef COMPILE_CUDA
    class Model *tmp_ptr_device_original_Neural_Network;

    // allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Model)));
    // |END| allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    model,
                                                    sizeof(class Model),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__cuModel__Copy_RPROP_minus_Parameters <<< 1, 1 >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Resillent propagation minus parameters.
    this->rprop_increase_factor = model->rprop_increase_factor;
    this->rprop_decrease_factor = model->rprop_decrease_factor;
    this->rprop_delta_min = model->rprop_delta_min;
    this->rprop_delta_max = model->rprop_delta_max;
    this->rprop_delta_zero = model->rprop_delta_zero;
    // |END| Resillent propagation minus parameters. |END|
#endif
}

__global__ void kernel__cuModel__Copy_RPROP_plus_Parameters(class Model const *const model, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Copy_RPROP_plus_Parameters(model); }

__host__ __device__ void cuModel::Copy_RPROP_plus_Parameters(class Model const *const model) {
#ifndef COMPILE_CUDA
    class Model *tmp_ptr_device_original_Neural_Network;

    // allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Model)));
    // |END| allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    model,
                                                    sizeof(class Model),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__cuModel__Copy_RPROP_plus_Parameters <<< 1, 1 >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Resillent propagation plus parameters.
    this->Copy_RPROP_minus_Parameters(model);

    this->loss_rprop = model->loss_rprop;
    this->loss_rprop_tm1 = model->loss_rprop_tm1;
    // |END| Resillent propagation plus parameters. |END|
#endif
}

__global__ void kernel__cuModel__Copy_SARProp_Parameters(class Model const *const model, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Copy_SARProp_Parameters(model); }

__host__ __device__ void cuModel::Copy_SARProp_Parameters(class Model const *const model) {
#ifndef COMPILE_CUDA
    class Model *tmp_ptr_device_original_Neural_Network;

    // allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Model)));
    // |END| allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    model,
                                                    sizeof(class Model),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__cuModel__Copy_SARProp_Parameters <<< 1, 1 >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // SARProp parameters.
    this->sarprop_weight_decay_shift = model->sarprop_weight_decay_shift;
    this->sarprop_step_error_threshold_factor = model->sarprop_step_error_threshold_factor;
    this->sarprop_step_error_shift = model->sarprop_step_error_shift;
    this->sarprop_temperature = model->sarprop_temperature;
    this->sarprop_epoch = model->sarprop_epoch;
    // |END| SARProp parameters. |END|
#endif
}

__global__ void kernel__cuModel__Copy_Adam_Parameters(class Model const *const model, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Copy_Adam_Parameters(model); }

__host__ __device__ void cuModel::Copy_Adam_Parameters(class Model const *const model) {
#ifndef COMPILE_CUDA
    class Model *tmp_ptr_device_original_Neural_Network;

    // allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Model)));
    // |END| allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    model,
                                                    sizeof(class Model),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__cuModel__Copy_Adam_Parameters <<< 1, 1 >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Adam parameters.
     this->adam_learning_rate = model->adam_learning_rate;
     this->adam_beta1 = model->adam_beta1;
     this->adam_beta2 = model->adam_beta2;
     this->adam_epsilon = model->adam_epsilon;
     this->use_adam_bias_correction = model->use_adam_bias_correction;
    // |END| Adam parameters. |END|
#endif
}

__global__ void kernel__cuModel__Copy_NosAdam_Parameters(class Model const *const model, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Copy_NosAdam_Parameters(model); }

__host__ __device__ void cuModel::Copy_NosAdam_Parameters(class Model const *const model) {
#ifndef COMPILE_CUDA
    class Model *tmp_ptr_device_original_Neural_Network;

    // allocate structure neural network global variable.
    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_original_Neural_Network, sizeof(class Model)));
    // |END| allocate structure neural network global variable. |END|
        
    // Assign structure neural network global variable.
    CUDA__Safe_Call(cudaMemcpy(tmp_ptr_device_original_Neural_Network,
                                                    model,
                                                    sizeof(class Model),
                                                    cudaMemcpyHostToDevice));
    // |END| Assign structure neural network global variable. |END|

    kernel__cuModel__Copy_NosAdam_Parameters <<< 1, 1 >>> (tmp_ptr_device_original_Neural_Network, this);

    // Delete neural network.
    CUDA__Safe_Call(cudaFree(tmp_ptr_device_original_Neural_Network));
    // |END| Delete neural network. |END|
#else
    // Adam parameters.
     this->adam_learning_rate = model->adam_learning_rate;
     this->adam_beta1 = model->adam_beta1;
     this->adam_beta2 = model->adam_beta2;
     this->adam_epsilon = model->adam_epsilon;
     this->use_adam_bias_correction = model->use_adam_bias_correction;
     this->adam_gamma = model->adam_gamma;
    // |END| Adam parameters. |END|
#endif
}

__global__ void kernel__cuModel__Copy_Dropout(var const *const ptr_array_probability_retained_unit_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->device__Copy_Dropout(ptr_array_probability_retained_unit_received); }

__host__ void cuModel::Copy__Dropout(class Model const *const model)
{
    var *tmp_ptr_device_array_probability_retained_unit_by_layer;

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_probability_retained_unit_by_layer, model->total_layers * sizeof(var)));

    for(size_t tmp_index_layer(0u); tmp_index_layer != model->total_layers; ++tmp_index_layer)
    {
        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_probability_retained_unit_by_layer[tmp_index_layer],
                                                        &(model->ptr_array_layers + tmp_index_layer)->dropout_values[0],
                                                        sizeof(var),
                                                        cudaMemcpyHostToDevice));
    }

    kernel__cuModel__Copy_Dropout <<< 1, 1 >>> (tmp_ptr_device_array_probability_retained_unit_by_layer, this);

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_probability_retained_unit_by_layer));
}

__device__ void cuModel::device__Copy_Dropout(var const *ptr_array_probability_retained_unit_received)
{
    struct cuLayer const *const last_layer(this->ptr_last_layer - 1); // Subtract output layer.
    struct cuLayer *layer_it(this->ptr_array_layers);

    for(; layer_it != last_layer; ++layer_it,
                                                                    ++ptr_array_probability_retained_unit_received)
    { this->Set__Probability_Retained_Unit(layer_it, *ptr_array_probability_retained_unit_received); }
}

__global__ void kernel__cuModel__Copy_Normalization(DL::LAYER_NORM::TYPE const *const ptr_array_normalization_by_layers_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->device__Copy_Normalization(ptr_array_normalization_by_layers_received); }

__host__ void cuModel::Copy__Normalization(class Model const *const model)
{
    DL::LAYER_NORM::TYPE *tmp_ptr_device_array_normalization_by_layers(NULL);

    CUDA__Safe_Call(cudaMalloc((void**)&tmp_ptr_device_array_normalization_by_layers, model->total_layers * sizeof(DL::LAYER_NORM::TYPE)));

    for(size_t tmp_index_layer(0u); tmp_index_layer != model->total_layers; ++tmp_index_layer)
    {
        CUDA__Safe_Call(cudaMemcpy(&tmp_ptr_device_array_normalization_by_layers[tmp_index_layer],
                                                        &(model->ptr_array_layers + tmp_index_layer)->type_normalization,
                                                        sizeof(bool),
                                                        cudaMemcpyHostToDevice));
    }

    kernel__cuModel__Copy_Normalization <<< 1, 1 >>> (tmp_ptr_device_array_normalization_by_layers, this);

    CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_normalization_by_layers));

    if(model->Use__Normalization())
    {
        if(model->Copy__Batch_Normalization_Neurons__Host_To_Device(model->ptr_array_normalized_batch_units_scales,
                                                                                                                                         model->ptr_array_normalized_batch_units_shifts,
                                                                                                                                         model->ptr_array_normalized_batch_units_means_averages,
                                                                                                                                         model->ptr_array_normalized_batch_units_variances_averages) == false)
        {
            ERR(L"From \"Copy__Batch_Normalization_Neurons__Host_To_Device\".",);
        }
    }
}

__device__ void cuModel::device__Copy_Normalization(DL::LAYER_NORM::TYPE const *ptr_array_normalization_by_layers_received)
{
    struct cuLayer const *const last_layer(this->ptr_last_layer - 1); // Subtract output layer.
    struct cuLayer *layer_it(this->ptr_array_layers);
        
    // Hidden layer.
    for(++ptr_array_normalization_by_layers_received,
        ++layer_it; layer_it != last_layer; ++layer_it,
                                                                                                ++ptr_array_normalization_by_layers_received)
    { this->Set__Batch_Renormalization(layer_it, *ptr_array_normalization_by_layers_received == DL::LAYER_NORM::BATCH_RENORMALIZATION); }
    // |END| Hidden layer. |END|
}
    
