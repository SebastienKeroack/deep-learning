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

#include "deep-learning/ops/fill.cuh"
#include "deep-learning/ops/zero.cuh"

#include "deep-learning/v1/learner/model.cuh"
#include "deep-learning/ops/reduce.cuh"
#include "deep-learning/ops/transpose.cuh"
#include "deep-learning/ops/multiply.cuh"
    
#include "deep-learning/v1/learner/model.hpp"

__device__ void cuModel::merge_mp_accu_loss(void)
{
    Reduce::Reduce<var>(this->number_threads,
                                     1_UZ,
                                     this->ptr_array_reduce_loss_values,
                                     this->ptr_array_loss_values,
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);

    Reduce::Reduce<var>(this->number_threads,
                                     1_UZ,
                                     this->ptr_array_reduce_accuracy_values[0],
                                     this->ptr_array_accuracy_values[0],
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);
    
    Reduce::Reduce<var>(this->number_threads,
                                     1_UZ,
                                     this->ptr_array_reduce_accuracy_values[1],
                                     this->ptr_array_accuracy_values[1],
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);

    Reduce::Reduce<var>(this->number_threads,
                                     1_UZ,
                                     this->ptr_array_reduce_accuracy_values[2],
                                     this->ptr_array_accuracy_values[2],
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);

    Reduce::Reduce<var>(this->number_threads,
                                     1_UZ,
                                     this->ptr_array_reduce_accuracy_values[3],
                                     this->ptr_array_accuracy_values[3],
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);
    
    Reduce::Reduce<var>(this->number_threads,
                                     1_UZ,
                                     this->ptr_array_reduce_accuracy_values[4],
                                     this->ptr_array_accuracy_values[4],
                                     this->ptr_array_dim3_grid_reduce_threads,
                                     this->ptr_array_dim3_block_reduce_threads);

    if(this->type_loss_function == DL::LOSS_FN::BIT)
    {
        Reduce::Reduce<size_t>(this->number_threads,
                                              1_UZ,
                                              this->ptr_array_reduce_bit_fail_values,
                                              this->ptr_array_number_bit_fail,
                                              this->ptr_array_dim3_grid_reduce_threads,
                                              this->ptr_array_dim3_block_reduce_threads);
    }
    
    // Synchronize to see the variable reduced.
    if(this->number_threads >= static_cast<size_t>(warpSize * 2)) { CUDA__Check_Error(); }

    if(this->type_loss_function == DL::LOSS_FN::BIT)
    {
        *this->ptr_array_number_bit_fail = *this->ptr_array_reduce_bit_fail_values;
    }

    *this->ptr_array_loss_values = *this->ptr_array_reduce_loss_values;

    this->ptr_array_accuracy_values[0][0] = this->ptr_array_reduce_accuracy_values[0][0];
    this->ptr_array_accuracy_values[1][0] = this->ptr_array_reduce_accuracy_values[1][0];
    this->ptr_array_accuracy_values[2][0] = this->ptr_array_reduce_accuracy_values[2][0];
    this->ptr_array_accuracy_values[3][0] = this->ptr_array_reduce_accuracy_values[3][0];
    this->ptr_array_accuracy_values[4][0] = this->ptr_array_reduce_accuracy_values[4][0];
}
    
__global__ void kernel__cuModel__Reset__Loss(class cuModel *ptr_cuModel_received)
{ ptr_cuModel_received->reset_loss(); }

__host__ __device__ void cuModel::reset_loss(void)
{
#ifdef __CUDA_ARCH__
    this->n_acc_trial = 0u;
    
    Zero_1D<size_t>(this->number_threads,
                                      this->ptr_array_number_loss,
                                      this->ptr_array_dim3_grid,
                                      this->ptr_array_dim3_block);
    
    Zero_1D<size_t>(this->number_threads,
                                      this->ptr_array_number_bit_fail,
                                      this->ptr_array_dim3_grid,
                                      this->ptr_array_dim3_block);
    
    Zero_1D<var>(this->number_threads,
                         this->ptr_array_loss_values,
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
    
    Zero_1D<var>(this->number_threads,
                         this->ptr_array_accuracy_values[0],
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
    
    Zero_1D<var>(this->number_threads,
                         this->ptr_array_accuracy_values[1],
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
    
    Zero_1D<var>(this->number_threads,
                         this->ptr_array_accuracy_values[2],
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
    
    Zero_1D<var>(this->number_threads,
                         this->ptr_array_accuracy_values[3],
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
    
    Zero_1D<var>(this->number_threads,
                         this->ptr_array_accuracy_values[4],
                         this->ptr_array_dim3_grid,
                         this->ptr_array_dim3_block);
#else
    kernel__cuModel__Reset__Loss <<< 1, 1 >>> (this);
#endif
}
    
__device__ var const * cuModel::get_out(size_t const thread_index_received) const
{
    return(&(this->ptr_last_layer - 1)->ptr_array_neuron_units->ptr_array_values[thread_index_received * (this->n_out + 1u)]); // Add bias
}

__device__ var const * cuModel::get_out(size_t const thread_index_received, size_t const time_step_index_received) const
{
    return(&(this->ptr_last_layer - 1)->ptr_array_neuron_units->ptr_array_values[thread_index_received + this->batch_size * this->total_neuron_units_allocated * time_step_index_received]);
}

__device__ var Activation_Derived(var const summation_received,
                                                var const value_received,
                                                enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const activation_function_received)
{
    switch(activation_function_received)
    {
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::LINEAR: return(Activation_Function_LINEAR_derive_t(1_r));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::LINEAR_PIECE: return(Activation_Function_LINEAR_PIECE_derive_t(1_r));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::LINEAR_PIECE_SYMMETRIC: return(Activation_Function_LINEAR_PIECE_SYMMETRIC_derive_t(1_r));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::SIGMOID: return(Activation_Function_SIGMOID_derive_t(1_r, value_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::SIGMOID_STEPWISE: return(Activation_Function_SIGMOID_STEPWISE_derive_t(1_r, value_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::TANH: return(Activation_Function_TANH_derive_t(1_r, value_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::TANH_STEPWISE: return(Activation_Function_TANH_STEPWISE_derive_t(1_r, value_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::THRESHOLD:
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::THRESHOLD_SYMMETRIC:
            ERR(L"Can not training the neural network with this type (%d) of activation function.",
                                    static_cast<size_t>(activation_function_received));
                break;
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::GAUSSIAN:
            return(Activation_Function_GAUSSIAN_derive_t(1_r,
                                                                                  value_received,
                                                                                  summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::GAUSSIAN_SYMMETRIC:
            return(Activation_Function_GAUSSIAN_SYMMETRIC_derive_t(1_r,
                                                                                                      value_received,
                                                                                                      summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::ELLIOT: return(Activation_Function_ELLIOT_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::ELLIOT_SYMMETRIC: return(Activation_Function_ELLIOT_SYMMETRIC_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::SINE: return(Activation_Function_SIN_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::SINE_SYMMETRIC: return(Activation_Function_SIN_SYMMETRIC_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::COSINE: return(Activation_Function_COS_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::COSINE_SYMMETRIC: return(Activation_Function_COS_SYMMETRIC_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::RELU: return(Activation_Function_RELU_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::LEAKY_RELU: return(Activation_Function_LRELU_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::PARAMETRIC_RELU: return(Activation_Function_PRELU_derive_t(1_r, summation_received));
        default:
            ERR(L"Can not find the derivative of activation function (type=%u).",
                                    static_cast<size_t>(activation_function_received));
                break;
    }

    return(0_r);
}

__device__ var Activation_Derived(var const summation_received,
                                                var const value_received,
                                                enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS const activation_function_received,
                                                DL::LOSS_FN::TYPE const type_loss_function_received)
{
    switch(activation_function_received)
    {
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::LINEAR: return(Activation_Function_LINEAR_derive_t(1_r));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::LINEAR_PIECE: return(Activation_Function_LINEAR_PIECE_derive_t(1_r));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::LINEAR_PIECE_SYMMETRIC: return(Activation_Function_LINEAR_PIECE_SYMMETRIC_derive_t(1_r));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::SIGMOID: return(Activation_Function_SIGMOID_derive_t(1_r, value_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::SIGMOID_STEPWISE: return(Activation_Function_SIGMOID_STEPWISE_derive_t(1_r, value_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::TANH: return(Activation_Function_TANH_derive_t(1_r, value_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::TANH_STEPWISE: return(Activation_Function_TANH_STEPWISE_derive_t(1_r, value_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::THRESHOLD:
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::THRESHOLD_SYMMETRIC:
            ERR(L"Can not training the neural network with this type (%d) of activation function.",
                                    static_cast<size_t>(activation_function_received));
                break;
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::GAUSSIAN:
            return(Activation_Function_GAUSSIAN_derive_t(1_r,
                                                                                value_received,
                                                                                summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::GAUSSIAN_SYMMETRIC:
            return(Activation_Function_GAUSSIAN_SYMMETRIC_derive_t(1_r,
                                                                                                    value_received,
                                                                                                    summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::ELLIOT: return(Activation_Function_ELLIOT_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::ELLIOT_SYMMETRIC: return(Activation_Function_ELLIOT_SYMMETRIC_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::SINE: return(Activation_Function_SIN_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::SINE_SYMMETRIC: return(Activation_Function_SIN_SYMMETRIC_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::COSINE: return(Activation_Function_COS_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::COSINE_SYMMETRIC: return(Activation_Function_COS_SYMMETRIC_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::RELU: return(Activation_Function_RELU_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::LEAKY_RELU: return(Activation_Function_LRELU_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::PARAMETRIC_RELU: return(Activation_Function_PRELU_derive_t(1_r, summation_received));
        case  DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::SOFTMAX: return(1_r);
        default:
            ERR(L"Can not find the derivative of activation function (type=%u).",
                                    static_cast<size_t>(activation_function_received));
                break;
    }

    return(0_r);
}

__device__ void Update_Accuracy(var const error_received,
                                                  var const accuracy_variance_received,
                                                  float *const ptr_accuracy_value_received)
{
    if(abs(error_received) <= accuracy_variance_received)
    { ++*ptr_accuracy_value_received; }
}

__device__ void Update_Accuracy__atomic(var const error_received,
                                                                var const accuracy_variance_received,
                                                                float *const ptr_accuracy_value_received)
{
    if(abs(error_received) <= accuracy_variance_received)
    { atomicAdd(ptr_accuracy_value_received, 1.0f); }
}

__device__ void update_loss(var const observed_output_received,
                                            var const desired_output_received,
                                            var const error_received,
                                            float *const ptr_loss_values_received,
                                            DL::LOSS_FN::TYPE const type_loss_function_received)
{
    var tmp_error;
        
    switch(type_loss_function_received)
    {
        case DL::LOSS_FN::ME:
        case DL::LOSS_FN::L1: tmp_error = error_received; break;
        case DL::LOSS_FN::MAE: tmp_error = abs(error_received); break;
        case DL::LOSS_FN::L2:
        case DL::LOSS_FN::MSE:
        case DL::LOSS_FN::RMSE:
            tmp_error = error_received * error_received; // (Û - U)2, square the difference
                break;
        case DL::LOSS_FN::MAPE:
            tmp_error = error_received / observed_output_received;

            tmp_error = abs(tmp_error);
                break;
        case DL::LOSS_FN::SMAPE:
            tmp_error = abs(error_received);

            tmp_error /= abs(desired_output_received) + abs(observed_output_received);
                break;
        case DL::LOSS_FN::MASE_SEASONAL:
        case DL::LOSS_FN::MASE_NON_SEASONAL:
            ERR(L"NotImplementedError");
                return;
        case DL::LOSS_FN::CROSS_ENTROPY:
            tmp_error = observed_output_received != 0_r ? observed_output_received : std::max<var>(observed_output_received, 1.0e-6_r); // Numerical stability.
            
            // TODO: Make cross-entropy multi label / binary.
            //if(this->Use__Multi_Label() || this->n_out == 1_UZ)
            //{
            //    tmp_error = -(desired_output_received * log(tmp_error) + (1_r - desired_output_received) * log(1_r - tmp_error));
            //}
            //else
            {
                tmp_error = -(desired_output_received * log(tmp_error));
            }
                break;
        default:
            ERR(L"Loss type (%d) is not managed in the switch.",
                                     type_loss_function_received);
                return;
    }

    *ptr_loss_values_received += static_cast<float>(tmp_error);
}

__device__ void Update_Error__atomic(var const observed_output_received,
                                                         var const desired_output_received,
                                                         var const error_received,
                                                         float *const ptr_loss_values_received,
                                                         DL::LOSS_FN::TYPE const type_loss_function_received)
{
    var tmp_error;
        
    switch(type_loss_function_received)
    {
        case DL::LOSS_FN::ME:
        case DL::LOSS_FN::L1: tmp_error = error_received; break;
        case DL::LOSS_FN::MAE: tmp_error = abs(error_received); break;
        case DL::LOSS_FN::L2:
        case DL::LOSS_FN::MSE:
        case DL::LOSS_FN::RMSE:
            tmp_error = error_received * error_received; // (Û - U)2, square the difference
                break;
        case DL::LOSS_FN::MAPE:
            tmp_error = error_received / observed_output_received;

            tmp_error = abs(tmp_error);
                break;
        case DL::LOSS_FN::SMAPE:
            tmp_error = abs(error_received);

            tmp_error /= abs(desired_output_received) + abs(observed_output_received);
                break;
        case DL::LOSS_FN::TYPE_LOSS_FUNCTION_MASE: // Non seasonal time series
            ERR(L"NotImplementedError");
        case DL::LOSS_FN::CROSS_ENTROPY:
            tmp_error = observed_output_received != 0_r ? observed_output_received : std::max<var>(observed_output_received, 1.0e-6_r); // Numerical stability.
            
            // TODO: Make cross-entropy multi label / binary.
            //if(this->Use__Multi_Label() || this->n_out == 1_UZ)
            //{
            //    tmp_error = -(desired_output_received * log(tmp_error) + (1_r - desired_output_received) * log(1_r - tmp_error));
            //}
            //else
            {
                tmp_error = -(desired_output_received * log(tmp_error));
            }
                break;
        default:
            ERR(L"Loss type (%d) is not managed in the switch.",
                                     type_loss_function_received);
                return;
    }

    atomicAdd(ptr_loss_values_received, static_cast<float>(tmp_error));
}

__device__ void Update_Error__Binary_Cross_Entropy(var const observed_output_received,
                                                                                var const desired_output_received,
                                                                                float *const ptr_loss_values_received)
{
    var tmp_error(desired_output_received * -log(observed_output_received) + (1_r - desired_output_received) * logf(1_r - observed_output_received));

    *ptr_loss_values_received += static_cast<float>(tmp_error);
}

__device__ void Update_Error__Binary_Cross_Entropy__atomic(var const observed_output_received,
                                                                                             var const desired_output_received,
                                                                                             float *const ptr_loss_values_received)
{
    var tmp_error(desired_output_received * -log(observed_output_received) + (1_r - desired_output_received) * logf(1_r - observed_output_received));

    atomicAdd(ptr_loss_values_received, static_cast<float>(tmp_error));
}

__device__ void Update_Error__Bit_Fail(var const error_received,
                                                          var const bit_fail_limit,
                                                          size_t *const ptr_bit_fail_values_received)
{
    if(abs(error_received) >= bit_fail_limit)
    { ++*ptr_bit_fail_values_received; }
}

__device__ void Update_Error__Bit_Fail__atomic(var const error_received,
                                                                       var const bit_fail_limit,
                                                                       size_t *const ptr_bit_fail_values_received)
{
    if(abs(error_received) >= bit_fail_limit)
    { atomicAdd(ptr_bit_fail_values_received, 1_UZ); }
}


__global__ void kernel__cuModel__Clear_Train_Arrays(class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->device__Clear_Train_Arrays(); }

void Model::Clear_Training_Arrays__CUDA(void)
{ kernel__cuModel__Clear_Train_Arrays <<< 1, 1 >>> (this->cumodel); }

__device__ void cuModel::device__Clear_Train_Arrays(void)
{
    struct dim3 tmp_dim3_grid,
                     tmp_dim3_block;
    
    // Weights slope.
    if(this->ptr_array_derivatives_parameters == nullptr)
    {
        var *tmp_ptr_array_derivate_weights(new var[this->number_threads * this->total_parameters_allocated]);
        if(tmp_ptr_array_derivate_weights == nullptr)
        {
            ERR(L"Can not allocate memory.");

            return;
        }
        this->ptr_array_derivatives_parameters = tmp_ptr_array_derivate_weights;
    }

    this->Get__Class_Device_Information_Array()->Get__CUDA_Device()->Grid_Block_1Dimensions(this->number_threads * this->total_parameters_allocated,
                                                                                                                                           0,
                                                                                                                                           tmp_dim3_grid,
                                                                                                                                           tmp_dim3_block);

    Zero_1D<var>(this->number_threads * this->total_parameters_allocated,
                        this->ptr_array_derivatives_parameters,
                        &tmp_dim3_grid,
                        &tmp_dim3_block);
    // |END| Weights slope. |END|
    
    this->Clear_Optimizer();
    
    this->warm_restarts_maximum_learning_rate = this->warm_restarts_initial_maximum_learning_rate;
    this->warm_restarts_T_i = this->warm_restarts_initial_T_i;
}

__device__ void cuModel::Clear_Optimizer(void)
{
    switch(this->type_optimizer_function)
    {
        case DL::OPTIMIZER::NONE: break;
        case DL::OPTIMIZER::GD:
            if(this->learning_momentum != 0.0f && this->ptr_array_previous_delta_parameters != nullptr)
            {
                Zero_1D<var>(this->total_parameters_allocated,
                                    this->ptr_array_previous_delta_parameters,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                break;
        case DL::OPTIMIZER::IRPROP_MINUS:
            // Previous train slopes.
            if(this->ptr_array_previous_derivatives_parameters != nullptr)
            {
                Zero_1D<var>(this->total_parameters_allocated,
                                    this->ptr_array_previous_derivatives_parameters,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
            // |END| Previous train slopes. |END|
                
            // Previous steps.
            if(this->ptr_array_previous_steps != nullptr)
            {
                Memory::Fill_1D<var>(this->total_parameters_allocated,
                                                                     this->ptr_array_previous_steps,
                                                                     this->rprop_delta_zero,
                                                                     this->ptr_array_dim3_grid + 1,
                                                                     this->ptr_array_dim3_block + 1);
            }
            // |END| Previous steps. |END|
                break;
        case DL::OPTIMIZER::IRPROP_PLUS:
            this->loss_rprop = FLT_MAX;
            this->loss_rprop_tm1 = FLT_MAX;
                
            // Previous train slopes.
            if(this->ptr_array_previous_derivatives_parameters != nullptr)
            {
                Zero_1D<var>(this->total_parameters_allocated,
                                    this->ptr_array_previous_derivatives_parameters,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
            // |END| Previous train slopes. |END|
                
            // Previous steps.
            if(this->ptr_array_previous_steps != nullptr)
            {
                Memory::Fill_1D<var>(this->total_parameters_allocated,
                                                                     this->ptr_array_previous_steps,
                                                                     this->rprop_delta_zero,
                                                                     this->ptr_array_dim3_grid + 1,
                                                                     this->ptr_array_dim3_block + 1);
            }
            // |END| Previous steps. |END|

            // Previous delta weights.
            if(this->ptr_array_previous_delta_parameters != nullptr)
            {
                Zero_1D<var>(this->total_parameters_allocated,
                                    this->ptr_array_previous_delta_parameters,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
            // |END| Previous delta weights. |END|
                break;
        case DL::OPTIMIZER::QUICKPROP: break;
        case DL::OPTIMIZER::SARPROP: break;
        case DL::OPTIMIZER::AMSGRAD:
            if(this->ptr_array_previous_biased_first_moment != nullptr)
            {
                Zero_1D<var>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_first_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                        
            if(this->ptr_array_previous_biased_second_moment != nullptr)
            {
                Zero_1D<var>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_second_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }

            if(this->ptr_array_previous_biased_second_moment_hat != nullptr)
            {
                Zero_1D<var>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_second_moment_hat,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                break;
        case DL::OPTIMIZER::ADAM:
        case DL::OPTIMIZER::ADAMAX:
            if(this->ptr_array_previous_biased_first_moment != nullptr)
            {
                Zero_1D<var>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_first_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                
            if(this->ptr_array_previous_biased_second_moment != nullptr)
            {
                Zero_1D<var>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_second_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                break;
        case DL::OPTIMIZER::NOSADAM:
            if(this->ptr_array_previous_biased_first_moment != nullptr)
            {
                Zero_1D<var>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_first_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }
                
            if(this->ptr_array_previous_biased_second_moment != nullptr)
            {
                Zero_1D<var>(this->total_parameters_allocated,
                                    this->ptr_array_previous_biased_second_moment,
                                    this->ptr_array_dim3_grid + 1,
                                    this->ptr_array_dim3_block + 1);
            }

            this->adam_previous_beta2 = 0_r;
                break;
        default:
            ERR(L"Can not reset parameters of the optimizer (%u).",
                        this->type_optimizer_function);
                break;
    }
        
    this->optimizer_time_step = 0_r;
    this->epoch_time_step = 1_r;
}
    
__global__ void kernel__cuModel__Set__Loss_Function(DL::LOSS_FN::TYPE const type_loss_function_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->set_loss_fn(type_loss_function_received); }

__host__ __device__ void cuModel::set_loss_fn(DL::LOSS_FN::TYPE const type_loss_function_received)
{
#ifndef __CUDA_ARCH__
    kernel__cuModel__Set__Loss_Function <<< 1, 1 >>> (type_loss_function_received, this);
#else
    this->type_loss_function = type_loss_function_received;
#endif
}

__global__ void kernel__cuModel__Set__Accuracy_Function(DL::ACCU_FN::TYPE const type_accuracy_function_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->set_accu_fn(type_accuracy_function_received); }

__host__ __device__ void cuModel::set_accu_fn(DL::ACCU_FN::TYPE const type_accuracy_function_received)
{
#ifndef __CUDA_ARCH__
    kernel__cuModel__Set__Accuracy_Function <<< 1, 1 >>> (type_accuracy_function_received, this);
#else
    this->type_accuracy_function = type_accuracy_function_received;
#endif
}

__global__ void kernel__cuModel__Set__Bit_Fail_Limit(var const bit_fail_limit, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->set_bit_fail_limit(bit_fail_limit); }

__host__ __device__ void cuModel::set_bit_fail_limit(var const bit_fail_limit)
{
#ifndef __CUDA_ARCH__
    kernel__cuModel__Set__Bit_Fail_Limit <<< 1, 1 >>> (bit_fail_limit, this);
#else
    this->bit_fail_limit = bit_fail_limit;
#endif
}
    
__global__ void kernel__cuModel__Set__Optimizer_Function(DL::OPTIMIZER::TYPE const type_optimizer_function_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->set_optimizer(type_optimizer_function_received); }

__host__ __device__ void cuModel::set_optimizer(DL::OPTIMIZER::TYPE const type_optimizer_function_received)
{
#ifndef __CUDA_ARCH__
    kernel__cuModel__Set__Optimizer_Function <<< 1, 1 >>> (type_optimizer_function_received, this);
#else
    if(this->type_optimizer_function == type_optimizer_function_received) { return; }
        
    // Deallocate old optimizer array.
    if(this->type_optimizer_function != DL::OPTIMIZER::NONE)
    { this->Deallocate__Parameter__Optimizer(); }
    // |END| Deallocate old optimizer array. |END|

    // allocate optimizer array.
    this->type_optimizer_function = type_optimizer_function_received;

    if(this->Allocate__Parameter__Optimizer() == false)
    {
        ERR(L"Can not allocate connections for optimizer function.",);

        return;
    }
    // |END| allocate optimizer array. |END|

    this->device__Clear_Train_Arrays();
#endif
}

__device__ real cuModel::warm_restarts_decay(void) {
  real const tmp_learning_rate_decay(
      this->warm_restarts_minimum_learning_rate +
      0.5_r *
          (this->warm_restarts_maximum_learning_rate -
           this->warm_restarts_minimum_learning_rate) *
          (1_r + cos(this->optimizer_time_step / this->warm_restarts_T_i *
                     DL::Math::PI<var>)));
    
    if(this->optimizer_time_step >= this->warm_restarts_T_i)
    {
        this->Clear_Optimizer();

        this->warm_restarts_T_i *= this->warm_restarts_multiplier;

        this->warm_restarts_maximum_learning_rate *= this->warm_restarts_decay_learning_rate;
    }

    return(tmp_learning_rate_decay);
}
    
// https://arxiv.org/pdf/1711.05101.pdf: Fixing Weight Decay Regularization in Adam
__device__ var  cuModel::normalized_wd(size_t const batch_size, size_t const training_size)
{ return(this->weight_decay * sqrt(batch_size / (training_size * this->epoch_time_step))); }

__device__ void cuModel::Update_Parameter(size_t const batch_size, size_t const training_size)
{
    if(this->get_l1() != 0_r)
    { this->Update_Derivative_Weight__Regularization__L1(batch_size); }

    if(this->get_l2() != 0_r)
    { this->Update_Derivative_Weight__Regularization__L2(batch_size); }
    
    switch(this->type_optimizer_function)
    {
        case DL::OPTIMIZER::GD:
        case DL::OPTIMIZER::IRPROP_MINUS:
        case DL::OPTIMIZER::IRPROP_PLUS:
        case DL::OPTIMIZER::QUICKPROP:
        case DL::OPTIMIZER::SARPROP:
        case DL::OPTIMIZER::ADAM:
        case DL::OPTIMIZER::ADAMAX:
        case DL::OPTIMIZER::AMSGRAD: 
        case DL::OPTIMIZER::NOSADAM: this->merge_mp_derivatives(); break;
        default:
          ERR(L"Unknow type optimizer function (%d | %ls) in the switch.",
              this->type_optimizer_function,
              DL::OPTIMIZER_NAME[this->type_optimizer_function].c_str());
                break;
    }

    switch(this->type_optimizer_function)
    {
        case DL::OPTIMIZER::GD: this->Update_Parameter__Gradient_Descent(batch_size, training_size, 0, this->total_parameters); break;
        case DL::OPTIMIZER::IRPROP_PLUS: this->Update_Parameter__iRPROP_plus(0, this->total_parameters); break;
        //case DL::OPTIMIZER::QUICKPROP: update_model_quickprop(this, this->get_n_data(), 0, this->total_parameters); break;
        //case DL::OPTIMIZER::SARPROP: update_model_sarprop(this, this->sarprop_epoch, 0, this->total_parameters); break;
        case DL::OPTIMIZER::ADAM: this->Update_Parameter__Adam(batch_size, training_size, 0, this->total_parameters); break;
        //case DL::OPTIMIZER::ADAMAX:
        //case DL::OPTIMIZER::TYPE_OPTIMIZER_SADAMAX: this->Update_Weight_AdaMax(0, this->total_parameters); break;
        case DL::OPTIMIZER::AMSGRAD: this->Update_Parameter__AMSGrad(batch_size, training_size, 0, this->total_parameters); break;
        default:
          ERR(L"Unknow type optimizer function (%d | %ls) in the switch.",
              this->type_optimizer_function,
              DL::OPTIMIZER_NAME[this->type_optimizer_function].c_str());
                break;
    }

    if(this->Get__Regularization__Max_Norm_Constraints() != 0_r)
    { this->Update_Weight_Regularization__Max_Norm_Constraints(); }

    this->Transpose_Weights();
}

__global__ void kernel__cuModel__Set__Accurancy_Variance(float const accurancy_variance_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Set__Accurancy_Variance(accurancy_variance_received); }

__host__ __device__ void cuModel::Set__Accurancy_Variance(float const accurancy_variance_received)
{
#ifndef __CUDA_ARCH__
    kernel__cuModel__Set__Accurancy_Variance <<< 1, 1 >>> (accurancy_variance_received, this);
        
    CUDA__Check_Error();
#else
    if(this->acc_var == accurancy_variance_received) { return; }

    this->acc_var = accurancy_variance_received;
#endif
}

__global__ void kernel__cuModel__Set__Time_Delays(size_t const time_delays_received, class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Set__Time_Delays(time_delays_received); }

__host__ __device__ void cuModel::Set__Time_Delays(size_t const time_delays_received)
{
#ifndef __CUDA_ARCH__
    kernel__cuModel__Set__Time_Delays <<< 1, 1 >>> (time_delays_received, this);
        
    CUDA__Check_Error();
#else
    if(this->n_time_delay == time_delays_received) { return; }

    this->n_time_delay = time_delays_received;
#endif
}
    
__device__ void cuModel::set_accu(DL::ENV::TYPE const type_accuracy_received, float const accurancy_received)
{
    switch(type_accuracy_received)
    {
        case DL::ENV::TRAIN: this->acc_train = accurancy_received; break;
        case DL::ENV::VALID: this->acc_valid = accurancy_received; break;
        case DL::ENV::TESTG: this->acc_testg = accurancy_received; break;
    }
}

__device__ void cuModel::set_loss(DL::ENV::TYPE const env_type, float const loss_received)
{
    switch(env_type)
    {
        case DL::ENV::TRAIN: this->loss_train = loss_received; break;
        case DL::ENV::VALID: this->loss_valid = loss_received; break;
        case DL::ENV::TESTG: this->loss_testg = loss_received; break;
    }
}

[[deprecated("Not properly implemented.")]] __host__ __device__ float cuModel::get_accu(DL::ENV::TYPE const env_type) const
{
#ifndef __CUDA_ARCH__
  // NotImplementedError
  // ...
  // ...
    return(100_r);
#else
    var tmp_accurancy;
        
    switch(env_type)
    {
        case DL::ENV::TRAIN: tmp_accurancy = this->acc_train; break;
        case DL::ENV::VALID: tmp_accurancy = this->acc_valid; break;
        case DL::ENV::TESTG: tmp_accurancy = this->acc_testg; break;
        case DL::ENV::NONE: tmp_accurancy = this->n_acc_trial == 0_UZ ? 0_r : this->ptr_array_accuracy_values[0][0] / static_cast<var>(this->n_acc_trial) * 100_r; break;
    }

    return(tmp_accurancy);
#endif
}

[[deprecated("Not properly implemented.")]] __host__ __device__ float cuModel::get_loss(DL::ENV::TYPE const env_type, size_t const number_digits_received) const
{
#ifndef __CUDA_ARCH__
  // NotImplementedError
  // ...
  // ...
    return(1_r);
#else
    float tmp_loss;

    switch(env_type)
    {
        case DL::ENV::TRAIN: tmp_loss = this->loss_train; break;
        case DL::ENV::VALID: tmp_loss = this->loss_valid; break;
        case DL::ENV::TESTG: tmp_loss = this->loss_testg; break;
        case DL::ENV::NONE:
            switch(this->type_loss_function)
            {
                case DL::LOSS_FN::ME: tmp_loss = this->get_me(); break;
                case DL::LOSS_FN::L1: tmp_loss = this->get_loss_l1(); break;
                case DL::LOSS_FN::MAE: tmp_loss = this->get_mae(); break;
                case DL::LOSS_FN::L2: tmp_loss = this->get_loss_l2(); break;
                case DL::LOSS_FN::MSE: tmp_loss = this->get_mse(); break;
                case DL::LOSS_FN::RMSE: tmp_loss = this->get_rmse(); break;
                case DL::LOSS_FN::MAPE: tmp_loss = this->get_mape(); break;
                case DL::LOSS_FN::SMAPE: tmp_loss = this->get_smape(); break;
                case DL::LOSS_FN::MASE_SEASONAL: tmp_loss = this->get_mase(); break;
                case DL::LOSS_FN::MASE_NON_SEASONAL: tmp_loss = this->get_mase(); break;
                case DL::LOSS_FN::CROSS_ENTROPY: tmp_loss = this->get_ace(); break;
                case DL::LOSS_FN::BIT: tmp_loss = this->get_bitfail(); break;
                default: tmp_loss = 1_r; break;
            }
                break;
        default: tmp_loss = 1_r; break;
    }

    return(tmp_loss);
#endif
}
    
__device__ float cuModel::get_me(void) const // https://en.wikipedia.org/wiki/Mean_absolute_error
{
    if(*this->ptr_array_number_loss != 0u)
    { return(*this->ptr_array_loss_values / static_cast<float>(*this->ptr_array_number_loss)); }
    else
    { return(1.0f); }
}
    
__device__ float cuModel::get_loss_l1(void) const
{ return(*this->ptr_array_loss_values); }
    
__device__ float cuModel::get_mae(void) const // https://en.wikipedia.org/wiki/Mean_absolute_error
{
    if(*this->ptr_array_number_loss != 0u)
    { return(*this->ptr_array_loss_values / static_cast<float>(*this->ptr_array_number_loss)); }
    else
    { return(1.0f); }
}
    
__device__ float cuModel::get_loss_l2(void) const
{ return(*this->ptr_array_loss_values); }
    
__device__ float cuModel::get_mse(void) const // https://en.wikipedia.org/wiki/Mean_squared_error
{
    if(*this->ptr_array_number_loss != 0u)
    { return(1.0f / static_cast<float>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values); }
    else
    { return(1.0f); }
}
    
__device__ float cuModel::get_rmse(void) const // https://en.wikipedia.org/wiki/Root-mean-square_deviation
{
    if(*this->ptr_array_number_loss != 0u)
    { return(sqrt(1.0f / static_cast<float>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values)); }
    else
    { return(1.0f); }
}
    
__device__ float cuModel::get_mape(void) const // https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
{
    if(*this->ptr_array_number_loss != 0u)
    { return(1.0f / static_cast<float>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values); }
    else
    { return(1.0f); }
}
    
__device__ float cuModel::get_smape(void) const // https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
{
    if(*this->ptr_array_number_loss != 0u)
    { return(1.0f / static_cast<float>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values); }
    else
    { return(1.0f); }
}

__device__ float cuModel::get_mase(void) const // https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
{
    // Non seasonal time series
    //if(*this->ptr_array_number_loss != 0u
    //   &&
    //   this->mean_absolute_error_denominator != 0.0f
    //   &&
    //   *this->ptr_array_number_loss > 1u)
    //{ return(1.0f / this->seq_w * (*this->ptr_array_loss_values / ((1.0f / static_cast<float>(this->seq_w - 1_UZ)) * this->mean_absolute_error_denominator))); }
    //else    { return(1.0f); }

    return(1.0f);
}

// TODO: optimize and check for the structure
__device__ float cuModel::get_ace(void) const // https://en.wikipedia.org/wiki/Cross_entropy
{
    if(*this->ptr_array_number_loss != 0u)
    { return(*this->ptr_array_loss_values / static_cast<float>(*this->ptr_array_number_loss)); }
    else
    { return(std::numeric_limits<float>::max()); }
}
//__device__ float cuModel::Get__CE(void) const // https://en.wikipedia.org/wiki/Cross_entropy
//{ return(*this->ptr_array_loss_values); }
    
__device__ float cuModel::get_bitfail(void) const // link
{ return(static_cast<float>(*this->ptr_array_number_bit_fail)); }

