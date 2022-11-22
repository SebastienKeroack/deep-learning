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

#include <curand_kernel.h>

__global__ void kernel__cuModel__Deallocate(class cuModel *const ptr_cuModel_received)
{ ptr_cuModel_received->Deallocate(); }

__host__ __device__ bool cuModel::Deallocate(void) {
#ifdef COMPILE_CUDA
    // Layer variable.
    SAFE_DELETE_ARRAY(this->ptr_array_number_neurons_by_layer); // size_t

    if(this->ptr_array_layers != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_layers->ptr_array_neuron_units);

        delete[](this->ptr_array_layers);
        this->ptr_array_layers = nullptr;
    }

    SAFE_DELETE_ARRAY(this->ptr_array_layers_Class_Storage_Dim3_Batch);
    // |END| Layer variable. |END|

    // Delete neurons variable.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_first_forward_connection_index); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_last_forward_connection_index); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_number_forward_connections); // delete[] array size_t.

    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_summations); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_values); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_values_hats); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_values_normalizes); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_means); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_variances); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_transposed_mean); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_transposed_variance); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_derivatives_means); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_derivatives_variances); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_r_corrections); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_d_corrections); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_means_averages); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_normalized_batch_units_variances_averages); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_errors); // delete[] array var.
        
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_type_activation_function); // delete[] array enum.

    this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
    
    this->Deallocate__Cell_Unit__Mask_Dropout_Zoneout();

    this->Deallocate__Neurons_Reduce_Summation();
    this->Deallocate__Neurons_Reduce_Error();
    this->Deallocate__Neurons_Reduce_Norms();

    this->Deallocate_Batch_Reduce();
    this->Deallocate__Normalized_Unit__Batch_Normalization();
    // |END| Delete neurons variable. |END|

    // Delete connections.
    SAFE_DELETE_ARRAY(this->ptr_array_transposed_weights); // delete[] array var.
    SAFE_DELETE_ARRAY(this->ptr_array_parameters); // delete[] array var.

    SAFE_DELETE_ARRAY(this->ptr_array_ptr_connections); // delete[] array void*.
        
    this->Deallocate__Parameter__Regularization();
    
    SAFE_DELETE_ARRAY(this->ptr_array_derivatives_parameters); // delete[] array var.
    // |END| Delete connections. |END|

    // Deallocate optimizer array.
    this->Deallocate__Parameter__Optimizer();
    // |END| Deallocate optimizer array. |END|

    // Deallocate cost.
    this->Deallocate_Cost();
    this->Deallocate_Reduce_Batch();
    this->Deallocate_Reduce_Cost();
    // |END| Deallocate cost. |END|

    // Delete cuRAND.
    if(this->ptr_array_cuRAND_State_MTGP32_weighted != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_cuRAND_State_MTGP32_weighted[0].k);

        delete[](this->ptr_array_cuRAND_State_MTGP32_weighted);
    }
    
    if(this->ptr_array_cuRAND_State_MTGP32_neuroyed != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_cuRAND_State_MTGP32_neuroyed[0].k);

        delete[](this->ptr_array_cuRAND_State_MTGP32_neuroyed);
    }
    // |END| Delete cuRAND |END|
        
    // Struct dim3 variable.
    SAFE_FREE(this->ptr_array_dim3_grid); // struct dim3.
    SAFE_FREE(this->ptr_array_dim3_block); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_grid_neurons); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_block_neurons); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_grid_neurons_DP); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_block_neurons_DP); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_grid_neurons_cuRAND); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_block_neurons_cuRAND); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_grid_batch_neurons); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_block_batch_neurons); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_grid_weights); // struct dim3.
    SAFE_FREE(this->ptr_array_layers_dim3_block_weights); // struct dim3.
    SAFE_FREE(this->ptr_array_neuron_units_dim3_grid_connections); // struct dim3.
    SAFE_FREE(this->ptr_array_neuron_units_dim3_block_connections); // struct dim3.

    SAFE_DELETE(this->ptr_Class_Storage_Dim3_Memcpy);
    // |END| Struct dim3 variable. |END|

    // Delete computation parameters.
    SAFE_DELETE(this->_ptr_Class_Device_Information_Array);
    // |END| Delete computation parameters |END|
#else
    kernel__cuModel__Deallocate <<< 1, 1 >>> (this);

    CUDA__Check_Error();
#endif

    return true;
}
    
__device__ void cuModel::Deallocate__Parameter__Optimizer(void)
{
    switch(this->type_optimizer_function)
    {
        case DL::OPTIMIZER::GD: this->Deallocate__Parameter__Gradient_Descent(); break;
        case DL::OPTIMIZER::IRPROP_MINUS: this->Deallocate__Parameter__iRPROP_minus(); break;
        case DL::OPTIMIZER::IRPROP_PLUS: this->Deallocate__Parameter__iRPROP_plus(); break;
        case DL::OPTIMIZER::ADAM:
        case DL::OPTIMIZER::ADAMAX:
        case DL::OPTIMIZER::NOSADAM: this->Deallocate__Parameter__Adam(); break;
        case DL::OPTIMIZER::AMSGRAD: this->Deallocate__Parameter__AMSGrad(); break;
        default:
          ERR(L"Unknow type optimizer function (%u | %ls) in the switch.",
              this->type_optimizer_function,
              DL::OPTIMIZER_NAME[this->type_optimizer_function].c_str());
                break;
    }
}

__device__ void cuModel::Deallocate__Parameter__Gradient_Descent(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_previous_delta_parameters);
}

__device__ void cuModel::Deallocate__Parameter__iRPROP_minus(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_previous_steps);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_derivatives_parameters);
}

__device__ void cuModel::Deallocate__Parameter__iRPROP_plus(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_previous_steps);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_delta_parameters);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_derivatives_parameters);
}

__device__ void cuModel::Deallocate__Parameter__Adam(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_first_moment);
    SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_second_moment);
}
    
__device__ void cuModel::Deallocate__Parameter__AMSGrad(void)
{
    this->Deallocate__Parameter__Adam();

    SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_second_moment_hat);
}

__device__ void cuModel::Deallocate__Parameter__Regularization(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_mask_regularized_parameters); // delete[] array var.
}

__device__ void cuModel::Deallocate_Cost(void)
{
    // Loss parameters.
    SAFE_DELETE_ARRAY(this->ptr_array_number_loss); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_number_bit_fail); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_loss_values); // delete[] array float.
    // |END| Loss parameters. |END|
    
    // Accuracy parameters.
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[0]); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[1]); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[2]); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[3]); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[4]); // delete[] array size_t.
    // |END| Accuracy parameters. |END|
}

__device__ void cuModel::Deallocate_Reduce_Batch(void)
{
    SAFE_FREE(this->ptr_array_dim3_grid_reduce_threads);
    SAFE_FREE(this->ptr_array_dim3_block_reduce_threads);

    SAFE_FREE(this->ptr_array_dim3_grid_reduce_threads_DP);
    SAFE_FREE(this->ptr_array_dim3_block_reduce_threads_DP);
}

__device__ void cuModel::Deallocate_Reduce_Cost(void)
{
    // Loss parameters.
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_number_loss);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_bit_fail_values);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_loss_values);
    // |END| Loss parameters. |END|
    
    // Accuracy parameters.
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_accuracy_values[0]);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_accuracy_values[1]);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_accuracy_values[2]);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_accuracy_values[3]);
    SAFE_DELETE_ARRAY(this->ptr_array_reduce_accuracy_values[4]);
    // |END| Accuracy parameters. |END|
}

__device__ void cuModel::Deallocate_Batch_Reduce(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_reduce_batch_size); // delete[] array size_t.

    SAFE_FREE(this->ptr_array_neuron_units_dim3_grid_reduce_batch); // free array dim3.
    SAFE_FREE(this->ptr_array_neuron_units_dim3_block_reduce_batch); // free array dim3.
}

__device__ void cuModel::Deallocate__Normalized_Unit__Batch_Normalization(void)
{
    if(this->ptr_array_2D_neurons_reduce_batch_mean != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_2D_neurons_reduce_batch_mean[0]); // delete[] array var.

        delete[](this->ptr_array_2D_neurons_reduce_batch_mean); // delete[] array var.
        this->ptr_array_2D_neurons_reduce_batch_mean = nullptr;
    }

    if(this->ptr_array_2D_neurons_reduce_batch_variance != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_2D_neurons_reduce_batch_variance[0]); // delete[] array var.

        delete[](this->ptr_array_2D_neurons_reduce_batch_variance); // delete[] array var.
        this->ptr_array_2D_neurons_reduce_batch_variance = nullptr;
    }
}

__device__ void cuModel::Deallocate__Neuron__Mask_Dropout_Bernoulli(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_af_units_mask_dropout_bernoulli); // delete[] array bool.
}

__device__ void cuModel::Deallocate__Cell_Unit__Mask_Dropout_Zoneout(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_cell_units_mask_dropout_zoneout); // delete[] array bool.
}

__device__ void cuModel::Deallocate__Neurons_Reduce_Summation(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_reduce_summation_size); // delete[] array size_t.

    if(this->ptr_array_2D_neurons_reduce_summation != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_2D_neurons_reduce_summation[0]); // delete[] array var.

        delete[](this->ptr_array_2D_neurons_reduce_summation); // delete[] array var.
        this->ptr_array_2D_neurons_reduce_summation = nullptr;
    }

    SAFE_FREE(this->ptr_array_neuron_units_dim3_grid_reduce_summation); // free array dim3.
    SAFE_FREE(this->ptr_array_neuron_units_dim3_block_reduce_summation); // free array dim3.
}

__device__ void cuModel::Deallocate__Neurons_Reduce_Error(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_reduce_error_size); // delete[] array size_t.

    if(this->ptr_array_2D_neurons_reduce_error != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_2D_neurons_reduce_error[0]); // delete[] array var.

        delete[](this->ptr_array_2D_neurons_reduce_error); // delete[] array var.
        this->ptr_array_2D_neurons_reduce_error = nullptr;
    }

    SAFE_FREE(this->ptr_array_neuron_units_dim3_grid_reduce_error); // free array dim3.
    SAFE_FREE(this->ptr_array_neuron_units_dim3_block_reduce_error); // free array dim3.
}

__device__ void cuModel::Deallocate__Neurons_Reduce_Norms(void)
{
    SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_reduce_norms_size); // delete[] array size_t.
    SAFE_DELETE_ARRAY(this->ptr_array_neuroyed_number_neurons_in_layer); // delete[] array size_t.

    if(this->ptr_array_2D_neurons_reduce_norms != nullptr)
    {
        SAFE_DELETE_ARRAY(this->ptr_array_2D_neurons_reduce_norms[0]); // delete[] array var.

        delete[](this->ptr_array_2D_neurons_reduce_norms); // delete[] array var.
        this->ptr_array_2D_neurons_reduce_norms = nullptr;
    }

    if(this->ptr_array_2D_neurons_dim3_grid_reduce_norms != NULL)
    {
        SAFE_FREE(this->ptr_array_2D_neurons_dim3_grid_reduce_norms[0]); // free array dim3.

        free(this->ptr_array_2D_neurons_dim3_grid_reduce_norms); // free array dim3.
        this->ptr_array_2D_neurons_dim3_grid_reduce_norms = NULL;
    }
    
    if(this->ptr_array_2D_neurons_dim3_block_reduce_norms != NULL)
    {
        SAFE_FREE(this->ptr_array_2D_neurons_dim3_block_reduce_norms[0]); // free array dim3.

        free(this->ptr_array_2D_neurons_dim3_block_reduce_norms); // free array dim3.
        this->ptr_array_2D_neurons_dim3_block_reduce_norms = NULL;
    }
}
