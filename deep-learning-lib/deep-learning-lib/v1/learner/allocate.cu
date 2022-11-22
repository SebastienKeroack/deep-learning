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
#include "deep-learning-lib/ops/fill.cuh"
#include "deep-learning-lib/ops/zero.cuh"
#include "deep-learning-lib/ops/transpose.cuh"
#include "deep-learning-lib/io/logger.hpp"

__device__ bool cuModel::Allocate_Weights_Transposed(void)
{
    if(this->total_weights_allocated == 0u)
    {
        ERR(L"Can not allocate memory! Total weights allocated equal zero.",);

        return false;
    }
    else if(this->ptr_array_transposed_weights == nullptr)
    {
        this->ptr_array_transposed_weights = new var[this->total_weights_allocated];
        if(this->ptr_array_transposed_weights == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->total_weights_allocated,
                            this->ptr_array_transposed_weights,
                            this->ptr_array_dim3_grid + 2,
                            this->ptr_array_dim3_block + 2);
    }

    return true;
}

__device__ bool cuModel::Allocate__Parameter(void)
{
    if(this->total_parameters_allocated == 0u)
    {
        this->ptr_array_parameters = new var[this->total_parameters];
        if(this->ptr_array_parameters == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->total_parameters,
                            this->ptr_array_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
        
        this->ptr_array_ptr_connections = new void*[this->total_parameters];
        if(this->ptr_array_ptr_connections == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<void*>(this->total_parameters,
                                this->ptr_array_ptr_connections,
                                this->ptr_array_dim3_grid + 1,
                                this->ptr_array_dim3_block + 1);

        this->total_weights_allocated = this->total_weights;
            
        this->total_parameters_allocated = this->total_parameters;
    }
    else
    {
        ERR(L"Can not allocate on allocated memory! Use reallocate function.",);

        return false;
    }

    return true;
}

__device__ bool cuModel::Allocate__Parameter__Optimizer(void)
{
    switch(this->type_optimizer_function)
    {
        case DL::OPTIMIZER::GD: return(this->Allocate__Parameter__Gradient_Descent());
        case DL::OPTIMIZER::IRPROP_MINUS: return(this->Allocate__Parameter__iRPROP_minus());
        case DL::OPTIMIZER::IRPROP_PLUS: return(this->Allocate__Parameter__iRPROP_plus());
        case DL::OPTIMIZER::ADAM:
        case DL::OPTIMIZER::ADAMAX:
        case DL::OPTIMIZER::NOSADAM: return(this->Allocate__Parameter__Adam());
        case DL::OPTIMIZER::AMSGRAD: return(this->Allocate__Parameter__AMSGrad());
        default:
          ERR(L"Unknow type optimizer function (%u | %ls) in the switch.",
              this->type_optimizer_function,
              DL::OPTIMIZER_NAME[this->type_optimizer_function].c_str());
                return false;
    }
}
    
__device__ bool cuModel::Allocate__Parameter__Gradient_Descent(void)
{
    if(this->learning_momentum != 0_r
        &&
        this->ptr_array_previous_delta_parameters == nullptr)
    {
        this->ptr_array_previous_delta_parameters = new var[this->total_parameters];
        if(this->ptr_array_previous_delta_parameters == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->total_parameters,
                            this->ptr_array_previous_delta_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return true;
}

__device__ bool cuModel::Allocate__Parameter__iRPROP_minus(void)
{
    if(this->ptr_array_previous_steps == nullptr)
    {
        this->ptr_array_previous_steps = new var[this->total_parameters];
        if(this->ptr_array_previous_steps == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Memory::Fill_1D<var>(this->total_parameters,
                                                             this->ptr_array_previous_steps,
                                                             this->rprop_delta_zero,
                                                             this->ptr_array_dim3_grid + 1,
                                                             this->ptr_array_dim3_block + 1);
    }
    
    if(this->ptr_array_previous_derivatives_parameters == nullptr)
    {
        this->ptr_array_previous_derivatives_parameters = new var[this->total_parameters];
        if(this->ptr_array_previous_derivatives_parameters == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->total_parameters,
                            this->ptr_array_previous_derivatives_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return true;
}

__device__ bool cuModel::Allocate__Parameter__iRPROP_plus(void)
{
    if(this->ptr_array_previous_steps == nullptr)
    {
        this->ptr_array_previous_steps = new var[this->total_parameters];
        if(this->ptr_array_previous_steps == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Memory::Fill_1D<var>(this->total_parameters,
                                                             this->ptr_array_previous_steps,
                                                             this->rprop_delta_zero,
                                                             this->ptr_array_dim3_grid + 1,
                                                             this->ptr_array_dim3_block + 1);
    }
    
    if(this->ptr_array_previous_delta_parameters == nullptr)
    {
        this->ptr_array_previous_delta_parameters = new var[this->total_parameters];
        if(this->ptr_array_previous_delta_parameters == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->total_parameters,
                            this->ptr_array_previous_delta_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }
    
    if(this->ptr_array_previous_derivatives_parameters == nullptr)
    {
        this->ptr_array_previous_derivatives_parameters = new var[this->total_parameters];
        if(this->ptr_array_previous_derivatives_parameters == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->total_parameters,
                            this->ptr_array_previous_derivatives_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return true;
}

__device__ bool cuModel::Allocate__Parameter__Adam(void)
{
    if(this->ptr_array_previous_biased_first_moment == nullptr)
    {
        this->ptr_array_previous_biased_first_moment = new var[this->total_parameters];
        if(this->ptr_array_previous_biased_first_moment == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->total_parameters,
                            this->ptr_array_previous_biased_first_moment,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }
    
    if(this->ptr_array_previous_biased_second_moment == nullptr)
    {
        this->ptr_array_previous_biased_second_moment = new var[this->total_parameters];
        if(this->ptr_array_previous_biased_second_moment == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->total_parameters,
                            this->ptr_array_previous_biased_second_moment,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return true;
}
    
__device__ bool cuModel::Allocate__Parameter__AMSGrad(void)
{
    this->Allocate__Parameter__Adam();
    
    if(this->ptr_array_previous_biased_second_moment_hat == nullptr)
    {
        this->ptr_array_previous_biased_second_moment_hat = new var[this->total_parameters];
        if(this->ptr_array_previous_biased_second_moment_hat == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->total_parameters,
                            this->ptr_array_previous_biased_second_moment_hat,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return true;
}

__global__ void kernel__cuModel__Allocate_Structure(size_t const total_layers_received,
                                                                                            size_t const maximum_allowable_memory_bytes_received,
                                                                                            class cuModel *const ptr_cuModel_received)
{
    if(ptr_cuModel_received->Allocate__Structure(total_layers_received, maximum_allowable_memory_bytes_received) == false)
    {
        ERR(L"An error has been triggered from the \"Allocate__Structure(%u, %zu)\" function.",
                                 total_layers_received,
                                 maximum_allowable_memory_bytes_received);

        return;
    }
}

__host__ __device__ bool cuModel::Allocate__Structure(size_t const n_layers, size_t const maximum_allowable_memory_bytes_received)
{
#ifdef __CUDA_ARCH__
    // Dimension.
    this->total_layers = n_layers;
    this->n_inp = 0u;
    this->n_out = 0u;
    this->total_neuron_units_allocated = this->total_neuron_units = 0u;
    this->total_block_units_allocated = this->total_block_units = 0u;
    this->total_cell_units_allocated = this->total_cell_units = 0u;
    this->total_parameters_allocated = this->total_parameters = 0u;
    this->total_weights_allocated = this->total_weights = 0u;
    
    size_t *tmp_ptr_array_number_neurons_by_layer(this->ptr_array_number_neurons_by_layer = new size_t[n_layers]);
    if(tmp_ptr_array_number_neurons_by_layer == nullptr)
    {
        ERR(L"Can not allocate memory. new size_t[nLayer(%u)]",
                                n_layers);

        return false;
    }
    memset(tmp_ptr_array_number_neurons_by_layer,
                0,
                n_layers * sizeof(size_t));
    
    //    allocate layers.
    struct cuLayer *layer_it(this->ptr_array_layers = new struct cuLayer[n_layers]);
    if(layer_it == nullptr)
    {
        ERR(L"Can not allocate memory. new size_t[nLayer(%u)]",
                                n_layers);

        return false;
    }
    struct cuLayer const *const last_layer(this->ptr_last_layer = layer_it + n_layers);
    //    |END| allocate layers. |END|
    
    // allocate dim3 neurons by layer.
    struct dim3 *tmp_ptr_array_layers_dim3_grid_neurons(static_cast<struct dim3*>(malloc(n_layers * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_grid_neurons == NULL)
    {
        ERR(L"Can not allocate memory. malloc(nLayers(%u) * sizeof(%u))",
                                n_layers,
                                sizeof(struct dim3));

        return false;
    }
    memset(tmp_ptr_array_layers_dim3_grid_neurons,
                    0,
                    n_layers * sizeof(struct dim3));
    this->ptr_array_layers_dim3_grid_neurons = tmp_ptr_array_layers_dim3_grid_neurons;

    struct dim3 *tmp_ptr_array_layers_dim3_block_neurons(static_cast<struct dim3*>(malloc(n_layers * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_block_neurons == NULL)
    {
        ERR(L"Can not allocate memory. malloc(nLayers(%u) * sizeof(%u))",
                                n_layers,
                                sizeof(struct dim3));

        return false;
    }
    memset(tmp_ptr_array_layers_dim3_block_neurons,
                    0,
                    n_layers * sizeof(struct dim3));
    this->ptr_array_layers_dim3_block_neurons = tmp_ptr_array_layers_dim3_block_neurons;
    // |END| allocate dim3 neurons by layer. |END|
    
    // allocate dim3 neurons dynamic parallelisme by layer.
    struct dim3 *tmp_ptr_array_layers_dim3_grid_neurons_DP(static_cast<struct dim3*>(malloc(n_layers * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_grid_neurons_DP == NULL)
    {
        ERR(L"Can not allocate memory. malloc(nLayers(%u) * sizeof(%u))",
                                n_layers,
                                sizeof(struct dim3));

        return false;
    }
    memset(tmp_ptr_array_layers_dim3_grid_neurons_DP,
                    0,
                    n_layers * sizeof(struct dim3));
    this->ptr_array_layers_dim3_grid_neurons_DP = tmp_ptr_array_layers_dim3_grid_neurons_DP;

    struct dim3 *tmp_ptr_array_layers_dim3_block_neurons_DP(static_cast<struct dim3*>(malloc(n_layers * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_block_neurons_DP == NULL)
    {
        ERR(L"Can not allocate memory. malloc(nLayers(%u) * sizeof(%u))",
                                n_layers,
                                sizeof(struct dim3));

        return false;
    }
    memset(tmp_ptr_array_layers_dim3_block_neurons_DP,
                    0,
                    n_layers * sizeof(struct dim3));
    this->ptr_array_layers_dim3_block_neurons_DP = tmp_ptr_array_layers_dim3_block_neurons_DP;
    // |END| allocate dim3 neurons dynamic parallelisme by layer. |END|
    
    // allocate dim3 neurons cuRAND by layer.
    struct dim3 *tmp_ptr_array_layers_dim3_grid_neurons_cuRAND(static_cast<struct dim3*>(malloc(n_layers * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_grid_neurons_cuRAND == NULL)
    {
        ERR(L"Can not allocate memory. malloc(nLayers(%u) * sizeof(%u))",
                                n_layers,
                                sizeof(struct dim3));

        return false;
    }
    memset(tmp_ptr_array_layers_dim3_grid_neurons_cuRAND,
                    0,
                    n_layers * sizeof(struct dim3));
    this->ptr_array_layers_dim3_grid_neurons_cuRAND = tmp_ptr_array_layers_dim3_grid_neurons_cuRAND;

    struct dim3 *tmp_ptr_array_layers_dim3_block_neurons_cuRAND(static_cast<struct dim3*>(malloc(n_layers * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_block_neurons_cuRAND == NULL)
    {
        ERR(L"Can not allocate memory. malloc(nLayers(%u) * sizeof(%u))",
                                n_layers,
                                sizeof(struct dim3));

        return false;
    }
    memset(tmp_ptr_array_layers_dim3_block_neurons_cuRAND,
                    0,
                    n_layers * sizeof(struct dim3));
    this->ptr_array_layers_dim3_block_neurons_cuRAND = tmp_ptr_array_layers_dim3_block_neurons_cuRAND;
    // |END| allocate dim3 neurons cuRAND by layer. |END|
    
    // allocate dim3 batch neurons by layer.
    struct dim3 *tmp_ptr_array_layers_dim3_grid_batch_neurons(static_cast<struct dim3*>(malloc(n_layers * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_grid_batch_neurons == NULL)
    {
        ERR(L"Can not allocate memory. malloc(nLayers(%u) * sizeof(%u))",
                                n_layers,
                                sizeof(struct dim3));

        return false;
    }
    memset(tmp_ptr_array_layers_dim3_grid_batch_neurons,
                    0,
                    n_layers * sizeof(struct dim3));
    this->ptr_array_layers_dim3_grid_batch_neurons = tmp_ptr_array_layers_dim3_grid_batch_neurons;

    struct dim3 *tmp_ptr_array_layers_dim3_block_batch_neurons(static_cast<struct dim3*>(malloc(n_layers * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_block_batch_neurons == NULL)
    {
        ERR(L"Can not allocate memory. malloc(nLayers(%u) * sizeof(%u))",
                                n_layers,
                                sizeof(struct dim3));

        return false;
    }
    memset(tmp_ptr_array_layers_dim3_block_batch_neurons,
                    0,
                    n_layers * sizeof(struct dim3));
    this->ptr_array_layers_dim3_block_batch_neurons = tmp_ptr_array_layers_dim3_block_batch_neurons;
    // |END| allocate dim3 batch neurons by layer. |END|
    
    // allocate dim3 weights by layer.
    struct dim3 *tmp_ptr_array_layers_dim3_grid_weights(static_cast<struct dim3*>(malloc(n_layers * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_grid_weights == NULL)
    {
        ERR(L"Can not allocate memory. malloc(nLayers(%u) * sizeof(%u))",
                                n_layers,
                                sizeof(struct dim3));

        return false;
    }
    memset(tmp_ptr_array_layers_dim3_grid_weights,
                    0,
                    n_layers * sizeof(struct dim3));
    this->ptr_array_layers_dim3_grid_weights = tmp_ptr_array_layers_dim3_grid_weights;

    struct dim3 *tmp_ptr_array_layers_dim3_block_weights(static_cast<struct dim3*>(malloc(n_layers * sizeof(struct dim3))));
    if(tmp_ptr_array_layers_dim3_block_weights == NULL)
    {
        ERR(L"Can not allocate memory. malloc(nLayers(%u) * sizeof(%u))",
                                n_layers,
                                sizeof(struct dim3));

        return false;
    }
    memset(tmp_ptr_array_layers_dim3_block_weights,
                    0,
                    n_layers * sizeof(struct dim3));
    this->ptr_array_layers_dim3_block_weights = tmp_ptr_array_layers_dim3_block_weights;
    // |END| allocate dim3 weights by layer. |END|
    
    //    allocate storage dim3 batch.
    class cuDims *tmp_ptr_array_storage_dim3(this->ptr_array_layers_Class_Storage_Dim3_Batch = new class cuDims[n_layers]);
    if(tmp_ptr_array_storage_dim3 == nullptr)
    {
        ERR(L"Can not allocate memory. new size_t[nLayer(%u)]",
                                n_layers);

        return false;
    }
    //    |END| allocate storage dim3 batch. |END|

    for(; layer_it != last_layer; ++layer_it)
    {
        layer_it->dropout_values[0] = 1_r;
        layer_it->dropout_values[1] = 0_r;

        layer_it->ptr_array_neuron_units = nullptr;
        layer_it->ptr_last_neuron_unit = nullptr;

        layer_it->ptr_number_neurons = tmp_ptr_array_number_neurons_by_layer++;
        
        layer_it->ptr_dim3_grid_neurons = tmp_ptr_array_layers_dim3_grid_neurons++;
        layer_it->ptr_dim3_block_neurons = tmp_ptr_array_layers_dim3_block_neurons++;
        
        layer_it->ptr_dim3_grid_neurons_DP = tmp_ptr_array_layers_dim3_grid_neurons_DP++;
        layer_it->ptr_dim3_block_neurons_DP = tmp_ptr_array_layers_dim3_block_neurons_DP++;
        
        layer_it->ptr_dim3_grid_neurons_cuRAND = tmp_ptr_array_layers_dim3_grid_neurons_cuRAND++;
        layer_it->ptr_dim3_block_neurons_cuRAND = tmp_ptr_array_layers_dim3_block_neurons_cuRAND++;
        
        layer_it->ptr_dim3_grid_batch_neurons = tmp_ptr_array_layers_dim3_grid_batch_neurons++;
        layer_it->ptr_dim3_block_batch_neurons = tmp_ptr_array_layers_dim3_block_batch_neurons++;
        
        layer_it->ptr_dim3_grid_weights = tmp_ptr_array_layers_dim3_grid_weights++;
        layer_it->ptr_dim3_block_weights = tmp_ptr_array_layers_dim3_block_weights++;

        layer_it->ptr_Class_Storage_Dim3_Batch = tmp_ptr_array_storage_dim3++;
    }

    this->ptr_array_transposed_weights = nullptr;
    this->ptr_array_parameters = nullptr;
    this->ptr_array_ptr_connections = nullptr;
    // |END| Dimension. |END|
        
    //    allocate storage dim3 memcpy.
    this->ptr_Class_Storage_Dim3_Memcpy = new class cuDims;
    if(this->ptr_Class_Storage_Dim3_Memcpy == nullptr)
    {
        ERR(L"Can not allocate memory. new size_t[nLayer(%u)]",
                                n_layers);

        return false;
    }
    //    |END| allocate storage dim3 memcpy. |END|

    // General parameters.
    this->type = DL::MODEL::FEEDFORWARD;
    this->connection_rate = 0_r;
    this->seq_w = 1u;
    this->n_time_delay = 0u;
    this->type_state_propagation = DL::PROPAGATION::INFERENCE;
    // |END| General parameters. |END|
        
    // Gradient descent parameters.
    this->learning_rate = 0.01_r;
    this->learning_momentum = 0.9_r;
    this->use_nesterov = true;
    this->ptr_array_previous_delta_parameters = nullptr;
    // |END| Gradient descent parameters. |END|
    
    // Quickprop parameters.
    this->quickprop_decay = -0.0001f;
    this->quickprop_mu = 1.75f;
    // |END| Quickprop parameters. |END|

    // Resillent propagation parameters.
    this->rprop_increase_factor = 1.2f;
    this->rprop_decrease_factor = 0.5f;
    this->rprop_delta_min = 1e-6f;
    this->rprop_delta_max = 50.0f;
    this->rprop_delta_zero = 0.1f;
    this->ptr_array_previous_steps = nullptr;
    //this->ptr_array_previous_delta_parameters = nullptr;
    this->ptr_array_previous_derivatives_parameters = nullptr;
    this->loss_rprop = FLT_MAX;
    this->loss_rprop_tm1 = FLT_MAX;
    // |END| Resillent propagation parameters. |END|
        
    // SARProp parameters.
     this->sarprop_weight_decay_shift = -6.644f;
     this->sarprop_step_error_threshold_factor = 0.1f;
     this->sarprop_step_error_shift = 1.385f;
     this->sarprop_temperature = 0.015f;
     this->sarprop_epoch = 0u;
    // |END| SARProp parameters. |END|
        
    // AMSGrad parameters.
    //    Adam parameters.
     this->adam_learning_rate = 0.001_r;
     this->adam_beta1 = 0.9_r;
     this->adam_beta2 = 0.999_r;
     this->adam_epsilon = 1.0e-8_r;
     this->use_adam_bias_correction = true;
     this->adam_gamma = 0.1_r;
     this->ptr_array_previous_biased_first_moment = nullptr;
     this->ptr_array_previous_biased_second_moment = nullptr;
    //    |END| Adam parameters. |END|
     this->ptr_array_previous_biased_second_moment_hat = nullptr;
    // |END| AMSGrad parameters. |END|
        
    // Warm restarts parameters.
    this->use_warm_restarts = false;
    this->warm_restarts_decay_learning_rate = 1_r;
    this->warm_restarts_maximum_learning_rate = this->warm_restarts_initial_maximum_learning_rate = 1_r;
    this->warm_restarts_minimum_learning_rate = 1.0e-7_r;
    this->warm_restarts_T_i = this->warm_restarts_initial_T_i = 1_r;
    this->warm_restarts_multiplier = 2_r;
    // |END| Warm restarts parameters. |END|

    // Training parameters.
    this->type_optimizer_function = DL::OPTIMIZER::NONE;
    this->type_loss_function = DL::LOSS_FN::NONE;
    this->bit_fail_limit = 0.35_r;
    this->ptr_array_derivatives_parameters = nullptr;
    this->optimizer_time_step = 0_r;
    this->epoch_time_step = 1_r;
    // |END| Training parameters. |END|
        
    // Regularization parameters.
    this->use_Dropout = false;
    this->ptr_array_mask_dropout_parameters = nullptr;
    this->ptr_array_mask_regularized_parameters = nullptr;
    this->regularization__max_norm_constraints = 0_r;
    this->regularization__l1 = 0_r;
    this->regularization__l2 = 0_r;
    this->weight_decay = 0_r;
    // |END| Regularization parameters. |END|
        
    // Normalization parameters.
    this->use_Batch_Renormalization = false;
    this->normalization_momentum_average = 0.01_r;
    this->normalization_epsilon = 1.0e-5_r;
    this->batch_renormalization_r_correction_maximum = 1_r;
    this->batch_renormalization_d_correction_maximum = 0_r;
    // |END| Normalization parameters. |END|

    // Loss parameters.
    this->ptr_array_number_loss = new size_t[1]; *this->ptr_array_number_loss = 0_UZ;
    this->ptr_array_number_bit_fail = new size_t[1]; *this->ptr_array_number_bit_fail = 0_UZ;
    this->ptr_array_loss_values = new var[1]; *this->ptr_array_loss_values = (std::numeric_limits<var>().max)();
    this->loss_train = FLT_MAX;
    this->loss_valid = FLT_MAX;
    this->loss_testg = FLT_MAX;
    // |END| Loss parameters. |END|
        
    // Accuracy parameters.
    if((this->ptr_array_accuracy_values[0] = new var[1]) == nullptr)
    {
        ERR(L"Can not allocate %zu bytes. At line %d.",
                                 sizeof(var),
                                 __LINE__);

        return false;
    }
    else { this->ptr_array_accuracy_values[0][0] = 0_r; }

    if((this->ptr_array_accuracy_values[1] = new var[1]) == nullptr)
    {
        ERR(L"Can not allocate %zu bytes. At line %d.",
                                 sizeof(var),
                                 __LINE__);

        return false;
    }
    else { this->ptr_array_accuracy_values[1][0] = 0_r; }

    if((this->ptr_array_accuracy_values[2] = new var[1]) == nullptr)
    {
        ERR(L"Can not allocate %zu bytes. At line %d.",
                                 sizeof(var),
                                 __LINE__);

        return false;
    }
    else { this->ptr_array_accuracy_values[0][2] = 0_r; }

    if((this->ptr_array_accuracy_values[3] = new var[1]) == nullptr)
    {
        ERR(L"Can not allocate %zu bytes. At line %d.",
                                 sizeof(var),
                                 __LINE__);

        return false;
    }
    else { this->ptr_array_accuracy_values[0][3] = 0_r; }
    
    if((this->ptr_array_accuracy_values[4] = new var[1]) == nullptr)
    {
        ERR(L"Can not allocate %zu bytes. At line %d.",
                                 sizeof(var),
                                 __LINE__);

        return false;
    }
    else { this->ptr_array_accuracy_values[0][4] = 0_r; }

    this->n_acc_trial = 0u;
    this->acc_var = 0.0f;
    this->acc_train = 0.0f;
    this->acc_valid = 0.0f;
    this->acc_testg = 0.0f;
    // |END| Accuracy parameters. |END|

    // Computation parameters.
    this->limit_device_runtime_pending_launch_count = 2048u; // Default fixed pool size.
    this->number_threads = 1u;
    this->cache_number_threads = 0u;
    this->batch_size = 1u;
    this->cache_batch_size = 0u;
    this->maximum_allowable_memory_bytes = maximum_allowable_memory_bytes_received;
    this->_ptr_Class_Device_Information_Array = nullptr;
    // |END| Computation parameters. |END|

    // cuRAND parameters.
    this->number_cuRAND_State_MTGP32_weighted = 0u;
    this->number_cuRAND_State_MTGP32_neuroyed = 0u;

    this->ptr_array_cuRAND_State_MTGP32_weighted = nullptr;
    this->ptr_array_cuRAND_State_MTGP32_neuroyed = nullptr;
    // |END| cuRAND parameters. |END|

    // Neurons variable.
    this->ptr_array_af_units_mask_dropout_bernoulli = nullptr;
    this->ptr_array_cell_units_mask_dropout_zoneout = nullptr;

    this->neurons_total_reduce_summation_size = 0u;
    this->neurons_total_reduce_error_size = 0u;
    this->neurons_total_reduce_batch_size = 0u;
    this->neurons_total_reduce_norms_size = 0u;

    this->ptr_array_neuron_units_first_forward_connection_index = nullptr;
    this->ptr_array_neuron_units_last_forward_connection_index = nullptr;
    this->ptr_array_neuron_units_number_forward_connections = nullptr;
    this->ptr_array_neuron_units_reduce_summation_size = nullptr;
    this->ptr_array_neuron_units_reduce_error_size = nullptr;
    this->ptr_array_neuron_units_reduce_batch_size = nullptr;
    this->ptr_array_neuron_units_reduce_norms_size = nullptr;
    this->ptr_array_neuroyed_number_neurons_in_layer = nullptr;

    this->ptr_array_neuron_units_summations = nullptr;
    this->ptr_array_neuron_units_values = nullptr;
    this->ptr_array_normalized_batch_units_values_hats = nullptr;
    this->ptr_array_normalized_batch_units_values_normalizes = nullptr;
    this->ptr_array_normalized_batch_units_means = nullptr;
    this->ptr_array_normalized_batch_units_variances = nullptr;
    this->ptr_array_neuron_units_transposed_mean = nullptr;
    this->ptr_array_neuron_units_transposed_variance = nullptr;
    this->ptr_array_normalized_batch_units_derivatives_means = nullptr;
    this->ptr_array_normalized_batch_units_derivatives_variances = nullptr;
    this->ptr_array_normalized_batch_units_means_averages = nullptr;
    this->ptr_array_normalized_batch_units_variances_averages = nullptr;
    this->ptr_array_normalized_batch_units_r_corrections = nullptr;
    this->ptr_array_normalized_batch_units_d_corrections = nullptr;
    this->ptr_array_normalized_batch_units_scales = nullptr;
    this->ptr_array_normalized_batch_units_shifts = nullptr;
    this->ptr_array_neuron_units_errors = nullptr;
    this->ptr_array_2D_neurons_reduce_summation = nullptr;
    this->ptr_array_2D_neurons_reduce_error = nullptr;
    this->ptr_array_2D_neurons_reduce_batch_mean = nullptr;
    this->ptr_array_2D_neurons_reduce_batch_variance = nullptr;
    this->ptr_array_2D_neurons_reduce_norms = nullptr;
    this->ptr_array_mask_dropout_parameters = nullptr;

    this->ptr_array_neuron_units_type_activation_function = nullptr;
    // |END| Neurons variable. |END|

    this->ptr_array_dim3_grid = static_cast<struct dim3*>(malloc(TOTAL_KERNEL_PARALLEL * sizeof(struct dim3)));
    if(this->ptr_array_dim3_grid == NULL)
    {
        ERR(L"Can not allocate memory. new size_t[TOTAL_KERNEL_PARALLEL(%u)]",
                                TOTAL_KERNEL_PARALLEL);

        return false;
    }
    memset(this->ptr_array_dim3_grid,
                0,
                TOTAL_KERNEL_PARALLEL * sizeof(struct dim3));

    this->ptr_array_dim3_block = static_cast<struct dim3*>(malloc(TOTAL_KERNEL_PARALLEL * sizeof(struct dim3)));
    if(this->ptr_array_dim3_block == NULL)
    {
        ERR(L"Can not allocate memory. new size_t[TOTAL_KERNEL_PARALLEL(%u)]",
                                TOTAL_KERNEL_PARALLEL);

        return false;
    }
    memset(this->ptr_array_dim3_block,
                0,
                TOTAL_KERNEL_PARALLEL * sizeof(struct dim3));
    
    // Struct dim3 variable.
    this->ptr_array_neuron_units_dim3_grid_connections = NULL;
    this->ptr_array_neuron_units_dim3_block_connections = NULL;
    
    this->ptr_array_dim3_grid_reduce_threads = NULL;
    this->ptr_array_dim3_block_reduce_threads = NULL;
    
    this->ptr_array_dim3_grid_reduce_threads_DP = NULL;
    this->ptr_array_dim3_block_reduce_threads_DP = NULL;
    
    this->ptr_array_neuron_units_dim3_grid_reduce_summation = NULL;
    this->ptr_array_neuron_units_dim3_block_reduce_summation = NULL;
    
    this->ptr_array_neuron_units_dim3_grid_reduce_error = NULL;
    this->ptr_array_neuron_units_dim3_block_reduce_error = NULL;

    this->ptr_array_neuron_units_dim3_grid_reduce_batch = NULL;
    this->ptr_array_neuron_units_dim3_block_reduce_batch = NULL;

    this->ptr_array_2D_neurons_dim3_grid_reduce_norms = NULL;
    this->ptr_array_2D_neurons_dim3_block_reduce_norms = NULL;
    // |END| Struct dim3 variable. |END|

    this->total_reduce_batch_size = 0u;
    this->total_reduce_batch_DP_size = 0u;

    this->ptr_array_reduce_number_loss = nullptr;
    this->ptr_array_reduce_loss_values = nullptr;
    this->ptr_array_reduce_bit_fail_values = nullptr;
    this->ptr_array_reduce_accuracy_values[0] = nullptr;
    this->ptr_array_reduce_accuracy_values[1] = nullptr;
    this->ptr_array_reduce_accuracy_values[2] = nullptr;
    this->ptr_array_reduce_accuracy_values[3] = nullptr;
    this->ptr_array_reduce_accuracy_values[4] = nullptr;
#else
    kernel__cuModel__Allocate_Structure <<< 1, 1 >>> (n_layers,
                                                                                                  maximum_allowable_memory_bytes_received,
                                                                                                  this);
        
    CUDA__Check_Error();

    if(this->Initialize_CUDA_Device() == false) {
      ERR(L"An error has been triggered from the "
          L"\"Initialize_CUDA_Device()\" function.", );

        return false;
    }
#endif
        
    return true;
}

__device__ bool cuModel::Allocate_Reduce_Threads(void)
{
    if(this->ptr_array_dim3_grid_reduce_threads == nullptr || this->ptr_array_dim3_grid_reduce_threads_DP == nullptr)
    {
        if(this->Allocate_Reduce_Threads_Dim() == false)
        {
            ERR(L"From \"Allocate_Reduce_Threads_Dim\"",);

            return false;
        }
        else if(this->Allocate_Reduce_Threads_Dim_DP() == false)
        {
            ERR(L"From \"Allocate_Reduce_Threads_Dim_DP\"",);

            return false;
        }
    }

    return true;
}

__device__ bool cuModel::Allocate_Reduce_Threads_Dim(void)
{
    if(this->ptr_array_dim3_grid_reduce_threads == nullptr)
    {
        size_t tmp_total_elements_to_reduce,
                          tmp_index_dim3(0u);
        
        class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;

        // Compute dimension reduce data batch.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = this->number_threads;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                              0,
                                                                                              tmp_dim3_grid,
                                                                                              tmp_dim3_block);
        
        // Get remaining elements to reduce and store it.
        this->total_reduce_batch_size = tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            ERR(L"No elements to reduce.",);

            return false;
        }
        // |END| Compute dimension reduce data batch. |END|
        
        // Allocating neurons reduce summation dim3 grid.
        struct dim3 *tmp_ptr_array_dim3_grid_reduce_threads(static_cast<struct dim3*>(malloc(tmp_total_elements_to_reduce * sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_grid_reduce_threads == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_dim3_grid_reduce_threads,
                    0,
                    tmp_total_elements_to_reduce * sizeof(struct dim3));
        this->ptr_array_dim3_grid_reduce_threads = tmp_ptr_array_dim3_grid_reduce_threads;
        // |END| Allocating neurons reduce summation dim3 grid. |END|
            
        // Allocating neurons reduce summation dim3 block.
        struct dim3 *tmp_ptr_array_dim3_block_reduce_threads(static_cast<struct dim3*>(malloc(tmp_total_elements_to_reduce * sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_block_reduce_threads == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_dim3_block_reduce_threads,
                    0,
                    tmp_total_elements_to_reduce * sizeof(struct dim3));
        this->ptr_array_dim3_block_reduce_threads = tmp_ptr_array_dim3_block_reduce_threads;
        // |END| Allocating neurons reduce summation dim3 block. |END|
        
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = this->number_threads;

        // Loop to reduce "number of elements" to one at the end.
        do
        {
            // Compute remaining results to reduce.
            tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                    0,
                                                                                                    tmp_ptr_array_dim3_grid_reduce_threads[tmp_index_dim3],
                                                                                                    tmp_ptr_array_dim3_block_reduce_threads[tmp_index_dim3]);

            // Get the remaining results to reduce.
            tmp_total_elements_to_reduce = tmp_ptr_array_dim3_grid_reduce_threads[tmp_index_dim3].x;

            // Increment index to dim3.
            ++tmp_index_dim3;
        } while(tmp_total_elements_to_reduce != 1u);
    }

    return true;
}

__device__ bool cuModel::Allocate_Reduce_Threads_Dim_DP(void)
{
    if(this->ptr_array_dim3_grid_reduce_threads_DP == nullptr)
    {
        size_t tmp_total_elements_to_reduce,
                          tmp_index_dim3(0u);
        
        class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                         tmp_dim3_block;

        // Compute dimension reduce data batch.
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = this->number_threads;
        
        // Dimension required to reduce the number of elements.
        tmp_ptr_CUDA_Device->Grid_Block_Reduce_Dynamic_Parallelisme(tmp_total_elements_to_reduce,
                                                                                                            tmp_ptr_CUDA_Device->Get__Maximum_Blocks_Per_Multiprocessor(),
                                                                                                            tmp_dim3_grid,
                                                                                                            tmp_dim3_block);
        
        // Get remaining elements to reduce and store it.
        this->total_reduce_batch_DP_size = tmp_total_elements_to_reduce = tmp_dim3_grid.x;

        if(tmp_total_elements_to_reduce == 0u)
        {
            ERR(L"No elements to reduce.",);

            return false;
        }
        // |END| Compute dimension reduce data batch. |END|
        
        // Allocating neurons reduce summation dim3 grid.
        struct dim3 *tmp_ptr_array_dim3_grid_threads_DP(static_cast<struct dim3*>(malloc(tmp_total_elements_to_reduce * sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_grid_threads_DP == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_dim3_grid_threads_DP,
                    0,
                    tmp_total_elements_to_reduce * sizeof(struct dim3));
        this->ptr_array_dim3_grid_reduce_threads_DP = tmp_ptr_array_dim3_grid_threads_DP;
        // |END| Allocating neurons reduce summation dim3 grid. |END|
            
        // Allocating neurons reduce summation dim3 block.
        struct dim3 *tmp_ptr_array_dim3_block_threads_DP(static_cast<struct dim3*>(malloc(tmp_total_elements_to_reduce * sizeof(struct dim3))));
        if(tmp_ptr_array_dim3_block_threads_DP == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_dim3_block_threads_DP,
                    0,
                    tmp_total_elements_to_reduce * sizeof(struct dim3));
        this->ptr_array_dim3_block_reduce_threads_DP = tmp_ptr_array_dim3_block_threads_DP;
        // |END| Allocating neurons reduce summation dim3 block. |END|
        
        // Number elements to reduce equal total data batch.
        tmp_total_elements_to_reduce = this->number_threads;

        // Loop to reduce "number of elements" to one at the end.
        do
        {
            // Compute remaining results to reduce.
            tmp_ptr_CUDA_Device->Grid_Block_Reduce_Dynamic_Parallelisme(tmp_total_elements_to_reduce,
                                                                                                                tmp_ptr_CUDA_Device->Get__Maximum_Blocks_Per_Multiprocessor(),
                                                                                                                tmp_ptr_array_dim3_grid_threads_DP[tmp_index_dim3],
                                                                                                                tmp_ptr_array_dim3_block_threads_DP[tmp_index_dim3]);

            // Get the remaining results to reduce.
            tmp_total_elements_to_reduce = tmp_ptr_array_dim3_grid_threads_DP[tmp_index_dim3].x;

            // Increment index to dim3.
            ++tmp_index_dim3;
        } while(tmp_total_elements_to_reduce != 1u);
    }

    return true;
}

__device__ bool cuModel::Allocate_Reduce_Cost(void)
{
    if(this->ptr_array_reduce_loss_values == nullptr)
    {
        if(this->total_reduce_batch_size == 0u)
        {
            ERR(L"Can not allocate memory! Reduce size equal zero.",);

            return false;
        }

        // Allocating reduce number loss.
        size_t *tmp_ptr_array_reduce_number_loss(new size_t[this->total_reduce_batch_size]);
        if(tmp_ptr_array_reduce_number_loss == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_reduce_number_loss,
                    0,
                    this->total_reduce_batch_size * sizeof(size_t));
        this->ptr_array_reduce_number_loss = tmp_ptr_array_reduce_number_loss;
        // |END| Allocating reduce number loss. |END|
        
        // Allocating reduce bit fail values.
        size_t *tmp_ptr_array_reduce_bit_fail_values(new size_t[this->total_reduce_batch_size]);
        if(tmp_ptr_array_reduce_bit_fail_values == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_reduce_bit_fail_values,
                    0,
                    this->total_reduce_batch_size * sizeof(size_t));
        this->ptr_array_reduce_bit_fail_values = tmp_ptr_array_reduce_bit_fail_values;
        // |END| Allocating reduce bit fail values. |END|
        
        // Allocating reduce loss values.
        var *tmp_ptr_array_reduce_loss_values(new var[this->total_reduce_batch_size]);
        if(tmp_ptr_array_reduce_loss_values == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_reduce_loss_values,
                    0,
                    this->total_reduce_batch_size * sizeof(var));
        this->ptr_array_reduce_loss_values = tmp_ptr_array_reduce_loss_values;
        // |END| Allocating reduce loss values.. |END|
        
        // Allocating reduce accuracy values.
        if((this->ptr_array_reduce_accuracy_values[0] = new var[this->total_reduce_batch_size]) == nullptr)
        {
            ERR(L"Can not allocate %zu bytes. At line %d.",
                                     this->total_reduce_batch_size * sizeof(var),
                                     __LINE__);

            return false;
        }
        else
        {
            memset(this->ptr_array_reduce_accuracy_values[0],
                        0,
                        this->total_reduce_batch_size * sizeof(var));
        }
        
        if((this->ptr_array_reduce_accuracy_values[1] = new var[this->total_reduce_batch_size]) == nullptr)
        {
            ERR(L"Can not allocate %zu bytes. At line %d.",
                                     this->total_reduce_batch_size * sizeof(var),
                                     __LINE__);

            return false;
        }
        else
        {
            memset(this->ptr_array_reduce_accuracy_values[1],
                        0,
                        this->total_reduce_batch_size * sizeof(var));
        }
        
        if((this->ptr_array_reduce_accuracy_values[2] = new var[this->total_reduce_batch_size]) == nullptr)
        {
            ERR(L"Can not allocate %zu bytes. At line %d.",
                                     this->total_reduce_batch_size * sizeof(var),
                                     __LINE__);

            return false;
        }
        else
        {
            memset(this->ptr_array_reduce_accuracy_values[2],
                        0,
                        this->total_reduce_batch_size * sizeof(var));
        }
        
        if((this->ptr_array_reduce_accuracy_values[3] = new var[this->total_reduce_batch_size]) == nullptr)
        {
            ERR(L"Can not allocate %zu bytes. At line %d.",
                                     this->total_reduce_batch_size * sizeof(var),
                                     __LINE__);

            return false;
        }
        else
        {
            memset(this->ptr_array_reduce_accuracy_values[3],
                        0,
                        this->total_reduce_batch_size * sizeof(var));
        }
        
        if((this->ptr_array_reduce_accuracy_values[4] = new var[this->total_reduce_batch_size]) == nullptr)
        {
            ERR(L"Can not allocate %zu bytes. At line %d.",
                                     this->total_reduce_batch_size * sizeof(var),
                                     __LINE__);

            return false;
        }
        else
        {
            memset(this->ptr_array_reduce_accuracy_values[4],
                        0,
                        this->total_reduce_batch_size * sizeof(var));
        }
        // |END| Allocating reduce accuracy values.. |END|
    }

    return true;
}

__device__ bool cuModel::Allocate__Neuron_Units(void)
{
    size_t tmp_number_neuron_units,
              i;
    
    if(this->total_neuron_units != 0_UZ)
    {
        struct cuLayer const *const last_layer(this->ptr_last_layer);
        struct cuLayer *layer_it(this->ptr_array_layers);

        struct cuNeuron *tmp_ptr_array_neuron_units(new struct cuNeuron[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }

        // Allocating neurons first index.
        size_t *tmp_ptr_array_neuron_units_first_connection_index(new size_t[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_first_connection_index == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<size_t>(this->total_neuron_units,
                                          tmp_ptr_array_neuron_units_first_connection_index,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons first index. |END|
            
        // Allocating neurons last index.
        size_t *tmp_ptr_array_neuron_units_last_connection_index(new size_t[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_last_connection_index == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<size_t>(this->total_neuron_units,
                                          tmp_ptr_array_neuron_units_last_connection_index,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons last index. |END|
            
        // Allocating neurons number connections.
        size_t *tmp_ptr_array_neuron_units_number_connections(new size_t[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_number_connections == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<size_t>(this->total_neuron_units,
                                          tmp_ptr_array_neuron_units_number_connections,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons number connections. |END|
        
        // Allocating neuron unit(s) summation(s).
        var *tmp_ptr_array_neuron_units_summations(new var[this->batch_size * this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_summations == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->batch_size * this->total_neuron_units,
                            tmp_ptr_array_neuron_units_summations,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neuron unit(s) summation(s). |END|
            
        // Allocating neuron unit(s) value(s).
        var *tmp_ptr_array_neuron_units_values(new var[this->batch_size * this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_values == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->batch_size * this->total_neuron_units,
                            tmp_ptr_array_neuron_units_values,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neuron unit(s) value(s). |END|
        
        // Allocating neuron unit(s) error(s).
        var *tmp_ptr_array_neuron_units_errors(new var[this->batch_size * this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_errors == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->batch_size * this->total_neuron_units,
                            tmp_ptr_array_neuron_units_errors,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neuron unit(s) error(s). |END|
        
        // Allocating neurons activation function.
        enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS *tmp_ptr_array_neuron_units_type_activation_function(new enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS[this->total_neuron_units]);
        if(tmp_ptr_array_neuron_units_type_activation_function == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS>(this->total_neuron_units,
                                                                                                                              tmp_ptr_array_neuron_units_type_activation_function,
                                                                                                                              this->ptr_array_dim3_grid + 3,
                                                                                                                              this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons activation function. |END|
        
        // Allocating neurons grid connections.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_connections(static_cast<struct dim3*>(malloc(this->total_neuron_units * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_grid_connections == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_dim3_grid_connections,
                    0,
                    this->total_neuron_units * sizeof(struct dim3));
        // |END| Allocating neurons grid connections. |END|
            
        // Allocating neurons block connections.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_connections(static_cast<struct dim3*>(malloc(this->total_neuron_units * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_block_connections == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_dim3_block_connections,
                    0,
                    this->total_neuron_units * sizeof(struct dim3));
        // |END| Allocating neurons block connections. |END|

        // Assign neurons variable.
        this->ptr_array_neuron_units_first_forward_connection_index = tmp_ptr_array_neuron_units_first_connection_index;
        this->ptr_array_neuron_units_last_forward_connection_index = tmp_ptr_array_neuron_units_last_connection_index;
        this->ptr_array_neuron_units_number_forward_connections = tmp_ptr_array_neuron_units_number_connections;

        this->ptr_array_neuron_units_summations = tmp_ptr_array_neuron_units_summations;
        this->ptr_array_neuron_units_values = tmp_ptr_array_neuron_units_values;
        this->ptr_array_neuron_units_errors = tmp_ptr_array_neuron_units_errors;

        this->ptr_array_neuron_units_type_activation_function = tmp_ptr_array_neuron_units_type_activation_function;

        this->ptr_array_neuron_units_dim3_grid_connections = tmp_ptr_array_neuron_units_dim3_grid_connections;
        this->ptr_array_neuron_units_dim3_block_connections = tmp_ptr_array_neuron_units_dim3_block_connections;
        // |END| Assign neurons variable. |END|
        
        for(; layer_it != last_layer; ++layer_it)
        {
            layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;

            if((tmp_number_neuron_units = *layer_it->ptr_number_neurons) != 0u)
            {
                // Assign neurons variable.
                for(i = 0u; i != tmp_number_neuron_units; ++i)
                {
                    tmp_ptr_array_neuron_units[i].ptr_first_forward_connection_index = tmp_ptr_array_neuron_units_first_connection_index++;
                    tmp_ptr_array_neuron_units[i].ptr_last_forward_connection_index = tmp_ptr_array_neuron_units_last_connection_index++;
                    tmp_ptr_array_neuron_units[i].ptr_number_forward_connections = tmp_ptr_array_neuron_units_number_connections++;

                    tmp_ptr_array_neuron_units[i].ptr_array_summations = tmp_ptr_array_neuron_units_summations++;
                    tmp_ptr_array_neuron_units[i].ptr_array_values = tmp_ptr_array_neuron_units_values++;
                    tmp_ptr_array_neuron_units[i].ptr_array_errors = tmp_ptr_array_neuron_units_errors++;
                    
                    tmp_ptr_array_neuron_units[i].ptr_type_activation_function = tmp_ptr_array_neuron_units_type_activation_function++;

                    tmp_ptr_array_neuron_units[i].ptr_dim3_grid_connections = tmp_ptr_array_neuron_units_dim3_grid_connections++;
                    tmp_ptr_array_neuron_units[i].ptr_dim3_block_connections = tmp_ptr_array_neuron_units_dim3_block_connections++;
                }
                // |END| Assign neurons variable. |END|
                
                tmp_ptr_array_neuron_units_summations += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_values += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_errors += (this->batch_size - 1u) * tmp_number_neuron_units;

                tmp_ptr_array_neuron_units += tmp_number_neuron_units;
            }

            layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
        }

        this->total_neuron_units_allocated = this->total_neuron_units;
    }

    return true;
}

__device__ bool cuModel::Allocate__Neurons_Reduce_Summation(void)
{
    size_t tmp_neurons_reduce_summation_size_so_far,
                      tmp_total_elements_to_reduce,
                      tmp_layer_reduce_summation_size,
                      tmp_number_neurons_in_layer,
                      tmp_index_dim3;
    
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_neuron_units_reduce_summation_size == nullptr)
    {
        // ONLY FOR DENSE LAYER.
        // TODO: Make shortcut layer compatible.
        struct cuLayer const *const last_layer(this->ptr_last_layer);
        struct cuLayer *layer_it;
        
        struct cuNeuron const *tmp_ptr_last_neuron_unit;
        struct cuNeuron *tmp_ptr_neuron_unit_it;
        
        class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                            tmp_dim3_block;

        // COMPUTE REDUCE SUMMATION SIZE.
        // Allocating neurons reduce summation size.
        size_t *tmp_ptr_array_neuron_units_reduce_summation_size(new size_t[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_reduce_summation_size == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<size_t>(this->total_neuron_units_allocated,
                                          tmp_ptr_array_neuron_units_reduce_summation_size,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons reduce summation size. |END|
        
        // Compute dimension reduce summation.
        this->ptr_array_neuron_units_reduce_summation_size = tmp_ptr_array_neuron_units_reduce_summation_size;
        
        for(tmp_neurons_reduce_summation_size_so_far = 0,
            tmp_ptr_neuron_unit_it = this->ptr_array_layers->ptr_array_neuron_units,
            tmp_ptr_last_neuron_unit = tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                                      ++tmp_ptr_array_neuron_units_reduce_summation_size)
        {
            // Number elements to reduce equal number of connections from the neuron.
            tmp_total_elements_to_reduce = *tmp_ptr_neuron_unit_it->ptr_number_forward_connections;
            
            // If is not the bias. (The bias have no elements to reduce.)
            if(tmp_total_elements_to_reduce != 0u)
            {
                // Dimension required to reduce the number of elements.
                tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                      0,
                                                                                                      tmp_dim3_grid,
                                                                                                      tmp_dim3_block);
            
                // Get remaining elements to reduce.
                tmp_total_elements_to_reduce = tmp_dim3_grid.x;
            }

            // Maximum remaining elements to reduce.
            *tmp_ptr_array_neuron_units_reduce_summation_size = tmp_total_elements_to_reduce;

            // Assign a pointer to the maximum reduce summation size of that neuron.
            tmp_ptr_neuron_unit_it->ptr_reduce_summation_size = tmp_ptr_array_neuron_units_reduce_summation_size;

            // Summation of the total maximum number of summation result.
            tmp_neurons_reduce_summation_size_so_far += tmp_total_elements_to_reduce;
        }

        this->neurons_total_reduce_summation_size = tmp_neurons_reduce_summation_size_so_far;

        if(tmp_neurons_reduce_summation_size_so_far == 0u)
        {
            ERR(L"No elements to reduce.",);

            return false;
        }
        // |END| Compute dimension reduce summation. |END|
        // |END| COMPUTE REDUCE SUMMATION SIZE. |END|
        
        // COMPUTE DIMENSION REDUCE SUMMATION.
        // Allocating neurons reduce summation.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        var **tmp_ptr_array_2D_neurons_position_reduce_summation_array(new var*[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_2D_neurons_position_reduce_summation_array == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var*>(this->total_neuron_units_allocated,
                             tmp_ptr_array_2D_neurons_position_reduce_summation_array,
                             this->ptr_array_dim3_grid + 3,
                             this->ptr_array_dim3_block + 3);

        var *tmp_ptr_array_neuron_units_reduce_summation_results(new var[this->batch_size * tmp_neurons_reduce_summation_size_so_far]);
        if(tmp_ptr_array_neuron_units_reduce_summation_results == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_reduce_summation_results,
                    0,
                    this->batch_size * tmp_neurons_reduce_summation_size_so_far * sizeof(var));
        // |END| Allocating neurons reduce summation. |END|
        
        // Allocating neurons reduce summation dim3 grid.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_summation(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_summation_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_grid_summation == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_dim3_grid_summation,
                    0,
                    tmp_neurons_reduce_summation_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce summation dim3 grid. |END|
            
        // Allocating neurons reduce summation dim3 block.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_summation(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_summation_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_block_summation == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_dim3_block_summation,
                    0,
                    tmp_neurons_reduce_summation_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce summation dim3 block. |END|
            
        // Assign global array.
        this->ptr_array_2D_neurons_reduce_summation = tmp_ptr_array_2D_neurons_position_reduce_summation_array;
        this->ptr_array_neuron_units_dim3_grid_reduce_summation = tmp_ptr_array_neuron_units_dim3_grid_summation;
        this->ptr_array_neuron_units_dim3_block_reduce_summation = tmp_ptr_array_neuron_units_dim3_block_summation;
        // |END| Assign global array. |END|
        
        // Loop through each layers.
        for(layer_it = this->ptr_array_layers; layer_it != last_layer; ++layer_it)
        {
            // Get neurons array from that layer.
            tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units;

            // Get the reduce summation size of each neurons in that layer.
            tmp_layer_reduce_summation_size = *tmp_ptr_neuron_unit_it->ptr_reduce_summation_size;
            
            // Get the number of neurons in layer.
            tmp_number_neurons_in_layer = *layer_it->ptr_number_neurons;
            
            // Loop through each neurons in the layer.
            for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_summation_array)
            {
                // Result.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_reduce_summation_array = tmp_ptr_array_neuron_units_reduce_summation_results;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_reduce_summation = tmp_ptr_array_2D_neurons_position_reduce_summation_array;
                // |END| Result. |END|
                
                // Number elements to reduce equal number of connections from the neuron.
                tmp_total_elements_to_reduce = *tmp_ptr_neuron_unit_it->ptr_number_forward_connections;

                // If is not the bias. (The bias have no elements to reduce.)
                if(tmp_total_elements_to_reduce != 0u)
                {
                    // Assign dim3 grid to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation = tmp_ptr_array_neuron_units_dim3_grid_summation++;
                    // Assign dim3 block to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation = tmp_ptr_array_neuron_units_dim3_block_summation++;

                    // Initialize index to zero.
                    tmp_index_dim3 = 0u;

                    // Loop to reduce "number of elements" to one at the end.
                    do
                    {
                        // Compute remaining results to reduce.
                        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                                0,
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)],
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_summation[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)]);

                        // Get the remaining results to reduce.
                        tmp_total_elements_to_reduce = tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_summation[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)].x;

                        // Increment index to dim3.
                        ++tmp_index_dim3;
                    } while(tmp_total_elements_to_reduce != 1u);
                    // |END| dim3. |END|

                    // Increment the begining results by the layer reduce summation size.
                    tmp_ptr_array_neuron_units_reduce_summation_results += tmp_layer_reduce_summation_size;
                }
            }
                
            // If some elements need to be reduce in the layer.
            if(tmp_layer_reduce_summation_size != 0u)
            {
                // Increment pointer by (number of neurons in layer minus bias) times (layer reduce summation size minus one).
                tmp_ptr_array_neuron_units_dim3_grid_summation += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_summation_size - 1u);
                tmp_ptr_array_neuron_units_dim3_block_summation += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_summation_size - 1u);
            }
        }
        // |END| COMPUTE DIMENSION REDUCE SUMMATION. |END|
    }

    return true;
}

__device__ bool cuModel::Allocate__Neurons_Reduce_Error(void)
{
    size_t tmp_neurons_reduce_error_size_so_far,
                      tmp_total_elements_to_reduce_layer,
                      tmp_total_elements_to_reduce,
                      tmp_layer_reduce_error_size,
                      tmp_number_neurons_in_layer,
                      tmp_index_dim3;
    
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_neuron_units_reduce_error_size == nullptr)
    {
        // ONLY FOR DENSE LAYER.
        // TODO: Make shortcut layer compatible.
        struct cuLayer const *const last_layer(this->ptr_last_layer);
        struct cuLayer *tmp_ptr_next_layer,
                                               *layer_it;
        
        struct cuNeuron const *tmp_ptr_last_neuron_unit;
        struct cuNeuron *tmp_ptr_neuron_unit_it;
        
        class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                            tmp_dim3_block;

        // COMPUTE REDUCE ERROR SIZE.
        // Allocating neurons reduce error size.
        size_t *tmp_ptr_array_neuron_units_reduce_error_size(new size_t[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_reduce_error_size == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<size_t>(this->total_neuron_units_allocated,
                                          tmp_ptr_array_neuron_units_reduce_error_size,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons reduce error size. |END|
        
        // Compute dimension reduce error.
        this->ptr_array_neuron_units_reduce_error_size = tmp_ptr_array_neuron_units_reduce_error_size;
        
        // Loop through each layers.
        for(tmp_neurons_reduce_error_size_so_far = 0,
            layer_it = this->ptr_array_layers,
            tmp_ptr_next_layer = layer_it + 1; layer_it != last_layer; ++layer_it,
                                                                                                                                   ++tmp_ptr_next_layer)
        {
            if(layer_it == this->ptr_array_layers // Input layer.
                ||
                layer_it == this->ptr_last_layer - 1) // Output layer.
            { tmp_total_elements_to_reduce_layer = 0u; }
            else
            // Number elements to reduce equal number of connections to the neuron.
            { tmp_total_elements_to_reduce_layer = *tmp_ptr_next_layer->ptr_number_neurons - 1u; } // Subtract bias.
            
            for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit,
                tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                   ++tmp_ptr_array_neuron_units_reduce_error_size)
            {
                // If no elements to reduce or the neuron is a bias.
                if(tmp_total_elements_to_reduce_layer == 0u || tmp_ptr_neuron_unit_it == tmp_ptr_last_neuron_unit - 1)
                { tmp_total_elements_to_reduce = 0u; }
                else
                {
                    // Number elements to reduce equal number of connections to the neuron.
                    tmp_total_elements_to_reduce = tmp_total_elements_to_reduce_layer;

                    // Dimension required to reduce the number of elements.
                    tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                          0,
                                                                                                          tmp_dim3_grid,
                                                                                                          tmp_dim3_block);
                
                    // Get remaining elements to reduce.
                    tmp_total_elements_to_reduce = tmp_dim3_grid.x;
                }
                
                // Maximum remaining elements to reduce.
                *tmp_ptr_array_neuron_units_reduce_error_size = tmp_total_elements_to_reduce;

                // Assign a pointer to the maximum reduce error size of that neuron.
                tmp_ptr_neuron_unit_it->ptr_reduce_error_size = tmp_ptr_array_neuron_units_reduce_error_size;

                // Summation of the total maximum number of error result.
                tmp_neurons_reduce_error_size_so_far += tmp_total_elements_to_reduce;
            }
        }

        this->neurons_total_reduce_error_size = tmp_neurons_reduce_error_size_so_far;

        if(tmp_neurons_reduce_error_size_so_far == 0u)
        {
            ERR(L"No elements to reduce.",);

            return false;
        }
        // |END| Compute dimension reduce error. |END|
        // |END| COMPUTE REDUCE ERROR SIZE. |END|
        
        // COMPUTE DIMENSION REDUCE ERROR.
        // Allocating neurons reduce error.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        var **tmp_ptr_array_2D_neurons_position_reduce_error_array(new var*[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_2D_neurons_position_reduce_error_array == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var*>(this->total_neuron_units_allocated,
                              tmp_ptr_array_2D_neurons_position_reduce_error_array,
                              this->ptr_array_dim3_grid + 3,
                              this->ptr_array_dim3_block + 3);

        var *tmp_ptr_array_neuron_units_reduce_error_results(new var[this->batch_size * tmp_neurons_reduce_error_size_so_far]);
        if(tmp_ptr_array_neuron_units_reduce_error_results == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_reduce_error_results,
                    0,
                    this->batch_size * tmp_neurons_reduce_error_size_so_far * sizeof(var));
        // |END| Allocating neurons reduce error. |END|
        
        // Allocating neurons reduce error dim3 grid.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_error(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_error_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_grid_error == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_dim3_grid_error,
                    0,
                    tmp_neurons_reduce_error_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce error dim3 grid. |END|
            
        // Allocating neurons reduce error dim3 block.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_error(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_error_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_block_error == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_dim3_block_error,
                    0,
                    tmp_neurons_reduce_error_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce error dim3 block. |END|
            
        // Assign global array.
        this->ptr_array_2D_neurons_reduce_error = tmp_ptr_array_2D_neurons_position_reduce_error_array;
        this->ptr_array_neuron_units_dim3_grid_reduce_error = tmp_ptr_array_neuron_units_dim3_grid_error;
        this->ptr_array_neuron_units_dim3_block_reduce_error = tmp_ptr_array_neuron_units_dim3_block_error;
        // |END| Assign global array. |END|
        
        // Loop through each layers.
        for(layer_it = this->ptr_array_layers,
            tmp_ptr_next_layer = layer_it + 1; layer_it != last_layer; ++layer_it,
                                                                                                                                   ++tmp_ptr_next_layer)
        {
            // Get neurons array from that layer.
            tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units;

            // Get the reduce error size of each neurons in that layer.
            tmp_layer_reduce_error_size = *tmp_ptr_neuron_unit_it->ptr_reduce_error_size;

            // Get the number of neurons in layer.
            tmp_number_neurons_in_layer = *layer_it->ptr_number_neurons;

            if(layer_it == this->ptr_array_layers // Input layer.
                ||
                layer_it == this->ptr_last_layer - 1) // Output layer.
            { tmp_total_elements_to_reduce_layer = 0u; }
            else
            // Number elements to reduce equal number of connections to the neuron.
            { tmp_total_elements_to_reduce_layer = *tmp_ptr_next_layer->ptr_number_neurons - 1u; } // Subtract bias.
            
            // Loop through each neurons in the layer.
            for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_error_array)
            {
                // Result.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_reduce_error_array = tmp_ptr_array_neuron_units_reduce_error_results;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_reduce_error = tmp_ptr_array_2D_neurons_position_reduce_error_array;
                // |END| Result. |END|
                
                // If we have elements to reduce and the neuron is not a bias.
                if(tmp_total_elements_to_reduce_layer != 0u && tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit - 1)
                {
                    // Number elements to reduce equal number of connections to the neuron.
                    tmp_total_elements_to_reduce = tmp_total_elements_to_reduce_layer;

                    // Assign dim3 grid to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error = tmp_ptr_array_neuron_units_dim3_grid_error++;
                    // Assign dim3 block to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error = tmp_ptr_array_neuron_units_dim3_block_error++;

                    // Initialize index to zero.
                    tmp_index_dim3 = 0u;

                    // Loop to reduce "number of elements" to one at the end.
                    do
                    {
                        // Compute remaining results to reduce.
                        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                                0,
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)],
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_error[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)]);

                        // Get the remaining results to reduce.
                        tmp_total_elements_to_reduce = tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_error[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)].x;

                        // Increment index to dim3.
                        ++tmp_index_dim3;
                    } while(tmp_total_elements_to_reduce != 1u);
                    // |END| dim3. |END|

                    // Increment the begining results by the layer reduce error size.
                    tmp_ptr_array_neuron_units_reduce_error_results += tmp_layer_reduce_error_size;
                }
            }
                
            // If some elements need to be reduce in the layer.
            if(tmp_layer_reduce_error_size != 0u)
            {
                // Increment pointer by (number of neurons in layer minus bias) times (layer reduce error size minus one).
                tmp_ptr_array_neuron_units_dim3_grid_error += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_error_size - 1u);
                tmp_ptr_array_neuron_units_dim3_block_error += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_error_size - 1u);
            }
        }
        // |END| COMPUTE DIMENSION REDUCE ERROR. |END|
    }

    return true;
}

__device__ bool cuModel::Allocate__Neurons_Reduce_Batch_Normalization(void)
{
    size_t tmp_neurons_reduce_batch_size_so_far,
                      tmp_total_elements_to_reduce,
                      tmp_layer_reduce_batch_size,
                      tmp_number_neurons_in_layer,
                      tmp_index_dim3;
    
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_neuron_units_reduce_batch_size == nullptr)
    {
        // ONLY FOR DENSE LAYER.
        // TODO: Make shortcut layer compatible.
        struct cuLayer const *const last_layer(this->ptr_last_layer);
        struct cuLayer *layer_it;
        
        struct cuNeuron const *tmp_ptr_last_neuron_unit;
        struct cuNeuron *tmp_ptr_neuron_unit_it;
        
        class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                            tmp_dim3_block;

        // COMPUTE REDUCE BATCH SIZE.
        // Allocating neurons reduce batch size.
        size_t *tmp_ptr_array_neuron_units_reduce_batch_size(new size_t[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_reduce_batch_size == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<size_t>(this->total_neuron_units_allocated,
                                          tmp_ptr_array_neuron_units_reduce_batch_size,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        this->ptr_array_neuron_units_reduce_batch_size = tmp_ptr_array_neuron_units_reduce_batch_size;
        // |END| Allocating neurons reduce batch size. |END|
        
        // Compute dimension reduce batch.
        for(tmp_neurons_reduce_batch_size_so_far = 0,
            tmp_ptr_neuron_unit_it = this->ptr_array_layers->ptr_array_neuron_units,
            tmp_ptr_last_neuron_unit = tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                                       ++tmp_ptr_array_neuron_units_reduce_batch_size)
        {
            // Number elements to reduce equal the size of batch.
            tmp_total_elements_to_reduce = this->batch_size;

            // If the neuron is a bias. Number of elements to reduce equal zero.
            if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections == 0u)
            { tmp_total_elements_to_reduce = 0u; }
            
            // If is not the bias. (The bias have no elements to reduce.)
            if(tmp_total_elements_to_reduce != 0u)
            {
                // Dimension required to reduce the number of elements.
                tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                      0,
                                                                                                      tmp_dim3_grid,
                                                                                                      tmp_dim3_block);
            
                // Get remaining elements to reduce.
                tmp_total_elements_to_reduce = tmp_dim3_grid.x;
            }

            // Maximum remaining elements to reduce.
            *tmp_ptr_array_neuron_units_reduce_batch_size = tmp_total_elements_to_reduce;

            // Assign a pointer to the maximum reduce norm size of that neuron.
            tmp_ptr_neuron_unit_it->ptr_reduce_batch_size = tmp_ptr_array_neuron_units_reduce_batch_size;

            // Summation of the total maximum number of batch result.
            tmp_neurons_reduce_batch_size_so_far += tmp_total_elements_to_reduce;
        }

        if(tmp_neurons_reduce_batch_size_so_far == 0u)
        {
            ERR(L"No elements to reduce.",);

            return false;
        }
        // |END| Compute dimension reduce batch. |END|
        // |END| COMPUTE REDUCE BATCH SIZE. |END|
        
        // COMPUTE DIMENSION REDUCE BATCH.
        // Allocating neurons reduce batch mean.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        var **tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array(new var*[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var*>(this->total_neuron_units_allocated,
                              tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array,
                              this->ptr_array_dim3_grid + 3,
                              this->ptr_array_dim3_block + 3);
        this->ptr_array_2D_neurons_reduce_batch_mean = tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array;

        var *tmp_ptr_array_neuron_units_reduce_batch_mean_results(new var[tmp_neurons_reduce_batch_size_so_far]);
        if(tmp_ptr_array_neuron_units_reduce_batch_mean_results == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_reduce_batch_mean_results,
                    0,
                    tmp_neurons_reduce_batch_size_so_far * sizeof(var));
        // |END| Allocating neurons reduce batch mean. |END|
        
        // Allocating neurons reduce batch variance.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        var **tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array(new var*[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var*>(this->total_neuron_units_allocated,
                              tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array,
                              this->ptr_array_dim3_grid + 3,
                              this->ptr_array_dim3_block + 3);
        this->ptr_array_2D_neurons_reduce_batch_variance = tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array;

        var *tmp_ptr_array_neuron_units_reduce_batch_variance_results(new var[tmp_neurons_reduce_batch_size_so_far]);
        if(tmp_ptr_array_neuron_units_reduce_batch_variance_results == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_reduce_batch_variance_results,
                    0,
                    tmp_neurons_reduce_batch_size_so_far * sizeof(var));
        // |END| Allocating neurons reduce batch variance. |END|
        
        // Allocating neurons reduce batch dim3 grid.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_batch(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_grid_batch == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_dim3_grid_batch,
                    0,
                    tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3));
        this->ptr_array_neuron_units_dim3_grid_reduce_batch = tmp_ptr_array_neuron_units_dim3_grid_batch;
        // |END| Allocating neurons reduce batch dim3 grid. |END|
            
        // Allocating neurons reduce batch dim3 block.
        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_batch(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_block_batch == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_dim3_block_batch,
                    0,
                    tmp_neurons_reduce_batch_size_so_far * sizeof(struct dim3));
        this->ptr_array_neuron_units_dim3_block_reduce_batch = tmp_ptr_array_neuron_units_dim3_block_batch;
        // |END| Allocating neurons reduce batch dim3 block. |END|
        
        // Loop through each layers.
        for(layer_it = this->ptr_array_layers; layer_it != last_layer; ++layer_it)
        {
            // Get neurons array from that layer.
            tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units;

            // Get the reduce batch size of each neurons in that layer.
            tmp_layer_reduce_batch_size = *tmp_ptr_neuron_unit_it->ptr_reduce_batch_size;
            
            // Get the number of neurons in layer.
            tmp_number_neurons_in_layer = *layer_it->ptr_number_neurons;
            
            // Loop through each neurons in the layer.
            for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array,
                                                                                                                                                                   ++tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array)
            {
                // Result.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array = tmp_ptr_array_neuron_units_reduce_batch_mean_results;
                *tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array = tmp_ptr_array_neuron_units_reduce_batch_variance_results;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_reduce_mean = tmp_ptr_array_2D_neurons_position_reduce_batch_mean_array;
                tmp_ptr_neuron_unit_it->ptr_array_reduce_variance = tmp_ptr_array_2D_neurons_position_reduce_batch_variance_array;
                // |END| Result. |END|
                
                // Number elements to reduce equal the size of batch
                tmp_total_elements_to_reduce = this->batch_size;
                
                // If the neuron is a bias. Number of elements to reduce equal zero.
                if(*tmp_ptr_neuron_unit_it->ptr_number_forward_connections == 0u)
                { tmp_total_elements_to_reduce = 0u; }
                
                // If is not the bias. (The bias have no elements to reduce.)
                if(tmp_total_elements_to_reduce != 0u)
                {
                    // Assign dim3 grid to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads = tmp_ptr_array_neuron_units_dim3_grid_batch++;
                    // Assign dim3 block to the pointer location.
                    tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_threads = tmp_ptr_array_neuron_units_dim3_block_batch++;

                    // Initialize index to zero.
                    tmp_index_dim3 = 0u;

                    // Loop to reduce "number of elements" to one at the end.
                    do
                    {
                        // Compute remaining results to reduce.
                        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                                0,
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)],
                                                                                                                tmp_ptr_neuron_unit_it->ptr_array_dim3_block_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)]);

                        // Get the remaining results to reduce.
                        tmp_total_elements_to_reduce = tmp_ptr_neuron_unit_it->ptr_array_dim3_grid_reduce_threads[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)].x;

                        // Increment index to dim3.
                        ++tmp_index_dim3;
                    } while(tmp_total_elements_to_reduce != 1u);
                    // |END| dim3. |END|

                    // Increment the begining results by the layer reduce batch size.
                    tmp_ptr_array_neuron_units_reduce_batch_mean_results += tmp_layer_reduce_batch_size;
                    tmp_ptr_array_neuron_units_reduce_batch_variance_results += tmp_layer_reduce_batch_size;
                }
            }
                
            // If some elements need to be reduce in the layer.
            if(tmp_layer_reduce_batch_size != 0u)
            {
                // Increment pointer by (number of neurons in layer minus bias) times (layer reduce batch size minus one).
                tmp_ptr_array_neuron_units_dim3_grid_batch += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_batch_size - 1u);
                tmp_ptr_array_neuron_units_dim3_block_batch += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_batch_size - 1u);
            }
        }
        // |END| COMPUTE DIMENSION REDUCE BATCH. |END|

        this->neurons_total_reduce_batch_size = tmp_neurons_reduce_batch_size_so_far;
    }

    return true;
}

__device__ bool cuModel::Allocate__Neurons_Reduce_Norms(void)
{
    size_t tmp_neurons_reduce_norms_size_so_far,
                      tmp_total_elements_to_reduce,
                      tmp_layer_reduce_norms_size,
                      tmp_number_neurons_in_layer,
                      tmp_index_dim3;
    
    if(this->total_neuron_units_allocated != 0u && this->ptr_array_neuron_units_reduce_norms_size == nullptr)
    {
        // ONLY FOR DENSE LAYER.
        // TODO: Make shortcut layer compatible.
        struct cuLayer const *const last_layer(this->ptr_last_layer);
        struct cuLayer *layer_it;
            
        struct cuNeuron const *tmp_ptr_last_neuron_unit;
        struct cuNeuron *tmp_ptr_neuron_unit_it;
            
        class cuDeviceProp const *const tmp_ptr_CUDA_Device(this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

        struct dim3 tmp_dim3_grid,
                            tmp_dim3_block;

        // COMPUTE REDUCE NORMS SIZE.
        // Allocating neurons reduce norms size.
        size_t *tmp_ptr_array_neuron_units_reduce_norms_size(new size_t[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_reduce_norms_size == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<size_t>(this->total_neuron_units_allocated,
                                          tmp_ptr_array_neuron_units_reduce_norms_size,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons reduce norms size. |END|
        
        // Compute dimension reduce norms.
        this->ptr_array_neuron_units_reduce_norms_size = tmp_ptr_array_neuron_units_reduce_norms_size;
        
        for(tmp_neurons_reduce_norms_size_so_far = 0,
            tmp_ptr_neuron_unit_it = this->ptr_array_layers->ptr_array_neuron_units,
            tmp_ptr_last_neuron_unit = tmp_ptr_neuron_unit_it + this->total_neuron_units_allocated; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
        {
            // Number elements to reduce equal number of connections from the neuron.
            tmp_total_elements_to_reduce = *tmp_ptr_neuron_unit_it->ptr_number_forward_connections;
            
            // If is not the bias. (The bias have no elements to reduce.)
            if(tmp_total_elements_to_reduce != 0u)
            {
                // Dimension required to reduce the number of elements.
                tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                      0,
                                                                                                      tmp_dim3_grid,
                                                                                                      tmp_dim3_block);

                // Get remaining elements to reduce.
                tmp_total_elements_to_reduce = tmp_dim3_grid.x;
            }

            // Maximum remaining elements to reduce.
            *tmp_ptr_array_neuron_units_reduce_norms_size = tmp_total_elements_to_reduce;

            // Assign a pointer to the maximum reduce norm size of that neuron.
            tmp_ptr_neuron_unit_it->ptr_reduce_norms_size = tmp_ptr_array_neuron_units_reduce_norms_size++;

            // Summation of the total maximum number of norms result.
            tmp_neurons_reduce_norms_size_so_far += tmp_total_elements_to_reduce;
        }

        this->neurons_total_reduce_norms_size = tmp_neurons_reduce_norms_size_so_far;

        if(tmp_neurons_reduce_norms_size_so_far == 0u)
        {
            ERR(L"No elements to reduce.",);

            return false;
        }
        // |END| Compute dimension reduce norms. |END|
        // |END| COMPUTE REDUCE NORMS SIZE. |END|
        
        // COMPUTE DIMENSION REDUCE NORMS.
        // Allocating neuroyed number neurons in layer.
        // "load" and "plus" technique is equivalent to the 2D array technique because both need to be at the size of "total_neuron_units_allocated"
        // in term of storage. But "load" and "plus" technique use the arithmetic power of coalescing threads in a warp.
        size_t *tmp_ptr_array_neuroyed_number_neurons_in_layer(new size_t[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuroyed_number_neurons_in_layer == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<size_t>(this->total_neuron_units_allocated,
                                          tmp_ptr_array_neuroyed_number_neurons_in_layer,
                                          this->ptr_array_dim3_grid + 3,
                                          this->ptr_array_dim3_block + 3);
        // |END| Allocating neuroyed number neurons in layer. |END|
        
        // Allocating neurons reduce norms.
        // 2D array position technique is equivalent to the "load" and "plus" technique because both
        // need to be at the size of "total_neuron_units_allocated" in term of storage. But 2D array don't need to use arithmetic.
        var **tmp_ptr_array_2D_neurons_position_reduce_norms_array(new var*[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_2D_neurons_position_reduce_norms_array == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var*>(this->total_neuron_units_allocated,
                              tmp_ptr_array_2D_neurons_position_reduce_norms_array,
                              this->ptr_array_dim3_grid + 3,
                              this->ptr_array_dim3_block + 3);

        var *tmp_ptr_array_neuron_units_reduce_norms_results(new var[tmp_neurons_reduce_norms_size_so_far]);
        if(tmp_ptr_array_neuron_units_reduce_norms_results == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_reduce_norms_results,
                    0,
                    tmp_neurons_reduce_norms_size_so_far * sizeof(var));
        // |END| Allocating neurons reduce norms. |END|
        
        // Allocating neurons reduce norms dim3 grid.
        struct dim3 **tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms(static_cast<struct dim3**>(malloc(this->total_neuron_units_allocated * sizeof(struct dim3*))));
        if(tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms,
                        0,
                        this->total_neuron_units_allocated * sizeof(struct dim3*));

        struct dim3 *tmp_ptr_array_neuron_units_dim3_grid_reduce_norms(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_norms_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_grid_reduce_norms == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_dim3_grid_reduce_norms,
                        0,
                        tmp_neurons_reduce_norms_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce norms dim3 grid. |END|
            
        // Allocating neurons reduce norms dim3 block.
        struct dim3 **tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms(static_cast<struct dim3**>(malloc(this->total_neuron_units_allocated * sizeof(struct dim3*))));
        if(tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms,
                        0,
                        this->total_neuron_units_allocated * sizeof(struct dim3*));

        struct dim3 *tmp_ptr_array_neuron_units_dim3_block_reduce_norms(static_cast<struct dim3*>(malloc(tmp_neurons_reduce_norms_size_so_far * sizeof(struct dim3))));
        if(tmp_ptr_array_neuron_units_dim3_block_reduce_norms == NULL)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        memset(tmp_ptr_array_neuron_units_dim3_block_reduce_norms,
                        0,
                        tmp_neurons_reduce_norms_size_so_far * sizeof(struct dim3));
        // |END| Allocating neurons reduce norms dim3 block. |END|
        
        // Assign global array.
        this->ptr_array_neuroyed_number_neurons_in_layer = tmp_ptr_array_neuroyed_number_neurons_in_layer;
        this->ptr_array_2D_neurons_reduce_norms = tmp_ptr_array_2D_neurons_position_reduce_norms_array;
        this->ptr_array_2D_neurons_dim3_grid_reduce_norms = tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms;
        this->ptr_array_2D_neurons_dim3_block_reduce_norms = tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms;
        // |END| Assign global array. |END|
        
        // Loop through each layers.
        for(layer_it = this->ptr_array_layers; layer_it != last_layer; ++layer_it)
        {
            // Get neurons array from that layer.
            tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units;

            // Get the reduce norms size of each neurons in that layer.
            tmp_layer_reduce_norms_size = *tmp_ptr_neuron_unit_it->ptr_reduce_norms_size;
            
            // Get the number of neurons in layer.
            tmp_number_neurons_in_layer = *layer_it->ptr_number_neurons;

            // Loop through each neurons in the layer.
            for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                    ++tmp_ptr_array_neuroyed_number_neurons_in_layer,
                                                                                                                                                                    ++tmp_ptr_array_2D_neurons_position_reduce_norms_array,
                                                                                                                                                                    ++tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms,
                                                                                                                                                                    ++tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms)
            {
                // Assign the number of neurons in the layer to the pointer.
                *tmp_ptr_array_neuroyed_number_neurons_in_layer = tmp_number_neurons_in_layer;
                
                // Result.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_reduce_norms_array = tmp_ptr_array_neuron_units_reduce_norms_results;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_reduce_norms = tmp_ptr_array_2D_neurons_position_reduce_norms_array;
                // |END| Result. |END|
                
                // Dim3 grid.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms = tmp_ptr_array_neuron_units_dim3_grid_reduce_norms;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_grid_reduce_norms = tmp_ptr_array_2D_neurons_position_dim3_grid_reduce_norms;
                // |END| Dim3 grid. |END|

                // Dim3 block.
                // Assign the position index of the begining results array from that array.
                *tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms = tmp_ptr_array_neuron_units_dim3_block_reduce_norms;

                // Assign the begining results array to that pointer.
                tmp_ptr_neuron_unit_it->ptr_array_2D_dim3_block_reduce_norms = tmp_ptr_array_2D_neurons_position_dim3_block_reduce_norms;
                // |END| Dim3 block. |END|

                // Number elements to reduce equal number of connections from the neuron.
                tmp_total_elements_to_reduce = *tmp_ptr_neuron_unit_it->ptr_number_forward_connections;
                
                // If is not the bias. (The bias have no elements to reduce.)
                if(tmp_total_elements_to_reduce != 0u)
                {
                    // Initialize index to zero.
                    tmp_index_dim3 = 0u;

                    // Loop to reduce "number of elements" to one at the end.
                    do
                    {
                        // Compute remaining results to reduce.
                        tmp_ptr_CUDA_Device->Grid_Block_Reduce_1Dimensions(tmp_total_elements_to_reduce,
                                                                                                                0,
                                                                                                                tmp_ptr_array_neuron_units_dim3_grid_reduce_norms[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)],
                                                                                                                tmp_ptr_array_neuron_units_dim3_block_reduce_norms[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)]);

                        // Get the remaining results to reduce.
                        tmp_total_elements_to_reduce = tmp_ptr_array_neuron_units_dim3_grid_reduce_norms[tmp_index_dim3 * (tmp_number_neurons_in_layer - 1u)].x;

                        // Increment index to dim3.
                        ++tmp_index_dim3;
                    } while(tmp_total_elements_to_reduce != 1u);
                    // |END| dim3. |END|
                    
                    // Increment the begining results by the layer reduce norms size.
                    tmp_ptr_array_neuron_units_reduce_norms_results += tmp_layer_reduce_norms_size;

                    // Increment the dim3 grid by one. (Access it by "iteration reduce" times "number neurons in layer minus bias".
                    ++tmp_ptr_array_neuron_units_dim3_grid_reduce_norms;

                    // Increment the dim3 grid by one. (Access it by "iteration reduce" times "number neurons in layer minus bias".
                    ++tmp_ptr_array_neuron_units_dim3_block_reduce_norms;
                }
            }
            
            // If some elements need to be reduce in the layer.
            if(tmp_layer_reduce_norms_size != 0u)
            {
                // Increment pointer by (number of neurons in layer minus bias) times (layer reduce summation size minus one).
                tmp_ptr_array_neuron_units_dim3_grid_reduce_norms += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_norms_size - 1u);
                tmp_ptr_array_neuron_units_dim3_block_reduce_norms += (tmp_number_neurons_in_layer - 1u) * (tmp_layer_reduce_norms_size - 1u);
            }
        }
        // |END| COMPUTE DIMENSION REDUCE NORMS. |END|
    }

    return true;
}

__device__ bool cuModel::Allocate__Normalized_Unit__Batch_Renormalization(void)
{
    if(this->total_neuron_units_allocated != 0u)
    {
        size_t tmp_number_neuron_units;

        var *tmp_ptr_array_parameters_scale_it(this->ptr_array_parameters + this->total_weights_allocated),
            *tmp_ptr_array_parameters_shift_it(this->ptr_array_parameters + this->total_weights_allocated + this->total_neuron_units_allocated),
        // TODO: Use only at training.
            *tmp_ptr_array_derivatives_parameters_scale_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated),
            *tmp_ptr_array_derivatives_parameters_shift_it(this->ptr_array_derivatives_parameters + this->total_weights_allocated + this->total_neuron_units_allocated);
        
        struct cuLayer const *const last_layer(this->ptr_last_layer);
        struct cuLayer *layer_it(this->ptr_array_layers);

        struct cuNeuron const *tmp_ptr_last_neuron_unit;
        struct cuNeuron *tmp_ptr_neuron_unit_it;
        
        // Allocating neuron unit(s) value(s) hat.
        var *tmp_ptr_array_neuron_units_values_hat(new var[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_values_hat == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_values_hat,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neuron unit(s) value(s) hat. |END|
        
        // Allocating neuron unit(s) value(s) normalize.
        var *tmp_ptr_array_neuron_units_values_normalize(new var[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_values_normalize == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_values_normalize,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neuron unit(s) value(s) normalize. |END|
        
        // Allocating neurons mean.
        var *tmp_ptr_array_neuron_units_mean_it(new var[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_mean_it == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_mean_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons mean. |END|
        
        // Allocating neurons variance.
        var *tmp_ptr_array_neuron_units_variance_it(new var[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_variance_it == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_variance_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons variance. |END|
        
        // Allocating neurons derivative mean.
        var *tmp_ptr_array_neuron_units_derivative_mean_it(new var[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_derivative_mean_it == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_derivative_mean_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons derivative mean. |END|
        
        // Allocating neurons derivative variance.
        var *tmp_ptr_array_neuron_units_derivative_variance_it(new var[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_derivative_variance_it == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_derivative_variance_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons derivative variance. |END|
        
        // Allocating neurons r correction.
        var *tmp_ptr_array_neuron_units_r_correction_it(new var[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_r_correction_it == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Zero_1D<var>(this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_r_correction_it,
                            this->ptr_array_dim3_grid + 3,
                            this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons r correction. |END|
        
        // Allocating neurons d correction.
        var *tmp_ptr_array_neuron_units_d_correction_it(new var[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_d_correction_it == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Zero_1D<var>(this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_d_correction_it,
                            this->ptr_array_dim3_grid + 3,
                            this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons d correction. |END|
        
        // Allocating neurons mean average.
        var *tmp_ptr_array_neuron_units_mean_average_it(new var[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_mean_average_it == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Zero_1D<var>(this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_mean_average_it,
                            this->ptr_array_dim3_grid + 3,
                            this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons mean average. |END|
        
        // Allocating neurons variance average.
        var *tmp_ptr_array_neuron_units_variance_average_it(new var[this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_variance_average_it == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Memory::Fill_1D<var>(this->total_neuron_units_allocated,
                                                             tmp_ptr_array_neuron_units_variance_average_it,
                                                             1_r,
                                                             this->ptr_array_dim3_grid + 3,
                                                             this->ptr_array_dim3_block + 3);
        // |END| Allocating neurons variance average. |END|
        
        this->ptr_array_normalized_batch_units_values_hats = tmp_ptr_array_neuron_units_values_hat;
        this->ptr_array_normalized_batch_units_values_normalizes = tmp_ptr_array_neuron_units_values_normalize;
        this->ptr_array_normalized_batch_units_scales = tmp_ptr_array_parameters_scale_it;
        this->ptr_array_normalized_batch_units_shifts = tmp_ptr_array_parameters_shift_it;
        this->ptr_array_normalized_batch_units_means = tmp_ptr_array_neuron_units_mean_it;
        this->ptr_array_normalized_batch_units_variances = tmp_ptr_array_neuron_units_variance_it;
        this->ptr_array_normalized_batch_units_derivatives_means = tmp_ptr_array_neuron_units_derivative_mean_it;
        this->ptr_array_normalized_batch_units_derivatives_variances = tmp_ptr_array_neuron_units_derivative_variance_it;
        this->ptr_array_normalized_batch_units_r_corrections = tmp_ptr_array_neuron_units_r_correction_it;
        this->ptr_array_normalized_batch_units_d_corrections = tmp_ptr_array_neuron_units_d_correction_it;
        this->ptr_array_normalized_batch_units_means_averages = tmp_ptr_array_neuron_units_mean_average_it;
        this->ptr_array_normalized_batch_units_variances_averages = tmp_ptr_array_neuron_units_variance_average_it;
        
        for(; layer_it != last_layer; ++layer_it)
        {
            if((tmp_number_neuron_units = *layer_it->ptr_number_neurons) != 0u)
            {
                for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit,
                    tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_values_hat,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_values_normalize,
                                                                                                                                                                      ++tmp_ptr_array_parameters_scale_it,
                                                                                                                                                                      ++tmp_ptr_array_parameters_shift_it,
                                                                                                                                                                      ++tmp_ptr_array_derivatives_parameters_scale_it,
                                                                                                                                                                      ++tmp_ptr_array_derivatives_parameters_shift_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_r_correction_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_d_correction_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_mean_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_variance_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_derivative_mean_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_derivative_variance_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_mean_average_it,
                                                                                                                                                                      ++tmp_ptr_array_neuron_units_variance_average_it)
                {
                    tmp_ptr_neuron_unit_it->ptr_array_values_hats = tmp_ptr_array_neuron_units_values_hat;
                    tmp_ptr_neuron_unit_it->ptr_array_values_normalizes = tmp_ptr_array_neuron_units_values_normalize;
                    tmp_ptr_neuron_unit_it->ptr_scale = tmp_ptr_array_parameters_scale_it; *tmp_ptr_array_parameters_scale_it = 1_r;
                    tmp_ptr_neuron_unit_it->ptr_shift = tmp_ptr_array_parameters_shift_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_scales = tmp_ptr_array_derivatives_parameters_scale_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_shifts = tmp_ptr_array_derivatives_parameters_shift_it;
                    tmp_ptr_neuron_unit_it->ptr_array_means = tmp_ptr_array_neuron_units_mean_it;
                    tmp_ptr_neuron_unit_it->ptr_array_variances = tmp_ptr_array_neuron_units_variance_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_means = tmp_ptr_array_neuron_units_derivative_mean_it;
                    tmp_ptr_neuron_unit_it->ptr_array_derivatives_variances = tmp_ptr_array_neuron_units_derivative_variance_it;
                    tmp_ptr_neuron_unit_it->ptr_r_correction = tmp_ptr_array_neuron_units_r_correction_it;
                    tmp_ptr_neuron_unit_it->ptr_d_correction = tmp_ptr_array_neuron_units_d_correction_it;
                    tmp_ptr_neuron_unit_it->ptr_mean_average = tmp_ptr_array_neuron_units_mean_average_it;
                    tmp_ptr_neuron_unit_it->ptr_variance_average = tmp_ptr_array_neuron_units_variance_average_it;
                }

                tmp_ptr_array_neuron_units_values_hat += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_values_normalize += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_mean_it += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_variance_it += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_derivative_mean_it += (this->batch_size - 1u) * tmp_number_neuron_units;
                tmp_ptr_array_neuron_units_derivative_variance_it += (this->batch_size - 1u) * tmp_number_neuron_units;
            }
        }
    }
    else { return false; }

    return true;
}

__device__ bool cuModel::Allocate__Neuron__Batch_Renormalization_Transpose(void)
{
    if(this->total_neuron_units_allocated != 0u)
    {
        struct cuLayer const *const last_layer(this->ptr_last_layer);
        struct cuLayer *layer_it(this->ptr_array_layers);

        struct cuNeuron const *tmp_ptr_last_neuron_unit;
        struct cuNeuron *tmp_ptr_neuron_unit_it;
        
        // Allocating neurons mean.
        var *tmp_ptr_array_neuron_units_transposed_mean_it(new var[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_transposed_mean_it == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_transposed_mean_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons mean. |END|
        
        // Allocating neurons variance.
        var *tmp_ptr_array_neuron_units_transposed_variance_it(new var[this->batch_size * this->total_neuron_units_allocated]);
        if(tmp_ptr_array_neuron_units_transposed_variance_it == nullptr)
        {
            ERR(L"Can not allocate memory.",);

            return false;
        }
        Zero_1D<var>(this->batch_size * this->total_neuron_units_allocated,
                            tmp_ptr_array_neuron_units_transposed_variance_it,
                            this->ptr_array_dim3_grid + 5,
                            this->ptr_array_dim3_block + 5);
        // |END| Allocating neurons variance. |END|
        
        this->ptr_array_neuron_units_transposed_mean = tmp_ptr_array_neuron_units_transposed_mean_it;
        this->ptr_array_neuron_units_transposed_variance = tmp_ptr_array_neuron_units_transposed_variance_it;
        
        for(; layer_it != last_layer; ++layer_it)
        {
            for(tmp_ptr_last_neuron_unit = layer_it->ptr_last_neuron_unit,
                tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
            {
                tmp_ptr_neuron_unit_it->ptr_array_transposed_mean = tmp_ptr_array_neuron_units_transposed_mean_it;
                tmp_ptr_neuron_unit_it->ptr_array_transposed_variance = tmp_ptr_array_neuron_units_transposed_variance_it;

                tmp_ptr_array_neuron_units_transposed_mean_it += this->batch_size;
                tmp_ptr_array_neuron_units_transposed_variance_it += this->batch_size;
            }
        }
    }
    else { return false; }

    return true;
}

__device__ bool cuModel::Allocate__Neuron__Mask_Dropout_Bernoulli(void)
{
    if(this->ptr_array_af_units_mask_dropout_bernoulli == nullptr)
    {
        if(this->total_neuron_units == 0u)
        {
            ERR(L"Can not allocate neurons mask dropout. no neuron available.",);

            return false;
        }

        bool *tmp_ptr_array_af_units_mask_dropout_bernoulli(new bool[this->total_neuron_units]);
        if(tmp_ptr_array_af_units_mask_dropout_bernoulli == nullptr)
        {
            ERR(L"Can not allocate neurons mask dropout.",);

            return false;
        }
        Zero_1D<bool>(this->total_neuron_units,
                              tmp_ptr_array_af_units_mask_dropout_bernoulli,
                              this->ptr_array_dim3_grid + 3,
                              this->ptr_array_dim3_block + 3);

        this->Reset__Parameter__AF_Units__Mask_Dropout(tmp_ptr_array_af_units_mask_dropout_bernoulli);
    }

    return true;
}

__device__ bool cuModel::Allocate__Batch_Normalization()
{
    // TODO: Reorganasition of the array. [------Weights-----][----Bias----][----Batch renormalization----]. Allocating with the size of each layer. No waste of memory.
    if(this->ptr_array_parameters != nullptr)
    {
        size_t const tmp_new_size(2u * this->total_neuron_units_allocated + this->total_parameters_allocated);
        
        if(this->Reallocate__Parameter(tmp_new_size) == false)
        {
            ERR(L"From \"Reallocate__Parameter\".",);

            return false;
        }
        else if(this->Allocate__Normalized_Unit__Batch_Renormalization() == false)
        {
            ERR(L"From \"Allocate__Normalized_Unit__Batch_Renormalization\".",);

            return false;
        }
        // TODO: allocate only at training.
        else if(this->Allocate__Neuron__Batch_Renormalization_Transpose() == false)
        {
            ERR(L"From \"Allocate__Neuron__Batch_Renormalization_Transpose\".",);

            return false;
        }
    }
    else { return false; }

    return true;
}

__device__ bool cuModel::Allocate__Parameter__Regularization(void)
{
    if(this->ptr_array_mask_regularized_parameters == nullptr)
    {
        this->ptr_array_mask_regularized_parameters = new var[this->total_parameters_allocated];
        if(this->ptr_array_mask_regularized_parameters == nullptr)
        {
            ERR(L"Can not allocate memory!",);

            return false;
        }
        Zero_1D<var>(this->total_parameters_allocated,
                            this->ptr_array_mask_regularized_parameters,
                            this->ptr_array_dim3_grid + 1,
                            this->ptr_array_dim3_block + 1);
    }

    return true;
}

[[deprecated("Not properly implemented.")]] __device__ bool cuModel::Allocate__Neuron(struct cuNeuron *ptr_neuron_received)
{
    /*
    if(ptr_neuron_received == NULL)
    {
        INFO(L"Allocate__Neuron => Neuron_unit is NULL");
        return false;
    }

    if(ptr_neuron_received->ptr_first_forward_connection_index == NULL)
    {
        ptr_neuron_received->ptr_first_forward_connection_index = static_cast<size_t*>(malloc(sizeof(size_t)));
        if(ptr_neuron_received->ptr_first_forward_connection_index == NULL)
        {
            INFO(L"Allocate__Neuron => Can not allocate memory. ptr_first_forward_connection_index = malloc(%u)", sizeof(size_t));
            return false;
        }
    }
    *ptr_neuron_received->ptr_first_forward_connection_index = 0u;

    if(ptr_neuron_received->ptr_last_forward_connection_index == NULL)
    {
        ptr_neuron_received->ptr_last_forward_connection_index = static_cast<size_t*>(malloc(sizeof(size_t)));
        if(ptr_neuron_received->ptr_last_forward_connection_index == NULL)
        {
            INFO(L"Allocate__Neuron => Can not allocate memory. ptr_last_forward_connection_index = malloc(%u)", sizeof(size_t));
            return false;
        }
    }
    *ptr_neuron_received->ptr_last_forward_connection_index = 0u;

    if(ptr_neuron_received->ptr_type_activation_function == NULL)
    {
        ptr_neuron_received->ptr_type_activation_function = static_cast<DL::ACTIVATION::TYPE*>(malloc(sizeof(enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS)));
        if(ptr_neuron_received->ptr_type_activation_function == NULL)
        {
            INFO(L"Allocate__Neuron => Can not allocate memory. ptr_type_activation_function = malloc(%u)", sizeof(enum DL::ENUM_TYPE_ACTIVATION_FUNCTIONS));
            return false;
        }
    }
    *ptr_neuron_received->ptr_type_activation_function = DL::ENUM_TYPE_ACTIVATION_FUNCTIONS::SIGMOID;
    this->Set__Activation_Function_Neuron(ptr_neuron_received);

    if(ptr_neuron_received->sum == NULL)
    {
        ptr_neuron_received->sum = static_cast<var*>(malloc(sizeof(var)));
        if(ptr_neuron_received->sum == NULL)
        {
            INFO(L"Allocate__Neuron => Can not allocate memory. sum = malloc(%u)", sizeof(var));
            return false;
        }
    }
    *ptr_neuron_received->sum = 0_r;

    if(ptr_neuron_received->value == NULL)
    {
        ptr_neuron_received->value = static_cast<var*>(malloc(sizeof(var)));
        if(ptr_neuron_received->value == NULL)
        {
            INFO(L"Allocate__Neuron => Can not allocate memory. value = malloc(%u)", sizeof(var));
            return false;
        }
    }
    *ptr_neuron_received->value = 0_r;
    */

    return true;
}
