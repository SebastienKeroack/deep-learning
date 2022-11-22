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

#pragma once

#include "deep-learning-lib/v1/data/enum/model.hpp"
#include "deep-learning-lib/data/enum/activation.hpp"
#include "deep-learning-lib/v1/data/enum/layer.hpp"
#include "deep-learning-lib/v1/data/enum/layer_activation.hpp"
#include "deep-learning-lib/v1/data/enum/layer_dropout.hpp"
#include "deep-learning-lib/v1/data/enum/layer_norm.hpp"
#include "deep-learning-lib/v1/data/enum/loss_fn.hpp"
#include "deep-learning-lib/v1/data/enum/optimizer.hpp"
#include "deep-learning-lib/v1/data/enum/propagation.hpp"
#include "deep-learning-lib/data/enum/env.hpp"

struct cuNeuron {
  // Default constructor.
  __device__ cuNeuron(void) {}

  // N: Number of threads.
  // T: Number of times to predict.
  // P: Number of parameters.

  // Dropout variable.
  bool *ptr_mask_dropout_bernoulli = nullptr;  // size[1].
  // |END| Dropout variable. |END|

  size_t *ptr_first_connection_index = nullptr;  // size[1].
  size_t *ptr_last_connection_index = nullptr;   // size[1].
  size_t *ptr_number_connections = nullptr;      // size[1].
  size_t *ptr_reduce_summation_size = nullptr;   // size[1].
  size_t *ptr_reduce_error_size = nullptr;       // size[1].
  size_t *ptr_reduce_norms_size = nullptr;       // size[1].
  size_t *ptr_reduce_batch_size = nullptr;       // size[1].

  var *ptr_array_summations = nullptr;      // size[N, T].
  var *ptr_array_values = nullptr;          // size[N, T].
  var *ptr_array_errors = nullptr;          // size[N, T].
  var **ptr_array_reduce_summation =
      nullptr;  // size[N, T], size[ptr_reduce_summation_size].
  var **ptr_array_reduce_error =
      nullptr;  // size[N, T], size[ptr_reduce_error_size].
  var **ptr_array_reduce_norms =
      nullptr;  // size[1], size[ptr_reduce_norms_size].
  var **ptr_array_reduce_mean =
      nullptr;  // size[1], size[ptr_reduce_batch_size].
  var **ptr_array_reduce_variance =
      nullptr;  // size[1], size[ptr_reduce_batch_size].

  DL::ACTIVATION::TYPE
      *ptr_type_activation_function = nullptr;  // size[1].

  // Batch renormalization variable.
  var *ptr_array_values_hats = nullptr;            // size[N, T].
  var *ptr_array_values_normalizes = nullptr;      // size[N, T].
  var *ptr_scale = nullptr;                        // size[1].
  var *ptr_shift = nullptr;                        // size[1].
  var *ptr_array_derivatives_scales = nullptr;     // size[N].
  var *ptr_array_derivatives_shifts = nullptr;     // size[N].
  var *ptr_array_means = nullptr;                  // size[N, T?].
  var *ptr_array_variances = nullptr;              // size[N, T?].
  var *ptr_array_transposed_mean = nullptr;        // size[N, T?].
  var *ptr_array_transposed_variance = nullptr;    // size[N, T?].
  var *ptr_array_derivatives_means = nullptr;      // size[N].
  var *ptr_array_derivatives_variances = nullptr;  // size[N].
  var *ptr_r_correction = nullptr;                 // size[1].
  var *ptr_d_correction = nullptr;                 // size[1].
  var *ptr_mean_average = nullptr;                 // size[1].
  var *ptr_variance_average = nullptr;             // size[1].
  // |END| Batch renormalization variable. |END|

  struct dim3 *ptr_dim3_grid_connections = NULL;   // size[1].
  struct dim3 *ptr_dim3_block_connections = NULL;  // size[1].
  struct dim3 *ptr_array_dim3_grid_reduce_summation =
      NULL;  // size[ptr_reduce_summation_size].
  struct dim3 *ptr_array_dim3_block_reduce_summation =
      NULL;  // size[ptr_reduce_summation_size].
  struct dim3 *ptr_array_dim3_grid_reduce_error =
      NULL;  // size[ptr_reduce_error_size].
  struct dim3 *ptr_array_dim3_block_reduce_error =
      NULL;  // size[ptr_reduce_error_size].
  struct dim3 *ptr_array_dim3_grid_reduce_threads =
      NULL;  // size[ptr_reduce_batch_size].
  struct dim3 *ptr_array_dim3_block_reduce_threads =
      NULL;  // size[ptr_reduce_batch_size].
  struct dim3 **ptr_array_2D_dim3_grid_reduce_norms =
      NULL;  // size[ptr_reduce_norms_size].
  struct dim3 **ptr_array_2D_dim3_block_reduce_norms =
      NULL;  // size[ptr_reduce_norms_size].

  // cuRAND.
  struct curandStateMtgp32 *ptr_cuRAND_State_MTGP32 = nullptr;
  // |END| cuRAND. |END|
};

struct cuLayer {
  // Default constructor.
  __device__ cuLayer(void) {}

  // N: Number of threads.
  // T: Number of times to predict.
  // H: Number of neurons in layer.
  // K: Number of blocks in layer.
  // C: Number of cells in layer.

  bool use_Batch_Stride = false;

  DL::LAYER::TYPE type_layer =
      DL::LAYER::NONE;
  DL::LAYER_ACTIVATION::TYPE type_activation =
      DL::LAYER_ACTIVATION::NONE;

  // FC layer variable.
  size_t *ptr_first_neuron_index = nullptr;  // size[H].
  size_t *ptr_last_neuron_index = nullptr;   // size[1].
  size_t *ptr_number_neurons = nullptr;      // size[1].

  struct cuNeuron *ptr_array_neuron_units = nullptr;  // size[H].
  struct cuNeuron *ptr_last_neuron_unit = nullptr;    // size[1].
  // |END| FC layer variable. |END|

  // Dropout layer variable.
  var dropout_values[2] = {0};
  // |END| Dropout layer variable. |END|

  // Batch renormalization layer variable.
  bool use_Batch_Renormalization = false;
  // |END| Batch renormalization layer variable. |END|

  struct dim3 *ptr_dim3_grid_neurons = nullptr;          // size[1].
  struct dim3 *ptr_dim3_block_neurons = nullptr;         // size[1].
  struct dim3 *ptr_dim3_grid_neurons_DP = nullptr;       // size[1].
  struct dim3 *ptr_dim3_block_neurons_DP = nullptr;      // size[1].
  struct dim3 *ptr_dim3_grid_neurons_cuRAND = nullptr;   // size[1].
  struct dim3 *ptr_dim3_block_neurons_cuRAND = nullptr;  // size[1].
  struct dim3 *ptr_dim3_grid_batch_neurons = nullptr;    // size[1].
  struct dim3 *ptr_dim3_block_batch_neurons = nullptr;   // size[1].
  struct dim3 *ptr_dim3_grid_weights = nullptr;          // size[1].
  struct dim3 *ptr_dim3_block_weights = nullptr;         // size[1].

  class cuDims *ptr_Class_Storage_Dim3_Batch = nullptr;  // size[1].
};

// get_n_data() / get_seq_w() : 1u
// total_parameters : 2u
// ((ptr_last_layer - 1) - (ptr_array_layers + 1)) + 1u : 3u
#define TOTAL_KERNEL_PARALLEL 9u

enum ENUM_TYPE_CURAND_GENERATOR : unsigned int {
  TYPE_CURAND_WEIGHTS = 0,
  TYPE_CURAND_BERNOULLI = 1u
};

class cuModel {
  // N: Number of threads.
  // T: Number of times to predict.
  // L: Number of layers.
  // H: Number of neurons.
  // K: Number of blocks.
  // C: Number of cells.
  // P: Number of parameters.
  // W: Number of weights.

 public:
  __host__ __device__ cuModel(void);
  __host__ __device__ ~cuModel(void);

  __host__ void Set__Limit_Device_Runtime_Pending_Launch_Count(
      size_t limit_device_runtime_pending_launch_count_received = 0u);
  __host__ __device__ void Set__Maximum_Allowable_Memory(
      size_t const available_memory_mbs_received);
  __host__ __device__ bool update_mem_thread_size(size_t number_threads_received);
  __host__ __device__ bool update_mem_batch_size(size_t const batch_size);
  __host__ __device__ void reset_loss(void);
  __device__ void merge_mp_accu_loss(void);
  __device__ void device__Clear_Train_Arrays(void);
  __device__ void compute_error(size_t const batch_size,
                                 var **const ptr_array_outputs_received);
  __device__ void FF__Compute__Error(size_t const batch_size,
                                     var **const ptr_array_outputs_received);
  __device__ void FF__Compute__Error__Standard(
      size_t const batch_size, var **const ptr_array_outputs_received);
  __device__ void FF__Compute__Error__Binary_Cross_Entropy(
      size_t const batch_size, var **const ptr_array_outputs_received);
  __device__ void FF__Compute__Error__Bit_Fail(
      size_t const batch_size, var **const ptr_array_outputs_received);
  __device__ void Test(size_t const batch_size,
                       var **const ptr_array_outputs_received,
                       size_t const time_step_index_received = 0u);
  __device__ void FF__Test(size_t const batch_size,
                           var **const ptr_array_outputs_received);
  __device__ void FF__Test__Standard(size_t const batch_size,
                                     var **const ptr_array_outputs_received);
  __device__ void FF__Test__Binary_Cross_Entropy(
      size_t const batch_size, var **const ptr_array_outputs_received);
  __device__ void FF__Test__Bit_Fail(size_t const batch_size,
                                     var **const ptr_array_outputs_received);
  __device__ void RNN__Test(size_t const batch_size,
                            var **const ptr_array_outputs_received,
                            size_t const time_step_index_received = 0u);
  __device__ void Initialize_Candidate_Weights(
      size_t const first_connection_received,
      size_t const last_connection_received, float const scale_factor_received);
  __device__ void Reset__Link_Connections(void);
  __device__ void Add_Candidate_Neuron(struct cuLayer *ptr_layer_received);
  __device__ void Update_Candidate_Slopes(
      class cuModel *ptr_cuModel_received = NULL);
  __device__ void Update_Candidate_Weights(
      size_t const number_examples_received);
  __device__ bool Set__Probability_Retained_Unit(
      size_t const index_layer_received,
      real const retention_probability_received,
      bool const scale_weights_received = true);
  __device__ bool Set__Probability_Retained_Unit(
      struct cuLayer *ptr_layer_received,
      real const retention_probability_received,
      bool const scale_weights_received = true);
  __device__ bool Set__Batch_Renormalization(size_t const index_layer_received,
                                             bool const Set__received = true);
  __device__ bool Set__Batch_Renormalization(
      struct cuLayer *const ptr_layer_received,
      bool const Set__received = true);
  __device__ void Scale_Weight__Dropout(
      var const scale_factor_received,
      struct cuLayer const *const layer_it);
  __device__ void Scale_Weight__FC__Forward__Dropout(
      var const scale_factor_received,
      struct cuLayer const *const layer_it);
  __host__ __device__ void set_loss_fn(
      DL::LOSS_FN::TYPE const
          type_loss_function_received);
  __host__ __device__ void set_accu_fn(
      DL::ACCU_FN::TYPE const
          type_accuracy_function_received);
  __host__ __device__ void set_bit_fail_limit(
      var const bit_fail_limit);
  __host__ __device__ void set_optimizer(
      DL::OPTIMIZER::TYPE const
          optimizer_function_received);
  __device__ void Deallocate__Parameter__Optimizer(void);
  __device__ void Deallocate__Parameter__Gradient_Descent(void);
  __device__ void Deallocate__Parameter__iRPROP_minus(void);
  __device__ void Deallocate__Parameter__iRPROP_plus(void);
  __device__ void Deallocate__Parameter__Adam(void);
  __device__ void Deallocate__Parameter__AMSGrad(void);
  __device__ void Deallocate__Parameter__Regularization(void);
  __device__ void Deallocate_Cost(void);
  __device__ void Deallocate_Reduce_Batch(void);
  __device__ void Deallocate_Reduce_Cost(void);
  __device__ void Deallocate_Batch_Reduce(void);
  __device__ void Deallocate__Normalized_Unit__Batch_Normalization(void);
  __device__ void Deallocate__Neurons_Reduce_Summation(void);
  __device__ void Deallocate__Neurons_Reduce_Error(void);
  __device__ void Deallocate__Neurons_Reduce_Norms(void);
  __device__ void Deallocate__Neuron__Mask_Dropout_Bernoulli(void);
  __device__ void Deallocate__Cell_Unit__Mask_Dropout_Zoneout(void);
  __device__ void Remove_Batch_Normalization(void);
  __device__ void Clear_Optimizer(void);
  __device__ void Reset__Parameter__Normalized_Unit(void);
  __device__ void Reset__Derivative_Parameter__Normalized_Unit(void);
  __device__ void Update_Parameter(size_t const batch_size,
                                   size_t const training_size);
  __device__ void Update_Parameter__Gradient_Descent(
      size_t const batch_size, size_t const training_size,
      size_t const start_index_received, size_t const end_index_received);
  __device__ void Update_Parameter__Gradient_Descent__CUDA(
      size_t const batch_size, size_t const training_size,
      size_t const start_index_received, size_t const end_index_received);
  __device__ void Update_Parameter__Gradient_Descent_Momentum__CUDA(
      size_t const batch_size, size_t const training_size,
      size_t const start_index_received, size_t const end_index_received);
  __device__ void Update_Parameter__Nesterov_Accelerated_Gradient__CUDA(
      size_t const batch_size, size_t const training_size,
      size_t const start_index_received, size_t const end_index_received);
  __device__ void Update_Parameter__iRPROP_plus(
      size_t const start_index_received, size_t const end_index_received);
  __device__ void Update_Parameter__iRPROP_plus__CUDA(
      size_t const start_index_received, size_t const end_index_received);
  __device__ void Update_Parameter__iRPROP_plus__CUDA__Dropout(
      size_t const start_index_received, size_t const end_index_received);
  __device__ void Update_Parameter__Adam(size_t const batch_size,
                                         size_t const training_size,
                                         size_t const start_index_received,
                                         size_t const end_index_received);
  __device__ void Update_Parameter__AMSGrad(size_t const batch_size,
                                            size_t const training_size,
                                            size_t const start_index_received,
                                            size_t const end_index_received);
  __device__ void Update_Weight_Regularization__Max_Norm_Constraints(void);
  __device__ void Update_Weight_Regularization__Max_Norm_Constraints__Neurons(
      struct cuLayer const *const layer_it,
      struct cuLayer const *const last_layer);
  __device__ void merge_mp_derivatives(void);
  __host__ __device__ void Launch_Randomize_Weights(
      var const minimum_weight_received, var const maximum_weight_received);
  __host__ __device__ void Set__Accurancy_Variance(
      float const accurancy_variance_received);
  __host__ __device__ void set_seq_w(
      size_t const time_delays_received);
  __device__ void set_accu(
      DL::ENV::TYPE const type_accuracy_received,
      float const accurancy_received);
  __device__ void set_loss(
      DL::ENV::TYPE const type_error_received,
      float const loss_received);
  __device__ void Indexing_Regularization_Parameters(void);
  __device__ void Indexing_Regularization__Weights__FC__Forward(
      struct cuLayer const *const layer_it);
  __device__ void Update_Derivative_Weight__Regularization__L1(
      size_t const batch_size);
  __device__ void Update_Derivative_Weight__Regularization__L2(
      size_t const batch_size);
  __device__ void Transpose_Layer_Forward__Batch_Normalization(
      struct cuLayer *const layer_it);
  __device__ void Transpose_Layer_Backward__Batch_Normalization(
      struct cuLayer *const layer_it);
  __device__ void Transpose_Weights(void);
  __device__ void Prepare__Global__Grids_Blocks_Dimensions(void);
  __device__ bool Prepare__Layers__Grids_Blocks_Dimensions(void);
  __device__ bool Prepare__Neurons__Grids_Blocks_Dimensions(void);
  __device__ void Prepare__Parameters__Grids_Blocks_Dimensions(void);
  __device__ void Prepare__Threads__Grids_Blocks_Dimensions(
      size_t const number_threads_received);
  __device__ void Prepare__Threads_Parameters__Grids_Blocks_Dimensions(
      size_t const number_threads_received);
  __device__ void Prepare__Batch__Grids_Blocks_Dimensions(
      size_t const batch_size);
  __device__ void Prepare__Batch_Layers__Grids_Blocks_Dimensions(
      size_t const batch_size);
  __device__ void Prepare__Batch_Neurons__Grids_Blocks_Dimensions(
      size_t const batch_size);
  __device__ void Copy__Neuron_Unit(
      struct cuNeuron *const ptr_copy_neuron_received,
      size_t const neuron_first_connection_index_received,
      size_t const neuron_last_connection_index_received,
      DL::ACTIVATION::TYPE const
          neuron_activation_function_received);
  __device__ void Copy__Neurons(
      size_t const *ptr_array_neuron_first_connection_index_received,
      size_t const *ptr_array_neuron_last_connection_index_received,
      DL::ACTIVATION::TYPE const
          *ptr_array_neuron_activation_function_received,
      struct cuNeuron *const ptr_array_copy_first_neuron_received,
      struct cuNeuron *const ptr_array_copy_last_neuron_received);
  __device__ void Copy__FC_to_FC(
      struct cuNeuron *ptr_copy_neuron_it_received,
      struct cuNeuron const *const ptr_copy_last_neuron_received,
      struct cuNeuron *const ptr_copy_first_neuron_received,
      size_t const *&ptr_array_neuron_units_first_connection_index_received,
      size_t const *&ptr_array_neuron_units_last_connection_index_received,
      DL::ACTIVATION::TYPE const *
          &ptr_array_neuron_units_activation_function_received);
  __device__ void Reset__Parameter__Mask_Dropout(
      bool *ptr_array_neuron_units_mask_dropout_received);
  __device__ void Dropout(void);
  __device__ void Dropout__FC_to(
      size_t &ref_sync_code_received,
      bool const use_parameters_dropout_received,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received);
  __device__ void Dropout_Bernoulli__FC_to_FC(
      size_t &ref_sync_code_received,
      bool const use_parameters_dropout_received,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received);
  __device__ void Dropout__FC_to__Batch_Normalization(
      size_t &ref_sync_code_received,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received);
  __device__ void Dropout_Bernoulli__FC_to_FC__Batch_Renormalization(
      size_t &ref_sync_code_received,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received);
  __device__ void Assign_Inputs(bool &ref_synchronized_received,
                                size_t const thread_index_received,
                                var const *ptr_array_inputs_received);
  __device__ void forward_pass(
      size_t const batch_size,
      var const *const *const Xm);
  __device__ void FF__Forward_Pass_Batch(
      size_t const batch_size,
      var const *const *const Xm);
  __device__ void Assign_Inputs_Batch(
      bool &ref_synchronized_received, size_t const batch_size,
      var const *const *const Xm);
  __device__ void Forward_Pass__FC_to(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to__Dropout_Bernoulli__Testing(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to__Dropout(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to__Batch_Renormalization__Loop(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void
  Forward_Pass__FC_to__Batch_Renormalization__Dropout_Bernoulli__Testing(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to__Batch_Renormalization__Training(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to__Batch_Renormalization__Dropout(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to_FC(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to_FC__Softmax(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to_FC__Dropout_Bernoulli__Testing(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to_FC__Dropout(
      bool &ref_synchronized_received, size_t const thread_index_received,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to_FC__Batch_Renormalization__Loop(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void
  Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout_Bernoulli__Testing(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to_FC__Batch_Renormalization__Training(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Forward_Pass__FC_to_FC__Batch_Renormalization__Dropout(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  //__device__ void Compute__Error_Tanh_FF(size_t const thread_index_received,
  //var const *const ptr_array_desireds_outputs_received);
  __device__ void backward_pass(size_t const batch_size);
  __device__ void FF__Backward_Pass_Batch(size_t const batch_size);
  __device__ void Backward_Pass__FC_to(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer *const ptr_next_layer_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Backward_Pass__FC_to__Dropout(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer *const ptr_next_layer_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Backward_Pass__FC_to__Batch_Renormalization(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Backward_Pass__FC_to__Batch_Renormalization__Dropout(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Backward_Pass__FC_to_FC(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer *const ptr_next_layer_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Backward_Pass__FC_to_FC__Dropout(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer *const ptr_next_layer_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Backward_Pass__FC_to_FC__Batch_Renormalization(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Backward_Pass__FC_to_FC__Batch_Renormalization__Dropout(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void update_derivatives(
      size_t const batch_size,
      size_t const time_step_index_received = 0u);
  __device__ void FF__Update_Derivative_Weight(
      size_t const batch_size);
  __device__ void Update_Derivative_Weight__FC_to(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Update_Derivative_Weight__FC_to__Dropout(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Update_Derivative_Weight__FC_to_FC(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);
  __device__ void Update_Derivative_Weight__FC_to_FC__Dropout(
      bool &ref_synchronized_received, size_t const batch_size,
      struct cuLayer *const layer_it,
      struct cuLayer const *const ptr_previous_layer_it_received,
      struct dim3 const *const ptr_dim3_batch_size_grid_received,
      struct dim3 const *const ptr_dim3_batch_size_block_received);

  __device__ bool Multi_Class_Classification(void) const;
  __host__ bool Initialize_CUDA_Device(void);
  __host__ bool Initialize_cuRAND(size_t const seed);
  __device__ bool Initialize_cuRAND_MTGP32(
      int const size_received,
      enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received,
      struct curandStateMtgp32 *const ptr_curandStateMtgp32);
  __host__ __device__ bool Allocate__Structure(
      size_t const n_layers,
      size_t const allowable_memory);
  __device__ bool Add_CUDA_Device(
      int const index_device_received,
      struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
  __device__ bool Reallocate_Connections(
      size_t const total_connections_received);
  __device__ bool Reallocate_Neurons(
      size_t const total_neuron_units_received,
      bool const reSet__neuron_position_received);
  __device__ bool Reallocate_Layers(size_t const total_layers_received);
  __device__ bool Reallocate__Parameter__Regularization(
      size_t const number_parameters_received);
  __device__ bool Reallocate__Parameter__Dropout_Bernoulli(
      size_t const number_parameters_received);
  __device__ bool Reallocate__Parameter__Optimizer(
      size_t const number_parameters_received);
  __device__ bool Reallocate__Parameter__Gradient_Descent(
      size_t const number_parameters_received);
  __device__ bool Reallocate__Parameter__iRPROP_minus(
      size_t const number_parameters_received);
  __device__ bool Reallocate__Parameter__iRPROP_plus(
      size_t const number_parameters_received);
  __device__ bool Reallocate__Parameter__Adam(
      size_t const number_parameters_received);
  __device__ bool Reallocate__Parameter__AMSGrad(
      size_t const number_parameters_received);
  __device__ bool Allocate_Weights_Transposed(void);
  __device__ bool Allocate__Parameter(void);
  __device__ bool Allocate__Parameter__Optimizer(void);
  __device__ bool Allocate__Parameter__Gradient_Descent(void);
  __device__ bool Allocate__Parameter__iRPROP_minus(void);
  __device__ bool Allocate__Parameter__iRPROP_plus(void);
  __device__ bool Allocate__Parameter__Adam(void);
  __device__ bool Allocate__Parameter__AMSGrad(void);
  __device__ bool Allocate__Parameter__Regularization(void);
  __device__ bool Allocate__Batch_Normalization(void);
  __device__ bool Allocate_Reduce_Threads(void);
  __device__ bool Allocate_Reduce_Threads_Dim(void);
  __device__ bool Allocate_Reduce_Threads_Dim_DP(void);
  __device__ bool Allocate_Reduce_Cost(void);
  __device__ bool Allocate__Neuron_Units(void);
  __device__ bool Allocate__Neurons_Reduce_Norms(void);
  __device__ bool Allocate__Neurons_Reduce_Summation(void);
  __device__ bool Allocate__Neurons_Reduce_Error(void);
  __device__ bool Allocate__Neurons_Reduce_Batch_Normalization(void);
  __device__ bool Allocate__Neuron__Mask_Dropout_Bernoulli(void);
  __device__ bool Allocate__Normalized_Unit__Batch_Renormalization(void);
  __device__ bool Allocate__Neuron__Batch_Renormalization_Transpose(void);
  __device__ bool Reallocate__Thread(size_t const number_threads_received);
  __device__ bool Reallocate__Batch(size_t const batch_size);
  __device__ bool Reallocate__Thread__Cost(size_t const batch_size);
  __device__ bool Reallocate_Reduce_Threads(size_t const batch_size);
  __device__ bool Reallocate_Reduce_Threads_Dim(
      size_t const batch_size);
  __device__ bool Reallocate_Reduce_Threads_Dim_DP(
      size_t const batch_size);
  __device__ bool Reallocate_Reduce_Cost(
      size_t const total_reduce_batch_size_received);
  __device__ bool Reallocate__Batch__Neuron_Unit(
      size_t const batch_size);
  __device__ bool Reallocate__Batch__Neuron_Reduce_Summation(
      size_t const batch_size);
  __device__ bool Reallocate__Batch__Neuron_Reduce_Error(
      size_t const batch_size);
  __device__ bool Reallocate__Normalized_Unit__Batch_Normalization(
      size_t const batch_size);
  __device__ bool Reallocate__Batch__Neuron_Batch_Normalization_Transpose(
      size_t const batch_size);
  __device__ bool Reallocate__Batch__Neuron_Batch_Normalization_Reduce(
      size_t const batch_size);
  __device__ bool Reallocate__Thread__Parameter(
      size_t const batch_size);
  __device__ bool Reallocate__Parameter(size_t const batch_size);
  __host__ bool Copy__Host_To_Device(
      class Model const *const ptr_host_Neural_Network_received,
      size_t const allowable_memory);
  __host__ __device__ void Copy__Optimizer_Parameters(
      class Model const *const model);
  __host__ __device__ void Copy__Warm_Restarts_Parameters(
      class Model const *const model);
  __host__ __device__ void Copy__Gradient_Descent_Parameters(
      class Model const *const model);
  __host__ __device__ void Copy__QuickProp_Parameters(
      class Model const *const model);
  __host__ __device__ void Copy__RPROP_minus_Parameters(
      class Model const *const model);
  __host__ __device__ void Copy__RPROP_plus_Parameters(
      class Model const *const model);
  __host__ __device__ void Copy__SARProp_Parameters(
      class Model const *const model);
  __host__ __device__ void Copy__Adam_Parameters(
      class Model const *const model);
  __host__ __device__ void Copy__NosAdam_Parameters(
      class Model const *const model);
  __host__ void Copy__Dropout(class Model const *const model);
  __host__ void Copy__Normalization(class Model const *const model);
  __device__ void device__Copy_Dropout(
      var const *ptr_array_probability_retained_unit_received);
  __device__ void device__Copy__Normalization(
      DL::LAYER_NORM::TYPE const
          *ptr_array_normalization_by_layers_received);
  __host__ __device__ bool Set__Regularization__L1(
      var const regularization__l1_received);
  __host__ __device__ bool Set__Regularization__L2(
      var const regularization__l2_received);
  __host__ __device__ bool Set__Regularization__Weight_Decay(
      var const regularization__weight_decay_received);
  __host__ __device__ bool Set__Regularization__Max_Norm_Constraints(
      var const regularization__max_norm_constraints_received);
  __host__ __device__ bool Set__Normalization_Momentum_Average(
      var const momentum_average_received);
  __host__ __device__ bool Set__Normalization_Epsilon(
      var const Set__Normalization_Epsilon);
  __host__ __device__ bool Set__Batch_Renormalization_r_Correction_Maximum(
      var const r_correction_maximum_received);
  __host__ __device__ bool Set__Batch_Renormalization_d_Correction_Maximum(
      var const d_correction_maximum_received);
  __device__ bool Allouable__Batch_Size(
      size_t const batch_size, size_t const maximum_threads_received,
      size_t &ref_batch_size_allouable_received,
      size_t &ref_number_threads_allouable_received);
  __device__ bool Use__Regularization_Parameter(void) const;
  __host__ __device__ bool Deallocate(void);
  bool use_Dropout = false;
  bool use_warm_restarts = false;
  bool use_nesterov = false;
  bool use_normalized_weight_decay = true;
  bool use_adam_bias_correction = true;
  bool use_Batch_Renormalization = false;

  __device__ int Total_Blocks_cuRAND_MTGP32(
      enum ENUM_TYPE_CURAND_GENERATOR const type_curand_generator_received);
  __device__ size_t Get__Limit_Device_Runtime_Pending_Launch_Count(void);
  size_t *ptr_array_number_loss = nullptr;  // Size[N].
  size_t *ptr_array_reduce_number_loss =
      nullptr;  // Size[total reduce batch size].
  size_t *ptr_array_number_bit_fail = nullptr;  // Size[N].
  size_t *ptr_array_reduce_bit_fail_values =
      nullptr;  // Size[total reduce batch size].
  size_t limit_device_runtime_pending_launch_count = 0;
  size_t number_active_threads = 1;
  size_t number_threads = 1;
  size_t cache_number_threads = 1;
  size_t batch_size = 1;
  size_t cache_batch_size = 0;
  size_t n_acc_trial = 0;
  size_t n_inp = 0;
  size_t n_out = 0;
  size_t n_time_delay = 0;
  size_t seq_w = 0;
  size_t total_neuron_units = 0;
  size_t total_neuron_units_allocated = 0;
  size_t total_block_units = 0;
  size_t total_block_units_allocated = 0;
  size_t total_cell_units = 0;
  size_t total_cell_units_allocated = 0;
  size_t total_parameters = 0;
  size_t total_parameters_allocated = 0;
  size_t total_weights = 0;
  size_t total_weights_allocated = 0;
  size_t total_layers = 0;
  size_t total_reduce_batch_size = 0;
  size_t total_reduce_batch_DP_size = 0;
  size_t neurons_total_reduce_summation_size = 0;
  size_t neurons_total_reduce_error_size = 0;
  size_t neurons_total_reduce_batch_size = 0;
  size_t neurons_total_reduce_norms_size = 0;
  size_t *ptr_array_number_neurons_by_layer = nullptr;  // size[L].
  size_t *ptr_array_neuron_units_first_forward_connection_index =
      nullptr;  // size[H].
  size_t *ptr_array_neuron_units_last_forward_connection_index =
      nullptr;  // size[H].
  size_t *ptr_array_neuron_units_number_forward_connections =
      nullptr;                                                     // size[H].
  size_t *ptr_array_neuron_units_reduce_summation_size = nullptr;  // size[H].
  size_t *ptr_array_neuron_units_reduce_error_size = nullptr;      // size[H].
  size_t *ptr_array_neuron_units_reduce_batch_size = nullptr;      // size[H].
  size_t *ptr_array_neuron_units_reduce_norms_size = nullptr;      // size[H].
  size_t *ptr_array_neuroyed_number_neurons_in_layer = nullptr;    // size[H].
  size_t number_cuRAND_State_MTGP32_neuroyed = 0;
  size_t number_cuRAND_State_MTGP32_weighted = 0;

  __host__ __device__ var get_accu(
      DL::ENV::TYPE const env_type) const;
  __host__ __device__ var
  get_loss(DL::ENV::TYPE const env_type,
            size_t const number_digits_received = 9u) const;
  __device__ var get_me(void) const;
  __device__ var get_loss_l1(void) const;
  __device__ var get_mae(void) const;
  __device__ var get_loss_l2(void) const;
  __device__ var get_mse(void) const;
  __device__ var get_rmse(void) const;
  __device__ var get_mape(void) const;
  __device__ var get_smape(void) const;
  __device__ var get_mase(void) const;
  __device__ var get_ace(void) const;
  __device__ var get_bitfail(void) const;
  var *ptr_array_loss_values = nullptr;            // Size[N].
  var *ptr_array_accuracy_values[5] = {nullptr};  // Size[N].
  var *ptr_array_reduce_loss_values = nullptr;  // Size[total reduce batch size].
  var *ptr_array_reduce_accuracy_values[5] = {
      nullptr};  // Size[total reduce batch size].
  var loss_train = 1_r;
  var loss_valid = 1_r;
  var loss_testg = 1_r;
  var loss_rprop = 1_r;
  var loss_rprop_tm1 = 1_r;
  var acc_var = 0.49_r;
  var acc_train = 0_r;
  var acc_valid = 0_r;
  var acc_testg = 0_r;

  DL::MODEL::TYPE type =
      DL::MODEL::NONE;
  DL::OPTIMIZER::TYPE type_optimizer_function =
      DL::OPTIMIZER::NONE;
  DL::LOSS_FN::TYPE type_loss_function =
      DL::LOSS_FN::NONE;
  DL::ACCU_FN::TYPE type_accuracy_function =
      DL::ACCU_FN::
          DISTANCE;
  DL::PROPAGATION::TYPE type_state_propagation =
      DL::PROPAGATION::
          INFERENCE;  // Dropout variable
  DL::ACTIVATION::TYPE
      *ptr_array_neuron_units_type_activation_function = nullptr;

  void **ptr_array_ptr_connections;

  struct cuLayer *ptr_array_layers = nullptr;  // size[L].
  struct cuLayer *ptr_last_layer = nullptr;    // size[1].

  /* Grid | Block:
          [0]: Total threads
          [1]: Total parameters
          [2]: Total weights
          [3]: Total neurons
          [4]: (threads - 1) * total parameters
          [5]: Batch * total neurons
          [6]: Max norm constraints
          [7]: Total threads DP
          [8]: Total weights cuRAND MTGP32 */
  struct dim3 *ptr_array_dim3_grid = NULL;   // Size[TOTAL_KERNEL_PARALLEL].
  struct dim3 *ptr_array_dim3_block = NULL;  // Size[TOTAL_KERNEL_PARALLEL].
  struct dim3 *ptr_array_dim3_grid_reduce_threads =
      NULL;  // Size[total reduce batch size].
  struct dim3 *ptr_array_dim3_block_reduce_threads =
      NULL;  // Size[total reduce batch size].
  struct dim3 *ptr_array_dim3_grid_reduce_threads_DP =
      NULL;  // Size[total reduce batch size].
  struct dim3 *ptr_array_dim3_block_reduce_threads_DP =
      NULL;  // Size[total reduce batch size].
  // Grid | Block: Each layer have a dimensions of X neurons to it.
  struct dim3 *ptr_array_layers_dim3_grid_neurons = NULL;          // Size[L].
  struct dim3 *ptr_array_layers_dim3_block_neurons = NULL;         // Size[L].
  struct dim3 *ptr_array_layers_dim3_grid_neurons_DP = NULL;       // Size[L].
  struct dim3 *ptr_array_layers_dim3_block_neurons_DP = NULL;      // Size[L].
  struct dim3 *ptr_array_layers_dim3_grid_neurons_cuRAND = NULL;   // Size[L].
  struct dim3 *ptr_array_layers_dim3_block_neurons_cuRAND = NULL;  // Size[L].
  // Grid | Block: Each layer have a dimensions of X neurons times batch size to
  // it.
  struct dim3 *ptr_array_layers_dim3_grid_batch_neurons = NULL;   // Size[L].
  struct dim3 *ptr_array_layers_dim3_block_batch_neurons = NULL;  // Size[L].
  // Grid | Block: Each layer have a dimensions of X weights to it.
  struct dim3 *ptr_array_layers_dim3_grid_weights = NULL;   // Size[L].
  struct dim3 *ptr_array_layers_dim3_block_weights = NULL;  // Size[L].
  // Grid | Block: Each neuron have a dimensions of X connections to it.
  struct dim3 *ptr_array_neuron_units_dim3_grid_connections = NULL;  // Size[H].
  struct dim3 *ptr_array_neuron_units_dim3_block_connections =
      NULL;  // Size[H].
  struct dim3 *ptr_array_neuron_units_dim3_grid_reduce_summation =
      NULL;  // Size[neurons total reduce summation size].
  struct dim3 *ptr_array_neuron_units_dim3_block_reduce_summation =
      NULL;  // Size[neurons total reduce summation size].
  struct dim3 *ptr_array_neuron_units_dim3_grid_reduce_error =
      NULL;  // Size[neurons total reduce error size].
  struct dim3 *ptr_array_neuron_units_dim3_block_reduce_error =
      NULL;  // Size[neurons total reduce error size].
  struct dim3 *ptr_array_neuron_units_dim3_grid_reduce_batch =
      NULL;  // Size[neurons total reduce batch size].
  struct dim3 *ptr_array_neuron_units_dim3_block_reduce_batch =
      NULL;  // Size[neurons total reduce batch size].
  struct dim3 **ptr_array_2D_neurons_dim3_grid_reduce_norms = NULL;  // Size[H].
  struct dim3 **ptr_array_2D_neurons_dim3_block_reduce_norms =
      NULL;  // Size[H].

  class cuDims *ptr_Class_Storage_Dim3_Memcpy = nullptr;  // size[1].
  class cuDims *ptr_array_layers_Class_Storage_Dim3_Batch =
      nullptr;  // size[L].

  __device__ var const *get_out(size_t const thread_index_received) const;
  __device__ var const *get_out(
      size_t const thread_index_received,
      size_t const time_step_index_received) const;
  __device__ real warm_restarts_decay(void);
  __device__ real normalized_wd(size_t const batch_size,
                                        size_t const training_size);
  __host__ __device__ var Get__Regularization__Max_Norm_Constraints(void) const;
  __host__ __device__ var Get__Regularization__L1(void) const;
  __host__ __device__ var Get__Regularization__L2(void) const;
  var *ptr_array_neuron_units_summations = nullptr;            // size[N, T, H].
  var *ptr_array_neuron_units_values = nullptr;                // size[N, T, H].
  var *ptr_array_normalized_batch_units_values_hats =
      nullptr;  // size[N, T, H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_values_normalizes =
      nullptr;  // size[N, T, H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_scales =
      nullptr;  // size[H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_shifts =
      nullptr;  // size[H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_means =
      nullptr;  // size[N, ?, H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_variances =
      nullptr;  // size[N, ?, H]. Batch renormalization variable.
  var *ptr_array_neuron_units_transposed_mean =
      nullptr;  // size[N, ?, H]. Batch renormalization variable.
  var *ptr_array_neuron_units_transposed_variance =
      nullptr;  // size[N, ?, H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_derivatives_means =
      nullptr;  // size[N, ?, H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_derivatives_variances =
      nullptr;  // size[N, ?, H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_r_corrections =
      nullptr;  // size[H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_d_corrections =
      nullptr;  // size[H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_means_averages =
      nullptr;  // size[H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_variances_averages =
      nullptr;  // size[H]. Batch renormalization variable.
  var *ptr_array_neuron_units_errors = nullptr;  // size[N, T, H].
  var **ptr_array_2D_neurons_reduce_summation =
      nullptr;  // Size[H], Size[N, T, neurons total reduce summation size].
  var **ptr_array_2D_neurons_reduce_error =
      nullptr;  // Size[H], Size[N, T, neurons total reduce error size].
  var **ptr_array_2D_neurons_reduce_batch_mean =
      nullptr;  // Size[H], Size[neurons total reduce batch size].
  var **ptr_array_2D_neurons_reduce_batch_variance =
      nullptr;  // Size[H], Size[neurons total reduce batch size].
  var **ptr_array_2D_neurons_reduce_norms =
      nullptr;  // Size[H], Size[neurons total reduce norms size].
  var *ptr_array_transposed_weights = nullptr;                 // Size[W].
  var *ptr_array_parameters = nullptr;                         // Size[P].
  var *ptr_array_derivatives_parameters = nullptr;             // Size[N, P].
  var *ptr_array_mask_regularized_parameters = nullptr;        // Size[P].
  var *ptr_array_previous_steps = nullptr;                     // Size[P].
  var *ptr_array_previous_delta_parameters = nullptr;          // Size[P].
  var *ptr_array_previous_derivatives_parameters = nullptr;    // Size[P].
  var *ptr_array_previous_biased_first_moment = nullptr;       // Size[P].
  var *ptr_array_previous_biased_second_moment = nullptr;      // Size[P].
  var *ptr_array_previous_biased_second_moment_hat = nullptr;  // Size[P].
  var learning_rate = 0.9_r;
  var learning_momentum = 0_r;
  var bit_fail_limit = 1_r;
  var regularization__max_norm_constraints = 0_r;
  var regularization__l1 = 0_r;
  var regularization__l2 = 0_r;
  var weight_decay = 0_r;
  var adam_learning_rate = 0.001_r;
  var adam_beta1 = 0.9_r;
  var adam_beta2 = 0.999_r;  // {0.99, 0.999}
  var adam_previous_beta2 = 0.999_r;
  var adam_epsilon = 1e-8_r;
  var adam_gamma = 0.1_r;  // {0.05, 0.1}
  var optimizer_time_step = 0_r;
  var epoch_time_step = 1_r;
  var warm_restarts_decay_learning_rate = 1_r;
  var warm_restarts_initial_maximum_learning_rate = 1_r;
  var warm_restarts_maximum_learning_rate = 1_r;
  var warm_restarts_minimum_learning_rate = 1e-7_r;
  var warm_restarts_initial_T_i = 1_r;
  var warm_restarts_T_i = 1_r;
  var warm_restarts_multiplier = 2_r;
  var normalization_momentum_average = 0.999_r;
  var normalization_epsilon = 1e-5_r;
  var batch_renormalization_r_correction_maximum = 1_r;
  var batch_renormalization_d_correction_maximum = 0_r;

  // Dropout variable.
  bool *ptr_array_units_mask_dropout_bernoulli = nullptr;  // Size[H].
  bool *ptr_array_cell_units_mask_dropout_zoneout = nullptr;

  var *ptr_array_mask_dropout_parameters = nullptr;  // Size[P].
  // |END| Dropout variable. |END|

  float quickprop_decay;
  float quickprop_mu;

  float rprop_increase_factor;
  float rprop_decrease_factor;
  float rprop_delta_min;
  float rprop_delta_max;
  float rprop_delta_zero;

  float sarprop_weight_decay_shift;
  float sarprop_step_error_threshold_factor;
  float sarprop_step_error_shift;
  float sarprop_temperature;
  size_t sarprop_epoch;

  __device__ void Printf_Parameters(bool const full_description_received);

  __device__ class cuDevicesProp *Get__Class_Device_Information_Array(
      void) const;

  __host__ __device__ size_t Get__Maximum_Allowable_Memory(void) const;
  __host__ __device__ size_t Get__Sizeof(size_t number_threads_received = 0,
                                         size_t batch_size = 0u) const;
  __host__ __device__ size_t
  Get__Batch_Sizeof(size_t batch_size = 0u) const;
  __host__ __device__ size_t
  Get__Threads_Sizeof(size_t number_threads_received = 0u) const;
  size_t maximum_allowable_memory_bytes = 0u;  // Bytes.

 private:
  __device__ bool Allocate__Neuron(struct cuNeuron *ptr_neuron_received);

  struct curandStateMtgp32 *ptr_array_cuRAND_State_MTGP32_weighted =
      nullptr;  // Size[number_cuRAND_State_MTGP32_weighted], MTGP32.
  struct curandStateMtgp32 *ptr_array_cuRAND_State_MTGP32_neuroyed =
      nullptr;  // Size[number_cuRAND_State_MTGP32_neuroyed], MTGP32.

  class cuDevicesProp *_ptr_Class_Device_Information_Array = nullptr;  // Ptr.
};

__device__ void Activation_Real(
    var &ref_value_received, var const summation_received,
    DL::ACTIVATION::TYPE const
        type_activation_function_received);

__device__ bool cuRAND_Bernoulli(float const probability_received,
                                 float const curand_uniform_received);

__device__ void Update_Accuracy(var const error_received,
                                var const bit_fail_limit,
                                float *const ptr_accuracy_value_received);
__device__ void Update_Accuracy__atomic(
    var const error_received, var const bit_fail_limit,
    float *const ptr_accuracy_value_received);

// Standard, update error.
__device__ void update_loss(var const observed_output_received,
                             var const desired_output_received,
                             var const error_received,
                             float *const ptr_loss_values_received,
                             DL::LOSS_FN::TYPE const
                                 type_loss_function_received);
__device__ void Update_Error__atomic(
    var const observed_output_received, var const desired_output_received,
    var const error_received, float *const ptr_loss_values_received,
    DL::LOSS_FN::TYPE const
        type_loss_function_received);

// Binary cross entropy, update error.
__device__ void Update_Error__Binary_Cross_Entropy(
    var const observed_output_received, var const desired_output_received,
    float *const ptr_loss_values_received);
__device__ void Update_Error__Binary_Cross_Entropy__atomic(
    var const observed_output_received, var const desired_output_received,
    float *const ptr_loss_values_received);

// Bit fail, update error.
__device__ void Update_Error__Bit_Fail(
    var const error_received, var const bit_fail_limit,
    size_t *const ptr_bit_fail_values_received);
__device__ void Update_Error__Bit_Fail__atomic(
    var const error_received, var const bit_fail_limit,
    size_t *const ptr_bit_fail_values_received);

__device__ var
Activation_Derived(var const summation_received, var const value_received,
                   DL::ACTIVATION::TYPE const
                       activation_function_received);

__device__ var
Activation_Derived(var const summation_received, var const value_received,
                   DL::ACTIVATION::TYPE const
                       activation_function_received,
                   DL::LOSS_FN::TYPE const
                       type_loss_function_received);