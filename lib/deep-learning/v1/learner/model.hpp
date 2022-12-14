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

// Framework:
#include "framework.hpp"

// Deep learning:
#include "deep-learning/v1/data/enum/group.hpp"
#include "deep-learning/data/dtypes.hpp"
#include "deep-learning/data/enum/env.hpp"
#include "deep-learning/v1/data/enum/model.hpp"
#include "deep-learning/v1/data/enum/layer.hpp"
#include "deep-learning/v1/data/enum/hyperopt.hpp"
#include "deep-learning/v1/data/enum/layer_activation.hpp"
#include "deep-learning/v1/data/enum/layer_dropout.hpp"
#include "deep-learning/v1/data/enum/layer_norm.hpp"
#include "deep-learning/v1/data/enum/loss_fn.hpp"
#include "deep-learning/v1/data/enum/optimizer.hpp"
#include "deep-learning/v1/data/enum/activation.hpp"
#include "deep-learning/v1/data/enum/initializer.hpp"
#include "deep-learning/v1/data/enum/propagation.hpp"
#include "deep-learning/ops/distributions/bernoulli.hpp"
#include "deep-learning/ops/distributions/real.hpp"
#include "deep-learning/ops/distributions/gaussian.hpp"
#include "deep-learning/ops/distributions/integer.hpp"

#ifdef COMPILE_CUDA
#include "deep-learning/v1/learner/model.cuh"
//#include "deep-learning/v1/data/datasets.cuh"
#endif

#include <stdio.h>
#include <functional>

//#define NO_PEEPHOLE

namespace DL::v1 {
// Forward declaration.
class Model;

#ifdef COMPILE_CUDA
class cuDatasets;
#endif

class DatasetV1;
class Datasets;
// |END| Forward declaration. |END|

template <typename T>
void euclidean_norm_st(size_t const str, size_t const end, real const max_norm,
                       T *const parameters);
template <typename T>
void euclidean_norm_mp(size_t const str, size_t const end, real const max_norm,
                       T *const parameters);

struct Layer_Parameters {
  Layer_Parameters(void) {}
  ~Layer_Parameters(void) {}

  bool use_bidirectional = false;

  LAYER::TYPE type_layer = LAYER::NONE;

  /* [0]:
      FC: Number of neurons.
      LSTM: Number of blocks.
      RESIDUAL: Block depth.
      POOLING: Kernel size.
     [1]:
      LSTM: Number of cells.
      POOLING: Stride.
     [2]:
      POOLING: Padding.
     [3]:
      POOLING: Dilation.
     [4]:
      POOLING: Ceil mode.*/
  size_t unit_parameters[5] = {0};
};

struct Neural_Network_Initializer {
  Neural_Network_Initializer(void) {}
  ~Neural_Network_Initializer(void);

  bool Input_Initialize(void);
  bool Template_Initialize(void);
  bool Build__Layer__FC(Layer_Parameters &ref_Layer_Parameters_received);
  bool Build__Layer__Pooling(Layer_Parameters &ref_Layer_Parameters_received);
  bool Build__Layer__LSTM(Layer_Parameters &ref_Layer_Parameters_received);
  bool Build__Layer__Residual(void);
  bool While__Push_Back__Layer(void);

  Model *Output_Initialize(size_t const allowable_memory = 32_UZ * KILOBYTE *
                                                           KILOBYTE) const;

  size_t seq_w = 0_UZ;
  size_t n_time_delay = 0_UZ;

  std::vector<Layer_Parameters> vector_layers_parameters;

  MODEL::TYPE type_neural_network = MODEL::NONE;
};

struct Activation_Function_Initializer {
  Activation_Function_Initializer(void) {}
  ~Activation_Function_Initializer(void);

  void Deallocate_Layers_Activation_Function(void);

  bool Allocate__Layers_Activation_Function(
      size_t const n_layers);
  bool Input_Initialize(size_t const n_layers,
                        MODEL::TYPE const type_network_received);
  bool Output_Initialize(Model *const ptr_Neural_Network_received) const;

  size_t number_layers = 0u;

  ACTIVATION::TYPE *ptr_array_type_layers_activation_function = nullptr;
};

struct Loss_Function_Initializer {
  Loss_Function_Initializer(void) {}
  ~Loss_Function_Initializer(void) {}

  void Output_Initialize(Model *const ptr_Neural_Network_received) const;

  bool Input_Initialize(void);

  // Bit.
  real bit_fail_limit = 0.5_r;

  LOSS_FN::TYPE type_loss_function = LOSS_FN::MSE;
};

struct Accuracy_Function_Initializer {
  Accuracy_Function_Initializer(void) {}
  ~Accuracy_Function_Initializer(void) {}

  void Output_Initialize(Model *const ptr_Neural_Network_received) const;

  bool Input_Initialize(void);

  ACCU_FN::TYPE type_accuracy_function = ACCU_FN::DISTANCE;
};

struct Optimizer_Function_Initializer {
  Optimizer_Function_Initializer(void) {}
  ~Optimizer_Function_Initializer(void) {}

  bool Input_Initialize(void);
  bool Output_Initialize(Model *const ptr_Neural_Network_received) const;

  /* values
      [0]:
          GD, AdaBound, AMSBound, Adam, AMSGrad, NosAdam:
              learning_rate=0.01, "learning rate."
          iRPROP+:
              rprop_increase_factor=1.2
      [1]:
          AdaBound, AMSBound:
              learning_rate_final=0.1, "Final (SGD) learning rate."
          Adam, AMSGrad, NosAdam:
              beta1=0.9, "Coefficients used for computing running averages of
     gradient." GD: learning_momentum=0.9. iRPROP+: rprop_decrease_factor=0.5
      [2]:
          AdaBound, AMSBound:
              beta1=0.9, "Coefficients used for computing running averages of
     gradient." Adam, AMSGrad, NosAdam: beta2=0.999, "Coefficients used for
     computing running averages of square gradient." GD: use_nesterov=1 iRPROP+:
              rprop_delta_max=50
      [3]:
          AdaBound, AMSBound:
              beta2=0.999, "Coefficients used for computing running averages of
     square gradient." Adam, AMSGrad, NosAdam: epsilon=1e-8, "Term added to the
     denominator to improve numerical stability." iRPROP+: rprop_delta_min=1e-6
      [4]:
          AdaBound, AMSBound:
              epsilon=1e-8, "Term added to the denominator to improve numerical
     stability." Adam, AMSGrad, NosAdam: bias_correction=true, "Moving average
     to estimate the first and second moments." iRPROP+: rprop_delta_zero=0.1
      [5]:
          AdaBound, AMSBound:
              bias_correction=true, "Moving average to estimate the first and
     second moments." NosAdam: gamma=0.1, "Hyperharmonic." [6]: AdaBound,
     AMSBound: learning_gamma=1e-3, "Convergence speed of the bound functions."
  */
  real values[7] = {0};

  real weight_decay = 0_r;

  OPTIMIZER::TYPE type_optimizer_function = OPTIMIZER::GD;
};

struct Warm_Restarts_Initializer {
  Warm_Restarts_Initializer(void) {}
  ~Warm_Restarts_Initializer(void) {}

  void Input_Initialize(void);

  bool Output_Initialize(Model *const ptr_Neural_Network_received) const;
  bool use_warm_restarts = false;

  real warm_restarts_decay_learning_rate = 1_r;
  real warm_restarts_maximum_learning_rate = 1_r;
  real warm_restarts_minimum_learning_rate = 1e-7_r;
  real warm_restarts_initial_T_i = 1_r;
  real warm_restarts_multiplier = 2_r;
};

struct LSUV_Parameters {
  LSUV_Parameters(void) {}
  ~LSUV_Parameters(void) {}

  real initial_bias = 0_r;
  real epsilon = 1e-7_r;
  real variance_target = 1_r;
  real variance_tolerance = 0.01_r;

  size_t maximum_number_trials = 10_UZ;
  size_t maximum_batch_size = 32_UZ;
};

struct Weights_Initializer {
  Weights_Initializer(void) {}
  ~Weights_Initializer(void) {}

  bool Input_Initialize(void);
  bool Output_Initialize(Model *const ptr_Neural_Network_received) const;

  real initial_bias = 0_r;

  /* values
      [0]:
          Uniform:
              lower_bound=-1
          LSUV:
              maximum_number_trials=10
      [1]:
          Uniform:
              upper_bound=1
          LSUV:
              maximum_batch_size=32
      [2]:
          LSUV:
              variance_target=1
      [3]:
          LSUV:
              variance_tolerance=1
  */
  real values[4] = {0};

  INITIALIZER::TYPE type_weights_initializer = INITIALIZER::GLOROT_GAUSSIAN;
};

struct Dropout_Initializer {
  Dropout_Initializer(void) {}
  ~Dropout_Initializer(void);

  void Deallocate__Layers_Using_Dropout(void);

  bool Allocate__Layers_Using_Dropout(size_t const n_layers);
  bool Input_Initialize(size_t const n_layers,
                        MODEL::TYPE const type_network_received);
  bool Output_Initialize(Model *const ptr_Neural_Network_received) const;

  size_t number_layers = 0u;

  bool *ptr_array_layers_use_coded_dropout = nullptr;

  real **ptr_array_layers_dropout_array_values = nullptr;

  LAYER_DROPOUT::TYPE *ptr_array_layers_type_dropout = nullptr;
};

struct Normalization_Initializer {
  Normalization_Initializer(void) {}
  ~Normalization_Initializer(void);

  void Deallocate__Layers_Using_Normalization(void);

  bool Allocate__Layers_Using_Normalization(
      size_t const n_layers);
  bool Input_Initialize(size_t const n_layers,
                        size_t const number_batch_received,
                        MODEL::TYPE const type_network_received);
  bool Output_Initialize(Model *const ptr_Neural_Network_received) const;
  bool *ptr_array_layers_normalization_before_activation = nullptr;

  size_t number_layers = 0u;

  LAYER_NORM::TYPE *ptr_array_layers_using_normalization = nullptr;

  real normalization_momentum_average = 0.01_r;  // 1 / number of mini-batch
  real normalization_epsilon = 1e-5_r;

  // Batch renormalization parameter.
  real batch_renormalization_r_correction_maximum = 1_r;
  real batch_renormalization_d_correction_maximum = 0_r;
  // |END| Batch renormalization parameter. |END|
};

struct Normalized_batch_unit {
  // N: Number of threads.
  // B: Batch size.
  // T: Number of times to predict.
  // P: Number of parameters.

  var *ptr_array_values_hats = nullptr;            // size[B, T].
  var *ptr_array_values_normalizes = nullptr;      // size[B, T].
  var *ptr_scale = nullptr;                        // size[1].
  var *ptr_shift = nullptr;                        // size[1].
  real *ptr_array_derivatives_scales = nullptr;     // size[N].
  real *ptr_array_derivatives_shifts = nullptr;     // size[N].
  var *ptr_array_means = nullptr;                  // size[N * T].
  var *ptr_array_variances = nullptr;              // size[N * T].
  real *ptr_array_derivatives_means = nullptr;      // size[N * T].
  real *ptr_array_derivatives_variances = nullptr;  // size[N * T].
  var *ptr_r_correction = nullptr;                 // size[T].
  var *ptr_d_correction = nullptr;                 // size[T].
  var *ptr_mean_average = nullptr;                 // size[T].
  var *ptr_variance_average = nullptr;             // size[T].
  real *ptr_array_errors = nullptr;                 // size[B, T].
};

struct Normalized_streaming_unit {
  // N: Number of threads.
  // B: Batch size.
  // T: Number of times to predict.
  // P: Number of parameters.

  var *ptr_array_values_hats = nullptr;            // size[B, T].
  var *ptr_array_values_normalizes = nullptr;      // size[B, T].
  var *ptr_scale = nullptr;                        // size[1].
  var *ptr_shift = nullptr;                        // size[1].
  real *ptr_array_derivatives_scales = nullptr;     // size[N].
  real *ptr_array_derivatives_shifts = nullptr;     // size[N].
  var *ptr_array_means = nullptr;                  // size[N * T].
  var *ptr_array_variances = nullptr;              // size[N * T].
  real *ptr_array_derivatives_means = nullptr;      // size[N * T].
  real *ptr_array_derivatives_variances = nullptr;  // size[N * T].
  var *ptr_r_correction = nullptr;                 // size[T].
  var *ptr_d_correction = nullptr;                 // size[T].
  var *ptr_mean_average = nullptr;                 // size[T].
  var *ptr_variance_average = nullptr;             // size[T].
  real *ptr_array_errors = nullptr;                 // size[B, T].
};

union Normalized_unit {
  Normalized_unit(void){};

  Normalized_batch_unit normalized_batch_units;

  Normalized_streaming_unit normalized_streaming_units;
};

struct Neuron_Ind {
  // N: Number of threads.
  // B: Batch size.
  // T: Number of times to predict.
  // P: Number of parameters.

  Neuron_Ind(void) {}
  ~Neuron_Ind(void) {}
};

struct Neuron_unit {
  // N: Number of threads.
  // B: Batch size.
  // T: Number of times to predict.
  // P: Number of parameters.

  Neuron_unit(void) {}
  ~Neuron_unit(void) {}

  size_t *ptr_first_connection_index = nullptr;  // size[1].
  size_t *ptr_last_connection_index = nullptr;   // size[1].
  size_t *ptr_number_connections = nullptr;      // size[1].

  var *ptr_array_summations = nullptr;  // size[B, T].
  real *ptr_array_errors = nullptr;      // size[B, T].
};

struct AF_unit {
  AF_unit(void) {}
  ~AF_unit(void) {}

  var *ptr_array_values = nullptr;          // size[B, T].
  real *ptr_array_errors = nullptr;          // size[B, T].

  ACTIVATION::TYPE *ptr_type_activation_function = nullptr;  // size[1].
};

struct AF_Ind_recurrent_unit {
  AF_Ind_recurrent_unit(void) {}
  ~AF_Ind_recurrent_unit(void) {}

  size_t *ptr_recurrent_connection_index = nullptr;  // size[1].

  var *ptr_array_pre_AFs = nullptr;         // size[B, T].
  var *ptr_array_AFs = nullptr;             // size[B, T].
  real *ptr_array_errors = nullptr;          // size[B, T].
  real *ptr_array_dAFs = nullptr;            // size[B, T].

  ACTIVATION::TYPE *ptr_type_activation_function = nullptr;  // size[1].
};

struct Basic_unit {
  // B: Batch size.
  // T: Number of times to predict.

  Basic_unit(void) {}
  ~Basic_unit(void) {}

  var *ptr_array_values = nullptr;  // size[B, T].
  real *ptr_array_errors = nullptr;  // size[B, T].
};

struct Basic_indice_unit {
  // B: Batch size.
  // T: Number of times to predict.

  Basic_indice_unit(void) {}
  ~Basic_indice_unit(void) {}

  size_t *ptr_array_indices = nullptr;  // size[B, T].

  var *ptr_array_values = nullptr;  // size[B, T].
  real *ptr_array_errors = nullptr;  // size[B, T].
};

struct CellUnit {
  CellUnit(void) {}
  ~CellUnit(void) {}

  bool *ptr_mask_dropout_zoneout_state = nullptr;
  bool *ptr_mask_dropout_zoneout_output = nullptr;

  size_t first_index_feedforward_connection_cell_input = 0_UZ;
  size_t last_index_feedforward_connection_cell_input = 0_UZ;
  size_t first_index_recurrent_connection_cell_input = 0_UZ;
  size_t last_index_recurrent_connection_cell_input = 0_UZ;
#ifndef NO_PEEPHOLE
  size_t index_peephole_input_gate = 0_UZ;
  size_t index_peephole_forget_gate = 0_UZ;
  size_t index_peephole_output_gate = 0_UZ;
#endif

  var *ptr_summation_cell_input = nullptr;
  var *ptr_summation_input_cell_input = nullptr;
  var *ptr_summation_recurrent_cell_input = nullptr;
  var *ptr_cell_input = nullptr;
  var *ptr_cell_state = nullptr;
  var *ptr_cell_state_activate = nullptr;
  var *ptr_cell_output = nullptr;
  real *ptr_delta_cell_input = nullptr;
  real *ptr_delta_cell_input_input = nullptr;
  real *ptr_delta_cell_recurrent_input = nullptr;
  real *ptr_delta_cell_state = nullptr;
  real *ptr_delta_cell_output = nullptr;

  // Normalized unit.
  union Normalized_unit *ptr_array_normalized_units = nullptr;  // size[3].
  union Normalized_unit *ptr_last_normalized_unit = nullptr;    // size[1].
  // |END| Normalized unit. |END|
};

struct BlockUnit {
  BlockUnit(void) {}
  ~BlockUnit(void) {}

  bool *ptr_array_mask_dropout_zoneout = nullptr;

  size_t first_index_connection = 0_UZ;
  size_t last_index_connection = 0_UZ;
  size_t first_index_feedforward_connection_input_gate = 0_UZ;
  size_t last_index_feedforward_connection_input_gate = 0_UZ;
  size_t first_index_feedforward_connection_forget_gate = 0_UZ;
  size_t last_index_feedforward_connection_forget_gate = 0_UZ;
  size_t first_index_feedforward_connection_output_gate = 0_UZ;
  size_t last_index_feedforward_connection_output_gate = 0_UZ;
  size_t first_index_recurrent_connection_input_gate = 0_UZ;
  size_t last_index_recurrent_connection_input_gate = 0_UZ;
  size_t first_index_recurrent_connection_forget_gate = 0_UZ;
  size_t last_index_recurrent_connection_forget_gate = 0_UZ;
  size_t first_index_recurrent_connection_output_gate = 0_UZ;
  size_t last_index_recurrent_connection_output_gate = 0_UZ;
#ifndef NO_PEEPHOLE
  size_t first_index_peephole_input_gate = 0_UZ;
  size_t last_index_peephole_input_gate = 0_UZ;
  size_t first_index_peephole_forget_gate = 0_UZ;
  size_t last_index_peephole_forget_gate = 0_UZ;
  size_t first_index_peephole_output_gate = 0_UZ;
  size_t last_index_peephole_output_gate = 0_UZ;
#endif

  var *ptr_array_summation_cells_inputs = nullptr;
  var *ptr_array_summation_input_cells_inputs = nullptr;
  var *ptr_array_summation_recurrent_cells_inputs = nullptr;
  var *ptr_summation_inputs_gates = nullptr;
  var *ptr_summation_input_inputs_gates = nullptr;
  var *ptr_summation_recurrent_inputs_gates = nullptr;
  var *ptr_summation_forgets_gates = nullptr;
  var *ptr_summation_input_forgets_gates = nullptr;
  var *ptr_summation_recurrent_forgets_gates = nullptr;
  var *ptr_summation_outputs_gates = nullptr;
  var *ptr_summation_input_outputs_gates = nullptr;
  var *ptr_summation_recurrent_outputs_gates = nullptr;
  var *ptr_array_cells_inputs = nullptr;
  var *ptr_array_cells_states = nullptr;
  var *ptr_array_cells_states_activates = nullptr;
  var *ptr_array_cells_outputs = nullptr;
  var *ptr_inputs_gates = nullptr;
  var *ptr_forgets_gates = nullptr;
  var *ptr_outputs_gates = nullptr;
  real *ptr_array_delta_cells_inputs = nullptr;
  real *ptr_array_delta_cells_input_inputs = nullptr;
  real *ptr_array_delta_cells_recurrent_inputs = nullptr;
  real *ptr_array_delta_cells_states = nullptr;
  real *ptr_array_delta_cells_outputs = nullptr;
  real *ptr_delta_inputs_gates = nullptr;
  real *ptr_delta_input_inputs_gates = nullptr;
  real *ptr_delta_recurrent_inputs_gates = nullptr;
  real *ptr_delta_forgets_gates = nullptr;
  real *ptr_delta_input_forgets_gates = nullptr;
  real *ptr_delta_recurrent_forgets_gates = nullptr;
  real *ptr_delta_outputs_gates = nullptr;
  real *ptr_delta_input_outputs_gates = nullptr;
  real *ptr_delta_recurrent_outputs_gates = nullptr;

  CellUnit *ptr_array_cell_units = nullptr;
  CellUnit *ptr_last_cell_unit = nullptr;

  ACTIVATION::TYPE activation_function_gate = ACTIVATION::SIGMOID;
  ACTIVATION::TYPE activation_function_io = ACTIVATION::TANH;

  // Normalized unit.
  union Normalized_unit *ptr_array_normalized_units = nullptr;  // size[6].
  union Normalized_unit *ptr_last_normalized_unit = nullptr;    // size[1].
  // |END| Normalized unit. |END|
};

struct Bidirectional_Layer;  // Forward declaration.

struct Layer {
  Layer(void) {}
  ~Layer(void) {}

  // N: Number of threads.
  // B: Batch size.
  // T: Number of times to predict.
  // H: Number of neurons in layer.
  // R: Number of renormalizations units in layer.
  // K: Number of blocks in layer.
  // C: Number of cells in layer.

  bool use_bidirectional = false;
  bool use_tied_parameter = false;
  bool use_coded_dropout = false;
  bool use_layer_normalization_before_activation = true;
  bool *ptr_array__mask__dropout__bernoulli = nullptr;  // size[H].
  bool *ptr_array__mask__dropout__shakedrop = nullptr;  // size[T].
  bool Use__Bidirectional(void) const { return (this->use_bidirectional); }
  bool Use__Tied_Parameter(void) const;
  bool Use__Coded_Dropout(void) const;
  bool Use__K_Sparsity(void) const;
  bool Use__Regularization__Constraint_Recurrent_Weight(void) const;
  bool Compare__Dimensions(Layer const &ref_source_Layer_received) const;

  size_t get_n_out(void) const;
  size_t Get__First_Connection_Index(void) const;
  size_t Get__Last_Connection_Index(void) const;
  size_t Get__K_Sparsity(void) const;
  size_t *ptr_number_outputs = nullptr;          // size[1].
  size_t *ptr_first_connection_index = nullptr;  // size[1].
  size_t *ptr_last_connection_index = nullptr;   // size[1].
  size_t first_bias_connection_index = 0_UZ;     // size[1].
  size_t last_bias_connection_index = 0_UZ;      // size[1].
  size_t block_depth = 0_UZ;
  size_t k_sparsity = 0_UZ;

  /* pooling_values:
          [0]: Kernel size.
          [1]: Stride.
          [2]: Padding.
          [3]: Dilation.
          [4]: Ceil mode. */
  size_t pooling_values[5] = {0};

  std::pair<size_t, var> *ptr_array_k_sparse_activities = nullptr;

  LAYER::TYPE type_layer = LAYER::NONE;
  GROUP::TYPE type_group = GROUP::NONE;
  LAYER_ACTIVATION::TYPE type_activation = LAYER_ACTIVATION::NONE;
  LAYER_DROPOUT::TYPE type_dropout = LAYER_DROPOUT::NONE;
  LAYER_NORM::TYPE type_normalization = LAYER_NORM::NONE;

  // Basic unit variable.
  Basic_unit *ptr_array_basic_units = nullptr;  // size[H].
  Basic_unit *ptr_last_basic_unit = nullptr;    // size[1].
  // |END| Basic unit variable. |END|

  // Basic indice unit variable.
  Basic_indice_unit *ptr_array_basic_indice_units = nullptr;  // size[H].
  Basic_indice_unit *ptr_last_basic_indice_unit = nullptr;    // size[1].
  // |END| Basic indice unit variable. |END|

  // FC layer variable.
  var *ptr_array_pre_summations = nullptr;        // size[1].
  Neuron_unit *ptr_array_neuron_units = nullptr;  // size[H].
  Neuron_unit *ptr_last_neuron_unit = nullptr;    // size[1].
  // |END| FC layer variable. |END|

  // AF unit(s) variable.
  var *ptr_array_pre_activation_functions = nullptr;  // size[1].
  AF_unit *ptr_array_AF_units = nullptr;              // size[H].
  AF_unit *ptr_last_AF_unit = nullptr;                // size[1].
  // |END| AF unit(s) variable. |END|

  // AF unit(s) variable.
  AF_Ind_recurrent_unit *ptr_array_AF_Ind_recurrent_units =
      nullptr;                                                      // size[H].
  AF_Ind_recurrent_unit *ptr_last_AF_Ind_recurrent_unit = nullptr;  // size[1].
  // |END| AF unit(s) variable. |END|

  // LSTM layer variable.
  BlockUnit *ptr_array_block_units = nullptr;  // size[K].
  BlockUnit *ptr_last_block_unit = nullptr;    // size[1].

  CellUnit *ptr_array_cell_units = nullptr;  // size[C].
  CellUnit *ptr_last_cell_unit = nullptr;    // size[1].
  // |END| LSTM layer variable. |END|

  // Bidirectional layer variable.
  Bidirectional_Layer *ptr_Bidirectional_Layer = nullptr;  // size[1].
  // |END| Bidirectional layer variable. |END|

  // Normalized unit.
  var *ptr_array_pre_normalization = nullptr;  // size[1].
  union Normalized_unit *ptr_array_normalized_units =
      nullptr;  // size[H || (H * 4 + C)].
  union Normalized_unit *ptr_last_normalized_unit = nullptr;  // size[1].
  // |END| Normalized unit. |END|

  // Layer(s) connections.
  std::vector<Layer const *> previous_connected_layers;
  std::vector<Layer const *> next_connected_layers;
  // |END| Layer(s) connections |END|

  /* dropout_values:
      Bernoulli:
          [0]: Keep probability.
      Uout:
          [0]: Dropout probability.
      Zoneout:
          [0]: Cell zoneout probability.
          [1]: Hidden zoneout probability.
      Alpha:
          [0]: Dropout probability.
          [1]: a.
          [2]: b. */
  real dropout_values[3] = {0};

  var const *Get__Array_Summations__Cell__Block_Input__Input__Activation(
      void) const;
  var const *Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(
      void) const;
  var const *Get__Array_Summations__Cell__Cell_State__Activation(void) const;
  var const *Get__Array_Summations__Block__Input_Gate__Input__Activation(
      void) const;
  var const *Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(
      void) const;
  var const *Get__Array_Summations__Block__Forget_Gate__Input__Activation(
      void) const;
  var const *Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(
      void) const;
  var const *Get__Array_Summations__Block__Output_Gate__Input__Activation(
      void) const;
  var const *Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(
      void) const;
  real const *Get__Array_Deltas__Cell__Block_Input__Input(void) const;
  real const *Get__Array_Deltas__Cell__Block_Input__Recurrent(void) const;
  real const *Get__Array_Deltas__Block__Input_Gate__Input(void) const;
  real const *Get__Array_Deltas__Block__Input_Gate__Recurrent(void) const;
  real const *Get__Array_Deltas__Block__Forget_Gate__Input(void) const;
  real const *Get__Array_Deltas__Block__Forget_Gate__Recurrent(void) const;
  real const *Get__Array_Deltas__Block__Output_Gate__Input(void) const;
  real const *Get__Array_Deltas__Block__Output_Gate__Recurrent(void) const;
  real *ptr_array_derivative_outputs = nullptr;
  var *ptr_array_outputs = nullptr;
  real constraint_recurrent_weight_lower_bound = 0_r;
  real constraint_recurrent_weight_upper_bound = 0_r;
  real alpha_sparsity = 1_r;

  bool Use__Bias(void) const { return (this->Use__Normalization() == false); }
  bool Use__Dropout(void) const {
    return (this->type_dropout != LAYER_DROPOUT::NONE);
  }
  bool Use__Dropout__Alpha(void) const {
    return (this->type_dropout == LAYER_DROPOUT::ALPHA);
  }
  bool Use__Dropout__Bernoulli(void) const {
    return (this->type_dropout == LAYER_DROPOUT::BERNOULLI);
  }
  bool Use__Dropout__Bernoulli__Inverted(void) const {
    return (this->type_dropout == LAYER_DROPOUT::BERNOULLI_INVERTED);
  }
  bool Use__Dropout__Gaussian(void) const {
    return (this->type_dropout == LAYER_DROPOUT::GAUSSIAN);
  }
  bool Use__Dropout__ShakeDrop(void) const {
    return (this->type_dropout == LAYER_DROPOUT::SHAKEDROP);
  }
  bool Use__Dropout__Uout(void) const {
    return (this->type_dropout == LAYER_DROPOUT::UOUT);
  }
  bool Use__Dropout__Zoneout(void) const {
    return (this->type_dropout == LAYER_DROPOUT::ZONEOUT);
  }
  bool Use__Normalization(void) const {
    return (this->type_normalization != LAYER_NORM::NONE);
  }
  bool Use__Batch_Normalization(void) const {
    return (this->type_normalization == LAYER_NORM::BATCH_NORMALIZATION);
  }
  bool Use__Batch_Renormalization(void) const {
    return (this->type_normalization == LAYER_NORM::BATCH_RENORMALIZATION);
  }
};

struct Bidirectional_Layer {
  Bidirectional_Layer(void) {}
  ~Bidirectional_Layer(void) {}

  Layer forward_layer;
  Layer backward_layer;
};

class Model {
 public:
  // N: Number of threads.
  // B: Batch size.
  // T: Number of times to predict.
  // L: Number of layers.
  // H: Number of neurons.
  // K: Number of blocks.
  // C: Number of cells.
  // P: Number of parameters.
  // W: Number of weights.

  Model(void) {}
  ~Model(void);

  Model &operator=(Model const &ref_source_Neural_Network_received);

  bool operator==(Model const &ref_source_Neural_Network_received);
  bool operator!=(Model const &ref_source_Neural_Network_received);

#if DEEPLEARNING_USE_ADEPT
  void compute_grad_adept(size_t const batch_size, real const *const *const Ym);
  void compute_grad_adept_fwp_st(size_t const batch_size,
                                 real const *const *const Ym);
  void compute_grad_adept_rec_st(size_t const batch_size,
                                 real const *const *const Ym);
  void compute_grad_adept_pre_train(size_t const batch_size);
  void compute_grad_adept_pre_train_fwp_st(size_t const batch_size);
  void compute_grad_adept_pre_train_rec_st(size_t const batch_size);
  void update_derivatives_adept(void);
#endif

  void Initialize__OpenMP(void);
  template <class U>
  void Initialize_Connections__FC(
      Layer *const layer_it, U *const ptr_previous_layer_array_units_received);
  template <class U>
  void Initialize_Connections__LSTM(
      Layer *const layer_it, U *const ptr_previous_layer_array_units_received);
  void Initialize_Connections__AF_Ind_Recurrent(Layer *const layer_it);
  void Initialize_Connections__Bias(Layer *const layer_it);
  void Initialize_Connections__LSTM__Bias(Layer *const layer_it);
  void Initialize_Connections__FC_to_FC(
      Layer *const layer_it, Layer const *const ptr_previous_layer_it_received);
  void Initialize_Connections__FC_to_LSTM(
      Layer *const layer_it, Layer const *const ptr_previous_layer_it_received);
  void Initialize_Connections__LSTM_to_FC(
      Layer *const layer_it, Layer const *const ptr_previous_layer_it_received);
  void Initialize_Connections__LSTM_to_LSTM(
      Layer *const layer_it, Layer const *const ptr_previous_layer_it_received);
  void Initialize_Connections__Basic_unit_to_FC(
      Layer *const layer_it, Layer const *const ptr_previous_layer_it_received);
  void Initialize_Connections__Basic_unit_to_LSTM(
      Layer *const layer_it, Layer const *const ptr_previous_layer_it_received);
  void Initialize_Connections__Basic_indice_unit_to_FC(
      Layer *const layer_it, Layer const *const ptr_previous_layer_it_received);
  void Initialize_Connections__Basic_indice_unit_to_LSTM(
      Layer *const layer_it, Layer const *const ptr_previous_layer_it_received);
  void layer_initialize_const_bias(real const bias,
                                  Layer const *const layer_it);
  void lstm_initialize_const_bias(real const bias,
                                        Layer const *const layer_it);
  void weights_initialize_uniform(var *ptr_array_weights_received,
                           var const *const ptr_last_weight_received,
                           real const lower_bound,
                           real const upper_bound);
  void lstm_initialize_uniform(real const lower_bound[5],
                                 real const upper_bound[5],
                                 Layer const *const layer_it);
  void indrec_initialize_uniform(Layer const *const layer_it);
  void indrec_initialize_uniform_ltm(void);
  void weights_initialize_gaussian(var *ptr_array_weights_received,
                            var const *const ptr_last_weight_received,
                            real const variance);
  void weights_initialize_identity(size_t const rows,
                            size_t const cols,
                            var *const weights);
  void weights_initialize_orthogonal(size_t const rows,
                              size_t const cols,
                                     real const scale, var *weights);
  void lstm_initialize_gaussian(real const fwp_cell_var,
                                real const fwp_gate_var,
                                real const rec_cell_var,
                                real const rec_gate_var,
                                real const phl_gate_var, Layer *const layer_it);
  void lstm_initialize_identity(Layer const *const layer_it);
  void lstm_initialize_orthogonal(Layer const *const layer_it);
  void reset_global_loss(void);
  void reset_loss(void);
  void merge_mp_accu_loss(void);
  void Merge__Accuracy__R(void);

  void Set__Maximum_Allowable_Memory(
      size_t const maximum_allowable_memory_bytes_received);
  void set_loss_fn(LOSS_FN::TYPE const type);
  void set_accu_fn(ACCU_FN::TYPE const type);
  void set_optimizer(OPTIMIZER::TYPE const type);
  void set_bit_fail_limit(double const bit_fail_limit);
  void set_accu(ENV::TYPE const env_type, double const accurancy_received);
  void set_loss(ENV::TYPE const env, double const loss);
  void set_clip_gradient(bool const use_clip_gradient);
  void Clip_Gradient__Loop(size_t const start_index_received,
                           size_t const end_index_received);
  void Clip_Gradient__OpenMP(size_t const start_index_received,
                             size_t const end_index_received);
  void Assign__Sparsity_Activities(size_t const number_threads_received);
  // TODO: Normalization tied.
  void Tied__Transpose(void);
  void Tied__Transpose(Layer *const ptr_layer_received);
  void Tied__Transpose__Weight(Layer *const ptr_layer_received);
  void Tied__Transpose__Weight__FC(
      Layer const *const ptr_coded_layer_it_received,
      Layer const *const ptr_mirror_layer_it_received);
  void Tied__Transpose__Weight__FC_Ind_RNN(
      Layer const *const ptr_encoded_layer_it_received,
      Layer const *const ptr_mirror_layer_it_received);
  void Tied__Transpose__Normalization(Layer *const ptr_layer_received);
  void Tied__Transpose__Normalization__Batch_Normalization(
      Layer const *const ptr_encoded_layer_it_received,
      Layer const *const ptr_mirror_layer_it_received);
  void Update_Parameter(size_t const batch_size,
                        size_t const training_size);
  void update_weights_st(size_t const batch_size,
                              size_t const training_size);
  void update_weights_mp(size_t const batch_size,
                                size_t const training_size);
  void Update_Parameter__Gradient_Descent(size_t const batch_size,
                                          size_t const training_size,
                                          size_t const start_index_received,
                                          size_t const end_index_received);
  void Update_Parameter__Gradient_Descent__Loop(
      size_t const batch_size, size_t const training_size,
      size_t const start_index_received, size_t const end_index_received);
  void Update_Parameter__Gradient_Descent_Momentum__Loop(
      size_t const batch_size, size_t const training_size,
      size_t const start_index_received, size_t const end_index_received);
  void Update_Parameter_Nesterov_Accelerated_Gradient__Loop(
      size_t const batch_size, size_t const training_size,
      size_t const start_index_received, size_t const end_index_received);
  void Update_Parameter__Gradient_Descent__OpenMP(
      size_t const batch_size, size_t const training_size,
      size_t const start_index_received, size_t const end_index_received);
  void Update_Parameter__Gradient_Descent_Momentum__OpenMP(
      size_t const batch_size, size_t const training_size,
      size_t const start_index_received, size_t const end_index_received);
  void Update_Parameter_Nesterov_Accelerated_Gradient__OpenMP(
      size_t const batch_size, size_t const training_size,
      size_t const start_index_received, size_t const end_index_received);
  void Update_Parameters__AdaBound(size_t const batch_size,
                                   size_t const training_size,
                                   size_t const start_index_received,
                                   size_t const end_index_received);
  void Update_Parameters__AdaBound__Loop(size_t const batch_size,
                                         size_t const training_size,
                                         size_t const start_index_received,
                                         size_t const end_index_received);
  void Update_Parameters__AdaBound__OpenMP(size_t const batch_size,
                                           size_t const training_size,
                                           size_t const start_index_received,
                                           size_t const end_index_received);
  void Update_Parameters__Adam(size_t const batch_size,
                               size_t const training_size,
                               size_t const start_index_received,
                               size_t const end_index_received);
  void Update_Parameters__Adam__Loop(size_t const batch_size,
                                     size_t const training_size,
                                     size_t const start_index_received,
                                     size_t const end_index_received);
  void Update_Parameters__Adam__OpenMP(size_t const batch_size,
                                       size_t const training_size,
                                       size_t const start_index_received,
                                       size_t const end_index_received);
  void Update_Parameters__AMSBound(size_t const batch_size,
                                   size_t const training_size,
                                   size_t const start_index_received,
                                   size_t const end_index_received);
  void Update_Parameters__AMSBound__Loop(size_t const batch_size,
                                         size_t const training_size,
                                         size_t const start_index_received,
                                         size_t const end_index_received);
  void Update_Parameters__AMSBound__OpenMP(size_t const batch_size,
                                           size_t const training_size,
                                           size_t const start_index_received,
                                           size_t const end_index_received);
  void Update_Parameters__AMSGrad(size_t const batch_size,
                                  size_t const training_size,
                                  size_t const start_index_received,
                                  size_t const end_index_received);
  void Update_Parameters__AMSGrad__Loop(size_t const batch_size,
                                        size_t const training_size,
                                        size_t const start_index_received,
                                        size_t const end_index_received);
  void Update_Parameters__AMSGrad__OpenMP(size_t const batch_size,
                                          size_t const training_size,
                                          size_t const start_index_received,
                                          size_t const end_index_received);
  void Update_Parameters__NosAdam(size_t const batch_size,
                                  size_t const training_size,
                                  size_t const start_index_received,
                                  size_t const end_index_received);
  void Update_Parameters__NosAdam__Loop(size_t const batch_size,
                                        size_t const training_size,
                                        size_t const start_index_received,
                                        size_t const end_index_received);
  void Update_Parameters__NosAdam__OpenMP(size_t const batch_size,
                                          size_t const training_size,
                                          size_t const start_index_received,
                                          size_t const end_index_received);
  void Update_Parameter__iRPROP_plus(size_t const start_index_received,
                                     size_t const end_index_received);
  void Update_Parameter__iRPROP_minus__Loop(size_t const start_index_received,
                                            size_t const end_index_received);
  void Update_Parameter__iRPROP_plus__Loop(size_t const start_index_received,
                                           size_t const end_index_received);
  void Update_Parameter__iRPROP_plus__OpenMP(size_t const start_index_received,
                                             size_t const end_index_received);
  void Dropout_Bernoulli(void);
  void Dropout_Bernoulli__Loop(void);
  void Dropout_Bernoulli__Layer__Loop(size_t const number_outputs_received,
                                      Layer *const layer_it);
  void Dropout_Bernoulli__OpenMP(void);
  void Dropout_Bernoulli__Layer__OpenMP(size_t const number_outputs_received,
                                        Layer *const layer_it);
  void Dropout_Zoneout(void);
  void Dropout_Zoneout__Loop(void);
  void Dropout_Zoneout__Block_Units__Loop(Layer *const layer_it);
  void Dropout_Zoneout__OpenMP(void);
  void Dropout_Zoneout__Block_Units__OpenMP(Layer *const layer_it);
  void Update_Weight_Regularization__Max_Norm_Constraints(
      size_t const start_index_received, size_t const end_index_received);
  void Update_Weight_Regularization__Max_Norm_Constraints__Loop(
      size_t const start_index_received, size_t const end_index_received);
  void Update_Weight_Regularization__Max_Norm_Constraints__Neurons__Loop(
      size_t const start_index_received, size_t const end_index_received,
      Layer const *const layer_it, Layer const *const last_layer);
  void Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__Loop(
      size_t const start_index_received, size_t const end_index_received,
      Layer const *const layer_it, Layer const *const last_layer);
  void Update_Weight_Regularization__Max_Norm_Constraints__OpenMP(
      size_t const start_index_received, size_t const end_index_received);
  void Update_Weight_Regularization__Max_Norm_Constraints__Neurons__OpenMP(
      size_t const start_index_received, size_t const end_index_received,
      Layer const *const layer_it, Layer const *const last_layer);
  void Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__OpenMP(
      size_t const start_index_received, size_t const end_index_received,
      Layer const *const layer_it, Layer const *const last_layer);
  void Update_Weight_Regularization__Constraint_Recurrent_Weight(
      size_t const start_index_received, size_t const end_index_received);
  void Update_Weight_Regularization__Constraint_Recurrent_Weight__FC_Ind_RNN(
      Layer const *const layer_it);
  void Update_Derivative_Weight__Regularization__L1(
      size_t const start_index_received, size_t const end_index_received,
      size_t const batch_size);
  void Update_Derivative_Weight__Regularization__L1__Loop(
      size_t const start_index_received, size_t const end_index_received,
      size_t const batch_size);
  void Update_Derivative_Weight__Regularization__L1__OpenMP(
      size_t const start_index_received, size_t const end_index_received,
      size_t const batch_size);
  void Update_Derivative_Weight__Regularization__L2(
      size_t const start_index_received, size_t const end_index_received,
      size_t const batch_size);
  void Update_Derivative_Weight__Regularization__L2__Loop(
      size_t const start_index_received, size_t const end_index_received,
      size_t const batch_size);
  void Update_Derivative_Weight__Regularization__L2__OpenMP(
      size_t const start_index_received, size_t const end_index_received,
      size_t const batch_size);
  void Update_Derivative_Weight__Regularization__SRIP(
      size_t const start_index_received, size_t const end_index_received,
      size_t const batch_size);
  void Update_Derivative_Weight__Regularization__SRIP__Loop(
      size_t const start_index_received, size_t const end_index_received,
      size_t const batch_size);
  void Update_Derivative_Weight__Regularization__SRIP__OpenMP(
      size_t const start_index_received, size_t const end_index_received,
      size_t const batch_size);
  void Sparse_K_Filter(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received, size_t const k_sparsity_received,
      std::pair<size_t, var> *const ptr_array_k_sparses_received,
      var *const ptr_array_inputs_received);
  void Sparse_K_Filter__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received, size_t const k_sparsity_received,
      std::pair<size_t, var> *const ptr_array_k_sparses_received,
      var *const ptr_array_inputs_received);
  void Sparse_K_Filter__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received, size_t const k_sparsity_received,
      std::pair<size_t, var> *const ptr_array_k_sparses_received,
      var *const ptr_array_inputs_received);
  void compute_loss(size_t const batch_size, real const *const *const Ym);
  void compute_loss_fwp_st(size_t const batch_size, real const *const *const Ym);
  void compute_loss_fwp_mp(size_t const batch_size, real const *const *const Ym);
  void compute_loss_rec_st(size_t const batch_size, real const *const *const Ym);
  void compute_loss_rec_mp(size_t const batch_size, real const *const *const Ym);
  void compute_loss_pre_train(size_t const batch_size);
  void compute_loss_pre_train_fwp_st(size_t const batch_size);
  void compute_loss_pre_train_fwp_mp(size_t const batch_size);
  void compute_loss_pre_train_rec_st(size_t const batch_size);
  void compute_loss_pre_train_rec_mp(size_t const batch_size);
  void compute_error(size_t const batch_size, real const *const *const Ym);
  void compute_error_fwp_st(size_t const batch_size,
                            real const *const *const Ym);
  void compute_error_fwp_mp(size_t const batch_size,
                            real const *const *const Ym);
  void compute_error_rec_st(size_t const batch_size,
                            real const *const *const Ym);
  void compute_error_rec_mp(size_t const batch_size,
                            real const *const *const Ym);
  void compute_error_pre_train(size_t const batch_size);
  void compute_error_pre_train_fwp_st(size_t const batch_size);
  void compute_error_pre_train_fwp_mp(size_t const batch_size);
  void compute_error_pre_train_rec_st(size_t const batch_size);
  void compute_error_pre_train_rec_mp(size_t const batch_size);
  void compute_r(size_t const batch_size, real const *const *const Ym);
  void compute_r_fwp_st(size_t const batch_size, real const *const *const Ym);
  void compute_r_fwp_mp(size_t const batch_size, real const *const *const Ym);
  void compute_r_rec_st(size_t const batch_size, real const *const *const Ym);
  void compute_r_rec_mp(size_t const batch_size, real const *const *const Ym);
  void compute_r_pre_train(size_t const batch_size);
  void compute_r_pre_train_fwp_st(size_t const batch_size);
  void compute_r_pre_train_fwp_mp(size_t const batch_size);
  void compute_r_pre_train_rec_st(size_t const batch_size);
  void compute_r_pre_train_rec_mp(size_t const batch_size);
  void FF__Forward_Pass_Batch__Loop(
      size_t const batch_size,
      real const *const *const ptr_array_inputs_received,
      Layer *const ptr_first_layer_received,
      Layer const *const last_layer);
  void FF__Forward_Pass_Batch__Pre_Training__Loop(
      size_t const batch_size,
      real const *const *const ptr_array_inputs_received);
  void Forward_Pass__Average_Pooling__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__FC__Loop(size_t const time_step_index_received,
                              size_t const batch_size,
                              size_t const input_unit_size_received,
                              var const *const ptr_array_inputs_received,
                              Layer *const layer_it);
  void Forward_Pass__Encode__FC__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Code__FC__Loop(size_t const time_step_index_received,
                                    size_t const batch_size,
                                    size_t const input_unit_size_received,
                                    var const *const ptr_array_inputs_received,
                                    Layer *const layer_it);
  void Forward_Pass__Decode__FC__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__LSTM__Loop(
      long long int const time_step_index_received,
      long long int const tmp_time_step_reverse_direction,
      long long int const tmp_time_step_start, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Encode__LSTM__Loop(
      long long int const time_step_index_received,
      long long int const tmp_time_step_reverse_direction,
      long long int const tmp_time_step_start, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Code__LSTM__Loop(
      long long int const time_step_index_received,
      long long int const tmp_time_step_reverse_direction,
      long long int const tmp_time_step_start, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Decode__LSTM__Loop(
      long long int const time_step_index_received,
      long long int const tmp_time_step_reverse_direction,
      long long int const tmp_time_step_start, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Max_Pooling__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Residual__Loop(size_t const batch_size, Layer *&layer_it);
  void Forward_Pass__Residual__Layer__Loop(
      bool const is_block_input_layer_received, size_t const batch_size,
      Layer *&layer_it);
  void Forward_Pass__Residual__FC__Loop(
      bool const is_block_input_layer_received,
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Average_Pooling__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received, size_t const output_size_received,
      size_t const kernel_size_received, size_t const stride_received,
      size_t const padding_received, size_t const dilation_received,
      var const *const ptr_array_inputs_received,
      var *const ptr_array_outputs_received);
  void Forward_Pass__Bias__Loop(size_t const time_step_index_received,
                                size_t const batch_size,
                                size_t const output_size_received,
                                var const *const ptr_array_bias_received,
                                var *const ptr_array_outputs_received);
  void Forward_Pass__FC__Loop(size_t const time_step_index_received,
                              size_t const batch_size,
                              size_t const input_size_received,
                              size_t const output_size_received,
                              var const *const ptr_array_inputs_received,
                              var const *const ptr_array_parameters_received,
                              var *const ptr_array_outputs_received);
  void Forward_Pass__FC_Ind_RNN__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_parameters_received,
      var const *const ptr_array_AFs_received,
      var const *const ptr_array_inputs_received,
      var *const ptr_array_outputs_received);
  void Forward_Pass__Batch_Normalization__Inference__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_shifts_received,
      var const *const ptr_array_means_averages_received,
      var const *const ptr_array_variances_averages_received,
      var *const ptr_array_output_normalizes_received);
  void Forward_Pass__Batch_Normalization__Training__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_shifts_received,
      var *const ptr_array_means_received,
      var *const ptr_array_variances_received,
      var *const ptr_array_means_averages_received,
      var *const ptr_array_variances_averages_received,
      var *const ptr_array_output_hats_received,
      var *const ptr_array_output_normalizes_received);
  void Forward_Pass__Batch_Renormalization__Training__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_shifts_received,
      var *const ptr_array_means_received,
      var *const ptr_array_variances_received,
      var *const ptr_array_means_averages_received,
      var *const ptr_array_variances_averages_received,
      var *const ptr_array_r_corrections_received,
      var *const ptr_array_d_corrections_received,
      var *const ptr_array_output_hats_received,
      var *const ptr_array_output_normalizes_received);
  void Forward_Pass__FC_AF__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      var *const ptr_array_outputs_received,
      ACTIVATION::TYPE const *const ptr_array_type_activations_received);
  void Forward_Pass__FC_AF__Softmax__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      var *const ptr_array_outputs_received);
  void Forward_Pass__Dropout__Bernoulli__Inverted__Loop(
      bool const *const ptr_array__mask__dropout__bernoulli_received,
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      real const inverse_retention_probability_divided_received,
      var *const ptr_array_inputs_received);
  void Forward_Pass__Dropout__Bernoulli__Training__Loop(
      bool const *const ptr_array__mask__dropout__bernoulli_received,
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, var *const ptr_array_inputs_received);
  void Forward_Pass__Dropout__Bernoulli__Inference__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      real const retention_probability_received,
      var *const ptr_array_inputs_received);
  void Forward_Pass__Dropout__Gaussian__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, real const variance,
      var *const ptr_array_inputs_received);
  void Forward_Pass__Dropout__ShakeDrop__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      bool *const ptr_array_mask_dopout_shakedrop_received,
      real const lower_bound, real const upper_bound,
      real const dropout_probability_received,
      var *const ptr_array_inputs_received);
  void Forward_Pass__Dropout__Uout__Loop(size_t const time_step_index_received,
                                         size_t const batch_size,
                                         size_t const input_size_received,
                                         real const beta_received,
                                         var *const ptr_array_inputs_received);
  void Forward_Pass__Max_Pooling__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const output_size_received,
      size_t const kernel_size_received, size_t const stride_received,
      size_t const padding_received, size_t const dilation_received,
      size_t *const ptr_array_indices_received,
      var const *const ptr_array_inputs_received,
      var *const ptr_array_outputs_received);
  void Forward_Pass__Zero_Padded_Identity__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const A_unit_size_received, size_t const B_unit_size_received,
      size_t const padding_received, var const *const ptr_array_A_received,
      var const *const ptr_array_B_received,
      var *const ptr_array_outputs_received);
  void FF__Forward_Pass_Batch__OpenMP(
      size_t const batch_size,
      real const *const *const ptr_array_inputs_received,
      Layer *const ptr_first_layer_received,
      Layer const *const last_layer);
  void FF__Forward_Pass_Batch__Pre_Training__OpenMP(
      size_t const batch_size,
      real const *const *const ptr_array_inputs_received);
  void Forward_Pass__Average_Pooling__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                size_t const batch_size,
                                size_t const input_unit_size_received,
                                var const *const ptr_array_inputs_received,
                                Layer *const layer_it);
  void Forward_Pass__Encode__FC__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Code__FC__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Decode__FC__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__LSTM__OpenMP(
      long long int const time_step_index_received,
      long long int const tmp_time_step_reverse_direction,
      long long int const tmp_time_step_start, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Encode__LSTM__OpenMP(
      long long int const time_step_index_received,
      long long int const tmp_time_step_reverse_direction,
      long long int const tmp_time_step_start, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Code__LSTM__OpenMP(
      long long int const time_step_index_received,
      long long int const tmp_time_step_reverse_direction,
      long long int const tmp_time_step_start, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Decode__LSTM__OpenMP(
      long long int const time_step_index_received,
      long long int const tmp_time_step_reverse_direction,
      long long int const tmp_time_step_start, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Max_Pooling__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Residual__OpenMP(size_t const batch_size,
                                      Layer *&layer_it);
  void Forward_Pass__Residual__Layer__OpenMP(
      bool const is_block_input_layer_received, size_t const batch_size,
      Layer *&layer_it);
  void Forward_Pass__Residual__FC__OpenMP(
      bool const is_block_input_layer_received,
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__Average_Pooling__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received, size_t const output_size_received,
      size_t const kernel_size_received, size_t const stride_received,
      size_t const padding_received, size_t const dilation_received,
      var const *const ptr_array_inputs_received,
      var *const ptr_array_outputs_received);
  void Forward_Pass__Bias__OpenMP(size_t const time_step_index_received,
                                  size_t const batch_size,
                                  size_t const output_size_received,
                                  var const *const ptr_array_bias_received,
                                  var *const ptr_array_outputs_received);
  void Forward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                size_t const batch_size,
                                size_t const input_size_received,
                                size_t const output_size_received,
                                var const *const ptr_array_inputs_received,
                                var const *const ptr_array_parameters_received,
                                var *const ptr_array_outputs_received);
  void Forward_Pass__FC_Ind_RNN__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_parameters_received,
      var const *const ptr_array_AFs_received,
      var const *const ptr_array_inputs_received,
      var *const ptr_array_outputs_received);
  void Forward_Pass__Batch_Normalization__Inference__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_shifts_received,
      var const *const ptr_array_means_averages_received,
      var const *const ptr_array_variances_averages_received,
      var *const ptr_array_output_normalizes_received);
  void Forward_Pass__Batch_Normalization__Training__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_shifts_received,
      var *const ptr_array_means_received,
      var *const ptr_array_variances_received,
      var *const ptr_array_means_averages_received,
      var *const ptr_array_variances_averages_received,
      var *const ptr_array_output_hats_received,
      var *const ptr_array_output_normalizes_received);
  void Forward_Pass__Batch_Renormalization__Training__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_shifts_received,
      var *const ptr_array_means_received,
      var *const ptr_array_variances_received,
      var *const ptr_array_means_averages_received,
      var *const ptr_array_variances_averages_received,
      var *const ptr_array_r_corrections_received,
      var *const ptr_array_d_corrections_received,
      var *const ptr_array_output_hats_received,
      var *const ptr_array_output_normalizes_received);
  void Forward_Pass__FC_AF__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      var *const ptr_array_outputs_received,
      ACTIVATION::TYPE const *const ptr_array_type_activations_received);
  void Forward_Pass__FC_AF__Softmax__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      var *const ptr_array_outputs_received);
  void Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(
      bool const *const ptr_array__mask__dropout__bernoulli_received,
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      real const inverse_retention_probability_divided_received,
      var *const ptr_array_inputs_received);
  void Forward_Pass__Dropout__Bernoulli__Training__OpenMP(
      bool const *const ptr_array__mask__dropout__bernoulli_received,
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, var *const ptr_array_inputs_received);
  void Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      real const retention_probability_received,
      var *const ptr_array_inputs_received);
  void Forward_Pass__Dropout__Gaussian__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, real const variance,
      var *const ptr_array_inputs_received);
  void Forward_Pass__Dropout__ShakeDrop__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      bool *const ptr_array_mask_dopout_shakedrop_received,
      real const lower_bound, real const upper_bound,
      real const dropout_probability_received,
      var *const ptr_array_inputs_received);
  void Forward_Pass__Dropout__Uout__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, real const beta_received,
      var *const ptr_array_inputs_received);
  void Forward_Pass__Max_Pooling__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const output_size_received,
      size_t const kernel_size_received, size_t const stride_received,
      size_t const padding_received, size_t const dilation_received,
      size_t *const ptr_array_indices_received,
      var const *const ptr_array_inputs_received,
      var *const ptr_array_outputs_received);
  void Forward_Pass__Zero_Padded_Identity__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const A_unit_size_received, size_t const B_unit_size_received,
      size_t const padding_received, var const *const ptr_array_A_received,
      var const *const ptr_array_B_received,
      var *const ptr_array_outputs_received);
  void RNN__Forward_Pass_Batch__Loop(
      size_t const batch_size,
      real const *const *const ptr_array_inputs_received,
      Layer *const ptr_first_layer_received,
      Layer const *const last_layer);
  void RNN__Forward_Pass_Batch__Pre_Training__Loop(
      size_t const batch_size,
      real const *const *const ptr_array_inputs_received);
  void Recurrent__Forward_Pass__Average_Pooling__Loop(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Dropout__Bernoulli__Inference__Loop(
      size_t const batch_size, size_t const input_unit_size_received,
      real const retention_probability_received,
      var *const ptr_array_inputs_received);
  void Recurrent__Forward_Pass__Dropout__ShakeDrop__Loop(
      size_t const batch_size, size_t const input_unit_size_received,
      bool *const ptr_array_mask_dopout_shakedrop_received,
      real const lower_bound, real const upper_bound,
      real const dropout_probability_received,
      var *const ptr_array_inputs_received);
  void Recurrent__Forward_Pass__FC__Loop(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Encode__FC__Loop(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Code__FC__Loop(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Decode__FC__Loop(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__LSTM__Loop(
      bool const forward_layer_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Encode__LSTM__Loop(
      bool const forward_layer_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Code__LSTM__Loop(
      bool const forward_layer_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Decode__LSTM__Loop(
      bool const forward_layer_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Max_Pooling__Loop(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Residual__Loop(size_t const batch_size,
                                               Layer *&layer_it);
  void Recurrent__Forward_Pass__Residual__Layer__Loop(
      bool const is_block_input_layer_received, size_t const batch_size,
      Layer *&layer_it);
  void Recurrent__Forward_Pass__Residual__FC__Loop(
      bool const is_block_input_layer_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Zero_Padded_Identity__Loop(
      size_t const batch_size, size_t const size_A_received,
      size_t const size_B_received, var const *const ptr_array_A_received,
      var const *const ptr_array_B_received, Layer *const layer_it);
  void Forward_Pass__LSTM__Gates_CIFO__Loop(
      long long int const time_step_index_received,
      long long int const time_step_reverse_direction_received,
      long long int const time_step_prediction_start_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__LSTM__Gates_CIF_AF_State__Loop(
      long long int const time_step_index_received,
      long long int const time_step_reverse_direction_received,
      long long int const time_step_prediction_start_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_input_block_inputs_received,
      var const *const ptr_array_summation_recurrent_block_inputs_received,
      var const *const ptr_array_summation_input_inputs_gates_received,
      var const *const ptr_array_summation_recurrent_inputs_gates_received,
      var const *const ptr_array_summation_input_forgets_gates_received,
      var const *const ptr_array_summation_recurrent_forgets_gates_received,
      Layer *const layer_it);
  void Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__Loop(
      long long int const time_step_index_received,
      long long int const time_step_reverse_direction_received,
      long long int const time_step_prediction_start_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_input_block_inputs_received,
      var const *const ptr_array_summation_recurrent_block_inputs_received,
      var const *const ptr_array_summation_input_inputs_gates_received,
      var const *const ptr_array_summation_recurrent_inputs_gates_received,
      var const *const ptr_array_summation_input_forgets_gates_received,
      var const *const ptr_array_summation_recurrent_forgets_gates_received,
      Layer *const layer_it);
  void Forward_Pass__LSTM__Output__Loop(
      long long int const time_step_index_received, size_t const batch_size,
      size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_input_outputs_gates_received,
      var const *const ptr_array_summation_recurrent_outputs_gates_received,
      Layer *const layer_it);
  void Forward_Pass__LSTM__Output__Zoneout__Loop(
      long long int const time_step_index_received,
      long long int const time_step_reverse_direction_received,
      long long int const time_step_prediction_start_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_input_outputs_gates_received,
      var const *const ptr_array_summation_recurrent_outputs_gates_received,
      Layer *const layer_it);
  void Forward_Pass__LSTM__States_AF__Loop(
      long long int const time_step_index_received, size_t const batch_size,
      size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_cell_states_received,
      Layer *const layer_it);
  void RNN__Forward_Pass_Batch__OpenMP(
      size_t const batch_size,
      real const *const *const ptr_array_inputs_received,
      Layer *const ptr_first_layer_received,
      Layer const *const last_layer);
  void RNN__Forward_Pass_Batch__Pre_Training__OpenMP(
      size_t const batch_size,
      real const *const *const ptr_array_inputs_received);
  void Recurrent__Forward_Pass__Average_Pooling__OpenMP(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(
      size_t const batch_size, size_t const input_unit_size_received,
      real const retention_probability_received,
      var *const ptr_array_inputs_received);
  void Recurrent__Forward_Pass__Dropout__ShakeDrop__OpenMP(
      size_t const batch_size, size_t const input_unit_size_received,
      bool *const ptr_array_mask_dopout_shakedrop_received,
      real const lower_bound, real const upper_bound,
      real const dropout_probability_received,
      var *const ptr_array_inputs_received);
  void Recurrent__Forward_Pass__FC__OpenMP(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Encode__FC__OpenMP(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Code__FC__OpenMP(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Decode__FC__OpenMP(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__LSTM__OpenMP(
      bool const forward_layer_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Encode__LSTM__OpenMP(
      bool const forward_layer_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Code__LSTM__OpenMP(
      bool const forward_layer_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Decode__LSTM__OpenMP(
      bool const forward_layer_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Max_Pooling__OpenMP(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Residual__OpenMP(size_t const batch_size,
                                                 Layer *&layer_it);
  void Recurrent__Forward_Pass__Residual__Layer__OpenMP(
      bool const is_block_input_layer_received, size_t const batch_size,
      Layer *&layer_it);
  void Recurrent__Forward_Pass__Residual__FC__OpenMP(
      bool const is_block_input_layer_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Recurrent__Forward_Pass__Zero_Padded_Identity__OpenMP(
      size_t const batch_size, size_t const size_A_received,
      size_t const size_B_received, var const *const ptr_array_A_received,
      var const *const ptr_array_B_received, Layer *const layer_it);
  void Forward_Pass__LSTM__Gates_CIFO__OpenMP(
      long long int const time_step_index_received,
      long long int const time_step_reverse_direction_received,
      long long int const time_step_prediction_start_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Forward_Pass__LSTM__Gates_CIF_AF_State__OpenMP(
      long long int const time_step_index_received,
      long long int const time_step_reverse_direction_received,
      long long int const time_step_prediction_start_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_input_block_inputs_received,
      var const *const ptr_array_summation_recurrent_block_inputs_received,
      var const *const ptr_array_summation_input_inputs_gates_received,
      var const *const ptr_array_summation_recurrent_inputs_gates_received,
      var const *const ptr_array_summation_input_forgets_gates_received,
      var const *const ptr_array_summation_recurrent_forgets_gates_received,
      Layer *const layer_it);
  void Forward_Pass__LSTM__Gates_CIF_AF_State__Zoneout__OpenMP(
      long long int const time_step_index_received,
      long long int const time_step_reverse_direction_received,
      long long int const time_step_prediction_start_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_input_block_inputs_received,
      var const *const ptr_array_summation_recurrent_block_inputs_received,
      var const *const ptr_array_summation_input_inputs_gates_received,
      var const *const ptr_array_summation_recurrent_inputs_gates_received,
      var const *const ptr_array_summation_input_forgets_gates_received,
      var const *const ptr_array_summation_recurrent_forgets_gates_received,
      Layer *const layer_it);
  void Forward_Pass__LSTM__Output__OpenMP(
      long long int const time_step_index_received, size_t const batch_size,
      size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_input_outputs_gates_received,
      var const *const ptr_array_summation_recurrent_outputs_gates_received,
      Layer *const layer_it);
  void Forward_Pass__LSTM__Output__Zoneout__OpenMP(
      long long int const time_step_index_received,
      long long int const time_step_reverse_direction_received,
      long long int const time_step_prediction_start_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_input_outputs_gates_received,
      var const *const ptr_array_summation_recurrent_outputs_gates_received,
      Layer *const layer_it);
  void Forward_Pass__LSTM__States_AF__OpenMP(
      long long int const time_step_index_received, size_t const batch_size,
      size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_cell_states_received,
      Layer *const layer_it);
  void backward_pass(size_t const batch_size);
  void Backward_Pass__Pre_Training(size_t const batch_size);
  void FF__Backward_Pass_Batch__Loop(size_t const batch_size);
  void FF__Backward_Pass_Batch__Pre_Training__Loop(size_t const batch_size);
  void Backward_Pass__FC__Loop(size_t const batch_size,
                               size_t const derivative_size_received,
                               real *const ptr_array_derivatives_received,
                               Layer const *const layer_it);
  void Backward_Pass__Average_Pooling__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      real *const ptr_array_derivatives_received, Layer const *const layer_it);
  void Backward_Pass__FC__Loop(size_t const time_step_index_received,
                               size_t const batch_size,
                               size_t const derivative_size_received,
                               real *const ptr_array_derivatives_received,
                               Layer const *const layer_it);
  void Backward_Pass__Max_Pooling__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      real *const ptr_array_derivatives_received, Layer const *const layer_it);
  void Backward_Pass__Residual__Loop(size_t const time_step_index_received,
                                     size_t const batch_size,
                                     size_t const derivative_size_received,
                                     real *const ptr_array_derivatives_received,
                                     Layer const *const layer_it);
  void Backward_Pass__Residual__Block__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      real *const ptr_array_derivatives_received, Layer const *const layer_it);
  void Backward_Pass__Residual__FC__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      real *const ptr_array_derivatives_received, Layer const *const layer_it);
  void Backward_Pass__Gradient__FC__Loop(size_t const time_step_index_received,
                                         size_t const batch_size,
                                         Layer const *const layer_it);
  void Backward_Pass__Gradient__Residual__Loop(size_t const batch_size,
                                               Layer const *const layer_it);
  void Backward_Pass__Gradient__Residual__Layer__Loop(
      bool const is_block_input_layer_received, size_t const batch_size,
      Layer *&layer_it);
  void Backward_Pass__Gradient__Residual__FC__Loop(
      bool const is_block_input_layer_received,
      size_t const time_step_index_received, size_t const batch_size,
      Layer const *const layer_it);
  void Backward_Pass__Average_Pooling__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const derivative_size_received,
      size_t const kernel_size_received, size_t const stride_received,
      size_t const padding_received, size_t const dilation_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Dropout__ShakeDrop__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      bool const *const ptr_array_mask_dopout_shakedrop_received,
      real const lower_bound, real const upper_bound,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__FC__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const derivative_size_received,
      real const *const ptr_array_derivative_inputs_received,
      var const *const ptr_array_parameters_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Identity__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Max_Pooling__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const derivative_size_received,
      size_t const padding_received,
      size_t const *const ptr_array_indices_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Residual__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const derivative_size_received,
      size_t const padding_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__FC__DF__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      ACTIVATION::TYPE const
          *const ptr_array_type_activations_functions_received,
      var const *const ptr_array_pre_AFs_received,
      var const *const ptr_array_AFs_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__FC__DF_Ind_RNN__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_parameters_received,
      ACTIVATION::TYPE const
          *const ptr_array_type_activations_functions_received,
      var const *const ptr_array_pre_AFs_received,
      var const *const ptr_array_AFs_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_dAFs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Batch_Normalization__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_means_received,
      var const *const ptr_array_variances_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_inputs_hats_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_scales_received,
      real *const ptr_array_derivatives_shifts_received,
      real *const ptr_array_derivatives_means_received,
      real *const ptr_array_derivatives_variances_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Batch_Normalization__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_means_received,
      var const *const ptr_array_variances_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_inputs_hats_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_scales_received,
      real *const ptr_array_derivatives_means_received,
      real *const ptr_array_derivatives_variances_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Batch_Renormalization__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_means_received,
      var const *const ptr_array_variances_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_r_corrections_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_inputs_hats_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_scales_received,
      real *const ptr_array_derivatives_shifts_received,
      real *const ptr_array_derivatives_means_received,
      real *const ptr_array_derivatives_variances_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Batch_Renormalization__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_means_received,
      var const *const ptr_array_variances_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_r_corrections_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_inputs_hats_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_scales_received,
      real *const ptr_array_derivatives_means_received,
      real *const ptr_array_derivatives_variances_received,
      real *const ptr_array_derivatives_received);
  void FF__Backward_Pass_Batch__OpenMP(size_t const batch_size);
  void FF__Backward_Pass_Batch__Pre_Training__OpenMP(size_t const batch_size);
  void Backward_Pass__FC__OpenMP(size_t const batch_size,
                                 size_t const derivative_size_received,
                                 real *const ptr_array_derivatives_received,
                                 Layer const *const layer_it);
  void Backward_Pass__Average_Pooling__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      real *const ptr_array_derivatives_received, Layer const *const layer_it);
  void Backward_Pass__FC__OpenMP(size_t const time_step_index_received,
                                 size_t const batch_size,
                                 size_t const derivative_size_received,
                                 real *const ptr_array_derivatives_received,
                                 Layer const *const layer_it);
  void Backward_Pass__Max_Pooling__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      real *const ptr_array_derivatives_received, Layer const *const layer_it);
  void Backward_Pass__Residual__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      real *const ptr_array_derivatives_received, Layer const *const layer_it);
  void Backward_Pass__Residual__Block__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      real *const ptr_array_derivatives_received, Layer const *const layer_it);
  void Backward_Pass__Residual__FC__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      real *const ptr_array_derivatives_received, Layer const *const layer_it);
  void Backward_Pass__Gradient__FC__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      Layer const *const layer_it);
  void Backward_Pass__Gradient__Residual__OpenMP(size_t const batch_size,
                                                 Layer const *const layer_it);
  void Backward_Pass__Gradient__Residual__Layer__OpenMP(
      bool const is_block_input_layer_received, size_t const batch_size,
      Layer *&layer_it);
  void Backward_Pass__Gradient__Residual__FC__OpenMP(
      bool const is_block_input_layer_received,
      size_t const time_step_index_received, size_t const batch_size,
      Layer const *const layer_it);
  void Backward_Pass__Average_Pooling__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const derivative_size_received,
      size_t const kernel_size_received, size_t const stride_received,
      size_t const padding_received, size_t const dilation_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Dropout__ShakeDrop__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      bool const *const ptr_array_mask_dopout_shakedrop_received,
      real const lower_bound, real const upper_bound,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__FC__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const derivative_size_received,
      real const *const ptr_array_derivative_inputs_received,
      var const *const ptr_array_parameters_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Identity__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Max_Pooling__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const derivative_size_received,
      size_t const padding_received,
      size_t const *const ptr_array_indices_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Residual__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const derivative_size_received,
      size_t const padding_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__FC__DF__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      ACTIVATION::TYPE const
          *const ptr_array_type_activations_functions_received,
      var const *const ptr_array_pre_AFs_received,
      var const *const ptr_array_AFs_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__FC__DF_Ind_RNN__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_parameters_received,
      ACTIVATION::TYPE const
          *const ptr_array_type_activations_functions_received,
      var const *const ptr_array_pre_AFs_received,
      var const *const ptr_array_AFs_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_dAFs_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Batch_Normalization__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_means_received,
      var const *const ptr_array_variances_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_inputs_hats_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_scales_received,
      real *const ptr_array_derivatives_shifts_received,
      real *const ptr_array_derivatives_means_received,
      real *const ptr_array_derivatives_variances_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Batch_Normalization__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_means_received,
      var const *const ptr_array_variances_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_inputs_hats_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_scales_received,
      real *const ptr_array_derivatives_means_received,
      real *const ptr_array_derivatives_variances_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Batch_Renormalization__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_means_received,
      var const *const ptr_array_variances_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_r_corrections_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_inputs_hats_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_scales_received,
      real *const ptr_array_derivatives_shifts_received,
      real *const ptr_array_derivatives_means_received,
      real *const ptr_array_derivatives_variances_received,
      real *const ptr_array_derivatives_received);
  void Backward_Pass__Batch_Renormalization__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received,
      var const *const ptr_array_means_received,
      var const *const ptr_array_variances_received,
      var const *const ptr_array_scales_received,
      var const *const ptr_array_r_corrections_received,
      var const *const ptr_array_inputs_received,
      var const *const ptr_array_inputs_hats_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_scales_received,
      real *const ptr_array_derivatives_means_received,
      real *const ptr_array_derivatives_variances_received,
      real *const ptr_array_derivatives_received);
  void RNN__Backward_Pass_Batch__Loop(size_t const batch_size);
  void RNN__Backward_Pass_Batch__Pre_Training__Loop(size_t const batch_size);
  void Recurrent__Backward_Pass__Average_Pooling__Loop(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__FC__Loop(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__LSTM__Loop(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__Max_Pooling__Loop(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__Residual__Loop(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__Residual__Block__Loop(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__Gradient__FC__Loop(
      size_t const batch_size, Layer const *const layer_it);
  void Recurrent__Backward_Pass__Gradient__LSTM__Loop(
      bool const forward_layer_received, size_t const batch_size,
      size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received, Layer *const layer_it);
  void Recurrent__Backward_Pass__Gradient__Residual__Loop(
      size_t const batch_size, Layer const *const layer_it);
  void Recurrent__Backward_Pass__Gradient__Residual__Layer__Loop(
      bool const is_block_input_layer_received, size_t const batch_size,
      Layer *&layer_it);
  void Recurrent__Backward_Pass__Gradient__Residual__FC__Loop(
      bool const is_block_input_layer_received, size_t const batch_size,
      Layer const *const layer_it);
  void Backward_Pass__LSTM__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_input_size_received,
      real const *const ptr_array_delta_input_block_inputs_received,
      real const *const ptr_array_delta_input_input_gates_received,
      real const *const ptr_array_delta_input_forget_gates_received,
      real const *const ptr_array_delta_input_output_gates_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Backward_Pass__LSTM_Derivative__Output__Loop(
      long long int const time_step_index_received,
      long long int const time_step_direction_received,
      long long int const time_step_prediction_end_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      real const *const ptr_array_delta_recurrent_block_inputs_received,
      real const *const ptr_array_delta_recurrent_input_gates_received,
      real const *const ptr_array_delta_recurrent_forget_gates_received,
      real const *const ptr_array_delta_recurrent_output_gates_received,
      Layer *const layer_it);
  void Backward_Pass__LSTM_Derivative__Cell_State_AF__Loop(
      long long int const time_step_index_received,
      long long int const time_step_direction_received,
      long long int const time_step_prediction_start_received,
      long long int const time_step_prediction_end_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_cell_states_received,
      Layer *const layer_it);
  void Backward_Pass__LSTM_Derivative__Gates_CIF_AF_State__Loop(
      long long int const time_step_index_received,
      long long int const time_step_direction_received,
      long long int const time_step_reverse_direction_received,
      long long int const time_step_prediction_start_received,
      long long int const time_step_prediction_end_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received, Layer *const layer_it);
  void RNN__Backward_Pass_Batch__OpenMP(size_t const batch_size);
  void RNN__Backward_Pass_Batch__Pre_Training__OpenMP(size_t const batch_size);
  void Recurrent__Backward_Pass__Average_Pooling__OpenMP(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__FC__OpenMP(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__LSTM__OpenMP(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__Max_Pooling__OpenMP(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__Residual__OpenMP(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__Residual__Block__OpenMP(
      size_t const batch_size, size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Recurrent__Backward_Pass__Gradient__FC__OpenMP(
      size_t const batch_size, Layer const *const layer_it);
  void Recurrent__Backward_Pass__Gradient__LSTM__OpenMP(
      bool const forward_layer_received, size_t const batch_size,
      size_t const derivative_input_size_received,
      real *const ptr_array_derivative_inputs_received, Layer *const layer_it);
  void Recurrent__Backward_Pass__Gradient__Residual__OpenMP(
      size_t const batch_size, Layer const *const layer_it);
  void Recurrent__Backward_Pass__Gradient__Residual__Layer__OpenMP(
      bool const is_block_input_layer_received, size_t const batch_size,
      Layer *&layer_it);
  void Recurrent__Backward_Pass__Gradient__Residual__FC__OpenMP(
      bool const is_block_input_layer_received, size_t const batch_size,
      Layer const *const layer_it);
  void Backward_Pass__LSTM__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_input_size_received,
      real const *const ptr_array_delta_input_block_inputs_received,
      real const *const ptr_array_delta_input_input_gates_received,
      real const *const ptr_array_delta_input_forget_gates_received,
      real const *const ptr_array_delta_input_output_gates_received,
      real *const ptr_array_derivative_inputs_received,
      Layer const *const layer_it);
  void Backward_Pass__LSTM_Derivative__Output__OpenMP(
      long long int const time_step_index_received,
      long long int const time_step_direction_received,
      long long int const time_step_prediction_end_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      real const *const ptr_array_delta_recurrent_block_inputs_received,
      real const *const ptr_array_delta_recurrent_input_gates_received,
      real const *const ptr_array_delta_recurrent_forget_gates_received,
      real const *const ptr_array_delta_recurrent_output_gates_received,
      Layer *const layer_it);
  void Backward_Pass__LSTM_Derivative__Cell_State_AF__OpenMP(
      long long int const time_step_index_received,
      long long int const time_step_direction_received,
      long long int const time_step_prediction_start_received,
      long long int const time_step_prediction_end_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received,
      var const *const ptr_array_summation_cell_states_received,
      Layer *const layer_it);
  void Backward_Pass__LSTM_Derivative__Gates_CIF_AF_State__OpenMP(
      long long int const time_step_index_received,
      long long int const time_step_direction_received,
      long long int const time_step_reverse_direction_received,
      long long int const time_step_prediction_start_received,
      long long int const time_step_prediction_end_received,
      size_t const batch_size, size_t const block_unit_size_received,
      size_t const cell_unit_size_received, Layer *const layer_it);
  void update_derivatives(size_t const batch_size, Layer *const layer_it,
                                Layer const *const last_layer);
  void Update_Derivative_Weight__Pre_Training(size_t const batch_size);
  void FF__Update_Derivative_Weight_Batch__Loop(size_t const batch_size,
                                                Layer *layer_it,
                                                Layer const *const last_layer);
  void FF__Update_Derivative_Weight_Batch__Pre_Training__Loop(
      size_t const batch_size);
  void Update_Derivative_Weight__FC__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Update_Derivative_Weight__FC__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const derivative_size_received,
      var const *const ptr_array_inputs_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Update_Derivative_Weight__Bias__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const unit_size_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void FF__Update_Derivative_Weight_Batch__OpenMP(
      size_t const batch_size, Layer *layer_it, Layer const *const last_layer);
  void FF__Update_Derivative_Weight_Batch__Pre_Training__OpenMP(
      size_t const batch_size);
  void Update_Derivative_Weight__FC__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Update_Derivative_Weight__FC__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const input_size_received, size_t const derivative_size_received,
      var const *const ptr_array_inputs_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Update_Derivative_Weight__Bias__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const unit_size_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void RNN__Update_Derivative_Weight_Batch__Loop(size_t const batch_size,
                                                 Layer *layer_it,
                                                 Layer const *const last_layer);
  void RNN__Update_Derivative_Weight_Batch__Pre_Training__Loop(
      size_t const batch_size);
  void Recurrent__Update_Derivative_Weight__FC__Loop(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Update_Derivative_Weight__FC_Ind_RNN__Loop(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      var const *const ptr_array_inputs_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Recurrent__Update_Derivative_Weight__LSTM__Loop(
      bool const forward_layer_received, size_t const batch_size,
      size_t const block_unit_size_received,
      size_t const cell_unit_size_received, size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      real const *const ptr_array_delta_block_inputs_received,
      real const *const ptr_array_delta_input_block_inputs_received,
      real const *const ptr_array_delta_recurrent_block_inputs_received,
      real const *const ptr_array_delta_input_gates_received,
      real const *const ptr_array_delta_input_input_gates_received,
      real const *const ptr_array_delta_recurrent_input_gates_received,
      real const *const ptr_array_delta_forget_gates_received,
      real const *const ptr_array_delta_input_forget_gates_received,
      real const *const ptr_array_delta_recurrent_forget_gates_received,
      real const *const ptr_array_delta_output_gates_received,
      real const *const ptr_array_delta_input_output_gates_received,
      real const *const ptr_array_delta_recurrent_output_gates_received,
      Layer *const layer_it);
  void Recurrent__Update_Derivative_Weight__LSTM__Bias__Loop(
      size_t const batch_size, size_t const layer_block_unit_size_received,
      size_t const layer_cell_unit_size_received,
      real const *const ptr_array_delta_block_inputs_received,
      real const *const ptr_array_delta_input_gates_received,
      real const *const ptr_array_delta_forget_gates_received,
      real const *const ptr_array_delta_output_gates_received,
      real *const ptr_array_cell_input_derivatives_bias_received,
      real *const ptr_array_input_gate_derivatives_bias_received,
      real *const ptr_array_forget_gate_derivatives_bias_received,
      real *const ptr_array_output_gate_derivatives_bias_received);
  void RNN__Update_Derivative_Weight_Batch__OpenMP(
      size_t const batch_size, Layer *layer_it, Layer const *const last_layer);
  void RNN__Update_Derivative_Weight_Batch__Pre_Training__OpenMP(
      size_t const batch_size);
  void Recurrent__Update_Derivative_Weight__FC__OpenMP(
      size_t const batch_size, size_t const input_unit_size_received,
      var const *const ptr_array_inputs_received, Layer *const layer_it);
  void Update_Derivative_Weight__FC_Ind_RNN__OpenMP(
      size_t const time_step_index_received, size_t const batch_size,
      size_t const derivative_size_received,
      var const *const ptr_array_inputs_received,
      real const *const ptr_array_derivative_inputs_received,
      real *const ptr_array_derivatives_received);
  void Recurrent__Update_Derivative_Weight__LSTM__OpenMP(
      bool const forward_layer_received, size_t const batch_size,
      size_t const block_unit_size_received,
      size_t const cell_unit_size_received, size_t const input_size_received,
      var const *const ptr_array_inputs_received,
      real const *const ptr_array_delta_block_inputs_received,
      real const *const ptr_array_delta_input_block_inputs_received,
      real const *const ptr_array_delta_recurrent_block_inputs_received,
      real const *const ptr_array_delta_input_gates_received,
      real const *const ptr_array_delta_input_input_gates_received,
      real const *const ptr_array_delta_recurrent_input_gates_received,
      real const *const ptr_array_delta_forget_gates_received,
      real const *const ptr_array_delta_input_forget_gates_received,
      real const *const ptr_array_delta_recurrent_forget_gates_received,
      real const *const ptr_array_delta_output_gates_received,
      real const *const ptr_array_delta_input_output_gates_received,
      real const *const ptr_array_delta_recurrent_output_gates_received,
      Layer *const layer_it);
  void Recurrent__Update_Derivative_Weight__LSTM__Bias__OpenMP(
      size_t const batch_size, size_t const layer_block_unit_size_received,
      size_t const layer_cell_unit_size_received,
      real const *const ptr_array_delta_block_inputs_received,
      real const *const ptr_array_delta_input_gates_received,
      real const *const ptr_array_delta_forget_gates_received,
      real const *const ptr_array_delta_output_gates_received,
      real *const ptr_array_cell_input_derivatives_bias_received,
      real *const ptr_array_input_gate_derivatives_bias_received,
      real *const ptr_array_forget_gate_derivatives_bias_received,
      real *const ptr_array_output_gate_derivatives_bias_received);
  void update_loss(real const q, real const y, int const thread_id = 0);
  void merge_mp_derivatives(size_t const begin, size_t const end);
  double loss_fn(real const q, real const y);
  real loss_fn_derivative(real const q, real const y, size_t const bs,
                          size_t const n_out);
  void Copy__Neuron_Unit(
      Neuron_unit const *const ptr_source_neuron_unit_received,
      Neuron_unit *const ptr_destination_neuron_unit_received);
  void Copy__Neuron_Units(
      size_t const start_index_received, size_t const end_index_received,
      Neuron_unit const *ptr_array_source_neuron_units_received);
  void Copy__AF_Units(size_t const start_index_received,
                      size_t const end_index_received,
                      AF_unit const *ptr_array_source_AF_units_received);
  void Copy__AF_Unit(AF_unit const *const ptr_source_AF_unit_received,
                     AF_unit *const ptr_destination_AF_unit_received);
  void Copy__AF_Ind_Recurrent_Units(
      size_t const start_index_received, size_t const end_index_received,
      AF_Ind_recurrent_unit const
          *ptr_array_source_AF_Ind_recurrent_units_received,
      bool const copy_connections_received = true);
  void Copy__AF_Ind_Recurrent_Unit(
      AF_Ind_recurrent_unit const
          *const ptr_source_AF_Ind_recurrent_unit_received,
      AF_Ind_recurrent_unit
          *const ptr_destination_AF_Ind_recurrent_unit_received,
      bool const copy_connections_received = true);
  void Copy__Normalized_Batch_Unit(
      size_t const number_units_received,
      Normalized_batch_unit const &ref_source_normalized_batch_unit_received,
      Normalized_batch_unit &ref_destination_normalized_batch_unit_received);
  void Copy__Block(BlockUnit const *const ptr_source_block_unit_received,
                   BlockUnit *const ptr_destination_block_unit_received);
  void Copy__Block__AF(BlockUnit const *const ptr_source_block_unit_received,
                       BlockUnit *const ptr_destination_block_unit_received);
  void Copy__Blocks(size_t const start_index_received,
                    size_t const end_index_received,
                    BlockUnit const *ptr_array_source_block_units_received,
                    bool const copy_connections_received = true);
  void Copy__Blocks__AF(
      size_t const start_index_received, size_t const end_index_received,
      BlockUnit const *ptr_array_source_block_units_received);
  template <class U>
  void Copy__Layer__FC(Layer const *const ptr_source_layer_received,
                       Layer *const ptr_destination_layer_received,
                       U *const ptr_source_first_U_received,
                       U *const ptr_destination_first_U_received,
                       U *const *ptr_source_array_ptr_connections_received,
                       U **ptr_destination_array_ptr_connections_received);
  void Copy__Layer__AF_Ind_Recurrent(
      Layer const *const ptr_source_layer_received,
      AF_Ind_recurrent_unit
          *const ptr_source_first_AF_Ind_recurrent_unit_received,
      AF_Ind_recurrent_unit
          *const ptr_destination_first_AF_Ind_recurrent_unit_received,
      AF_Ind_recurrent_unit *const *ptr_source_array_ptr_connections_received,
      AF_Ind_recurrent_unit **ptr_destination_array_ptr_connections_received);
  template <class U>
  void Copy__Layer__LSTM(Layer const *const ptr_source_layer_received,
                         Layer *const ptr_destination_layer_received,
                         CellUnit *const ptr_source_first_cell_unit_received,
                         U *const ptr_source_first_U_received,
                         U *const ptr_destination_first_U_received,
                         void *const *ptr_source_array_ptr_connections_received,
                         void **ptr_destination_array_ptr_connections_received);
  void Indexing_Regularization_Parameters(void);
  void Indexing_Regularization_Parameters__Pre_training(void);
  void Indexing_Regularization__Weights__FC__Forward(
      real const mask_received, Layer const *const layer_it);
  void Indexing_Regularization__Weights__AF_Ind_Recurrent(
      real const mask_received, Layer const *const layer_it);
  void Indexing_Regularization__Weights__LSTM(real const mask_received,
                                              Layer const *const layer_it);
  void Indexing_Regularization__Bias(real const mask_received,
                                     Layer const *const layer_it);
  void Reset__Parameter__Mask_Dropout(
      bool *ptr_array_units_mask_dropout_bernoulli_received);
  void Reset__Parameters__Cell_Unit__Mask_Dropout(
      bool *ptr_array_cell_units_mask_dropout_received);
  void clear(void);
  void Deallocate(void);
  void Deallocate__Sparse_K_Filter(void);
  void Deallocate__Parameter__Optimizer(void);
  void Deallocate__Parameter__Gradient_Descent(void);
  void Deallocate__Parameter__iRPROP_minus(void);
  void Deallocate__Parameter__iRPROP_plus(void);
  void Deallocate__Parameter__Adam(void);
  void Deallocate__Parameter__AMSGrad(void);
  void Deallocate__Parameter__Regularization(void);
  void Deallocate__Generator__Dropout_Bernoulli(void);
  void Deallocate__Generator__Dropout_Gaussian(void);
  void Deallocate__Generator__Dropout_ShakeDrop(void);
  void Deallocate__Generator__Dropout_Uout(void);
  void Deallocate__Generator__Dropout_Zoneout(void);
  void Deallocate__Neuron__Mask_Dropout_Bernoulli(void);
  void Deallocate__Layer__Mask_Dropout_ShakeDrop(void);
  void Deallocate__Cell_Unit__Mask_Dropout_Zoneout(void);
  void Deallocate__Parameter__Batch_Normalization(void);
  void Deallocate__Normalized_Unit(void);
  void Deallocate__Normalized_Unit__Batch_Normalization(void);
  void Deallocate__Normalized_Unit__Batch_Renormalization(void);
  void Clear_Optimizer(void);
  void Order__Layers__Connection(void);
  void Order__Layers__Output(void);
  void Order__Layer__Output(bool const is_sequentiel_received,
                            Layer *const ptr_layer_received);
  void Order__Layer__Output__Pre_Training(bool const is_sequentiel_received,
                                          Layer *const ptr_layer_received);
  void Order__Layer__Basic(Layer *const layer_it);
  void Order__Layer__Basic_unit(Layer *const layer_it);
  void Order__Layer__Basic_indice(Layer *const layer_it);
  void Order__Layer__Basic_indice_unit(Layer *const layer_it);
  void Order__Layer__Neuron(Layer *const layer_it);
  void Order__Layer__Neuron_Unit(Layer *const layer_it);
  void Order__Layer__AF(Layer *const layer_it);
  void Order__Layer__AF_Unit(Layer *const layer_it);
  void Order__Layer__AF_Unit__Dropout_Bernoulli(Layer *const layer_it);
  void Order__Layer__AF_Ind_Recurrent(Layer *const layer_it);
  void Order__Layer__AF_Ind_Recurrent_Unit(Layer *const layer_it);
  void Order__Layer__AF_Ind_Recurrent_Unit__Dropout_Bernoulli(
      Layer *const layer_it);
  void Order__Layer__LSTM(Layer *const layer_it);
  void Order__Layer__Block_Unit(Layer *const layer_it);
  void Order__Layer__Block_Unit__Dropout_Zoneout(Layer *const layer_it);
  void Order__Layer__Normalization(Layer *const layer_it);
  void Order__Layer__Batch_Normalization(Layer *const layer_it);
  void Order__Layer__Batch_Renormalization(Layer *const layer_it);
  void Reset__Parameter__Normalized_Unit(void);
  void Reset__Derivative_Parameter__Normalized_Unit(void);
  void clear_training_arrays(void);
  void Clear__Parameter__Normalized_Unit(void);
  bool transfer_learning(Model *&ptr_destination_Neural_Network_received) const;
  bool user_controls(void);
  bool User_Controls__Optimizer__Gradient_Descent(void);
  bool User_Controls__Optimizer__iRPROP(void);
  bool User_Controls__Optimizer__AdaBound(void);
  bool User_Controls__Optimizer__Adam(void);
  bool User_Controls__Optimizer__NosAdam(void);
  bool User_Controls__Regularization(void);
  bool User_Controls__Dropout(void);
  bool User_Controls__Normalization(void);
  bool User_Controls__Normalization_Layer(void);
  bool User_Controls__Tied__Parameter(void);
  bool User_Controls__K_Sparse(void);
  bool User_Controls__Weights_Initializer(void);
  bool User_Controls__Optimizer_Function_Initializer(void);
  bool User_Controls__Loss_Function_Initializer(void);
  bool User_Controls__Accuracy_Function_Initializer(void);
  bool User_Controls__Optimizer_Function(void);
  bool User_Controls__Warm_Restarts(void);
  bool User_Controls__Accuracy_Variance(void);
  bool User_Controls__Time_Delays(void);
  bool User_Controls__Clip_Gradient(void);
  bool User_Controls__Max_Norm_Constaints(void);
  bool User_Controls__L1_Regularization(void);
  bool User_Controls__L2_Regularization(void);
  bool User_Controls__SRIP_Regularization(void);
  bool User_Controls__Maximum__Batch_Size(void);
  bool User_Controls__OpenMP(void);
  bool Copy__Optimizer_Parameters(
      Model const *const ptr_Neural_Network_received,
      bool const copy_delta_optimizer_received = false);
  bool Copy__Delta__Gradient_Descent(
      Model const *const ptr_Neural_Network_received);
  bool Copy__Delta__iRPROP_minus(
      Model const *const ptr_Neural_Network_received);
  bool Copy__Delta__iRPROP_plus(Model const *const ptr_Neural_Network_received);
  bool Copy__Delta__Adam(Model const *const ptr_Neural_Network_received);
  bool Copy__Delta__AMSGrad(Model const *const ptr_Neural_Network_received);
  void Copy__Warm_Restarts_Parameters(
      Model const *const ptr_Neural_Network_received);
  void Copy__Gradient_Descent_Parameters(
      Model const *const ptr_Neural_Network_received);
  void Copy__QuickProp_Parameters(
      Model const *const ptr_Neural_Network_received);
  void Copy__RPROP_minus_Parameters(
      Model const *const ptr_Neural_Network_received);
  void Copy__RPROP_plus_Parameters(
      Model const *const ptr_Neural_Network_received);
  void Copy__SARProp_Parameters(Model const *const ptr_Neural_Network_received);
  void Copy__Adam_Parameters(Model const *const ptr_Neural_Network_received);
  void Copy__NosAdam_Parameters(Model const *const ptr_Neural_Network_received);
  void Copy__AdaBound_Parameters(
      Model const *const ptr_Neural_Network_received);
  void Copy__Training_Parameters(
      Model const *const ptr_Neural_Network_received);
  void Copy__Initializer__Weight_Parameter(
      Model const &ref_source_Neural_Network_received);
  void Copy__Regularization(Model const *const ptr_Neural_Network_received);
  void Copy__Tied_Weight(Layer const *ptr_array_source_layers_received,
                         Layer const *const ptr_last_source_layer_received,
                         Layer *ptr_array_destination_layers_received);
  void Copy__Sparse_K_Filters(Layer const *ptr_array_source_layers_received,
                              Layer const *const ptr_last_source_layer_received,
                              Layer *ptr_array_destination_layers_received);
  void Copy__Constraint_Recurrent_Weight(
      Layer const *ptr_array_source_layers_received,
      Layer const *const ptr_last_source_layer_received,
      Layer *ptr_array_destination_layers_received);
  void Copy__Loss(Model const *const ptr_Neural_Network_received);
  void Copy__Accuracy(Model const *const ptr_Neural_Network_received);
  void Copy__Dropout(Layer const *ptr_array_source_layers_received,
                     Layer const *const ptr_last_source_layer_received,
                     Layer *ptr_array_destination_layers_received);
  void Copy__Normalization(Layer const *ptr_array_source_layers_received,
                           Layer const *const ptr_last_source_layer_received,
                           Layer *ptr_array_destination_layers_received);
  void Copy__Normalization(
      Model const *const ptr_source_Neural_Network_received);
  void Copy__Normalized_Units(
      size_t const start_index_received, size_t const end_index_received,
      union Normalized_unit const *ptr_array_source_normalized_units_received);
  template <class U, LAYER::TYPE const E>
  void Layer__Forward__Neuron_Information__Connection(
      std::wstring &out, Neuron_unit const *const unit,
      U const *const first_inp_unit);
  template <class U, LAYER::TYPE const E>
  void Layer__LSTM_Information__Connection(std::wstring &out,
                                           BlockUnit const *const block_it,
                                           U const *const first_inp_unit);

  bool Allocate__Structure(size_t const n_layers,
                           size_t const allowable_memory = 32_UZ * KILOBYTE *
                                                           KILOBYTE);
  bool copy(Model const &ref_source_Neural_Network_received,
            bool const initialize_parallel_computation_received = true,
            bool const copy_delta_optimizer_received = false,
            size_t const allowable_memory = 0_UZ);
  bool Update(Model const &ref_source_Neural_Network_received,
              bool const initialize_parallel_computation_received = false,
              bool const update_delta_optimizer_received = false);
  bool Assign__Layers(Layer_Parameters const *const ptr_array_layers_received);
  bool Assign__Layer(Layer *&layer_it,
                     Layer_Parameters const *const ptr_array_layers_received);
  bool Assign__Residual_Block(
      Layer *&layer_it,
      Layer_Parameters const *const ptr_array_layers_received);
  bool Assign__Residual_Layer(
      Layer *&layer_it,
      Layer_Parameters const *const ptr_array_layers_received);
  bool Assign__Post__Layers(void);
  bool Assign__Post__Layer(Layer *&layer_it);
  bool Assign__Post__Residual_Block(Layer *&layer_it);
  bool Assign__Post__Residual_Layer(bool const is_block_input_layer_received,
                                    Layer *&layer_it);
  bool compile(size_t const n_layers,
               size_t const number_recurrent_depth_received,
               MODEL::TYPE const type_network_received,
               Layer_Parameters const *const ptr_array_layers_received,
               size_t const allowable_memory = 32_UZ * KILOBYTE * KILOBYTE);
  bool Allouable__Batch_Size(size_t const desired_batch_size_received,
                             size_t &ref_batch_size_allouable_received,
                             size_t &ref_number_threads_allouable_received);
  bool Information__Output_Layer(std::wstring &out, Layer const *const layer,
                                 Layer const *const prev_layer);
  bool Information__Layer__AF(std::wstring &out, Layer const *const layer);
  bool Information__Layer__AF_Ind_Recurrent(std::wstring &out,
                                            Layer const *const layer);
  bool Information__Layer__Bias(std::wstring &out, Layer const *const layer);
  bool Information__Layer__Normalization(std::wstring &out,
                                         Layer const *const layer);
  bool Information__Normalized_Unit(size_t const n_units,
                                    LAYER_NORM::TYPE const type,
                                    union Normalized_unit const *const unit,
                                    std::wstring &out);
  bool Information__Layer__FC(std::wstring &out, Layer const *const layer,
                              Layer const *const prev_layer);
  bool Information__Layer__LSTM(std::wstring &out, Layer const *const layer,
                                Layer const *const prev_layer);
  bool Deinitialize__OpenMP(void);
  bool Multi_Class_Classification(void) const;
  bool update_mem_thread_size(size_t const desired_number_threads_received);
  bool update_mem_batch_size(size_t const desired_batch_size_received,
                          bool const force_update_received = false);
  bool Reallocate__Thread(size_t const number_threads_received);
  bool Reallocate__Thread__Sparse_K_Filter(
      size_t const number_threads_received);
  bool Reallocate__Thread__Cost(size_t const number_threads_received);
  bool Reallocate__Thread__Normalized_Unit__Batch_Normalization(
      size_t const number_threads_received);
  bool Reallocate__Thread__Parameter(size_t const number_threads_received);
  bool Reallocate__Thread__Generator__Dropout__Bernoulli(
      size_t const number_threads_received);
  bool Reallocate__Thread__Generator__Dropout__Gaussian(
      size_t const number_threads_received);
  bool Reallocate__Thread__Generator__Dropout__ShakeDrop(
      size_t const number_threads_received);
  bool Reallocate__Thread__Generator__Dropout__Uout(
      size_t const number_threads_received);
  bool Reallocate__Thread__Generator__Dropout__Zoneout(
      size_t const number_threads_received);
  bool Reallocate__Batch(size_t const batch_size);
  bool Reallocate__Batch__Basic_Unit(size_t const batch_size);
  bool Reallocate__Batch__Basic_Indice_Unit(size_t const batch_size);
  bool Reallocate__Batch__Neuron_Unit(size_t const batch_size);
  bool Reallocate__Batch__AF_Unit(size_t const batch_size);
  bool Reallocate__Batch__AF_Ind_Recurrent_Unit(size_t const batch_size);
  bool Reallocate__Batch__LSTM(size_t const batch_size);
  bool Reallocate__Batch__Dropout__ShakeDrop(size_t const batch_size);
  bool Reallocate__Normalized_Unit__Batch_Normalization(
      size_t const batch_size);
  bool Reallocate__Parameter(size_t const number_parameters_received);
  bool Reallocate__Parameter__Regularization(
      size_t const number_parameters_received);
  bool Reallocate__Parameter__Optimizer(
      size_t const number_parameters_received);
  bool Reallocate__Parameter__Gradient_Descent(
      size_t const number_parameters_received);
  bool Reallocate__Parameter__iRPROP_minus(
      size_t const number_parameters_received);
  bool Reallocate__Parameter__iRPROP_plus(
      size_t const number_parameters_received);
  bool Reallocate__Parameter__Adam(size_t const number_parameters_received);
  bool Reallocate__Parameter__AMSGrad(size_t const number_parameters_received);
  bool load(std::wstring const &path_params,
            std::wstring const &path_spec_params,
            size_t const allowable_memory);
  bool Load_Dimension__Neuron(Neuron_unit *const ptr_neuron_received,
                              std::wifstream &file);
  bool Load_Dimension__AF(AF_unit *const ptr_AF_received, std::wifstream &file);
  bool Load_Dimension__AF_Ind_Recurrent(
      AF_Ind_recurrent_unit *const ptr_AF_Ind_received, std::wifstream &file);
  bool Load_Dimension__Normalized_Unit(
      size_t const number_units_received,
      LAYER_NORM::TYPE const type_normalization_received,
      union Normalized_unit *const ptr_normalized_unit_received,
      std::wifstream &file);
  bool Load_Dimension__Block(size_t const layer_number_block_units_received,
                             size_t const layer_number_cell_units_received,
                             LAYER_NORM::TYPE const type_normalization_received,
                             BlockUnit *const ptr_block_received,
                             std::wifstream &file);
  bool Load_Dimension__Cell_Units(
      Layer *const layer_it, CellUnit *&ptr_reference_array_cells_received,
      std::wifstream &file);
  template <class U, LAYER::TYPE const E>
  bool Load_Dimension__Connection(size_t index_received,
                                  var *const ptr_array_parameters_received,
                                  U *const ptr_first_U_unit_received,
                                  U **ptr_array_ptr_U_unit_connection_received,
                                  std::wifstream &file);
  template <class U, LAYER::TYPE const E>
  bool Load_Dimension__Neuron__Forward__Connection(
      Neuron_unit *const ptr_neuron_received,
      U *const ptr_first_U_unit_received, std::wifstream &file);
  template <class U, LAYER::TYPE const E>
  bool Load_Dimension__Block__Connection(
      BlockUnit *const ptr_block_unit_it_received,
      U *const ptr_first_U_unit_received, std::wifstream &file);
  template <class U, LAYER::TYPE const E>
  bool Load_Dimension__FC(Layer *const layer_it,
                          U *const ptr_first_U_unit_received,
                          std::wifstream &file);
  bool Load_Dimension__AF(Layer *const layer_it, std::wifstream &file);
  bool Load_Dimension__AF_Ind_Recurrent(Layer *const layer_it,
                                        std::wifstream &file);
  template <class U, LAYER::TYPE const E>
  bool Load_Dimension__LSTM(Layer *const layer_it,
                            U *const ptr_first_U_unit_received,
                            std::wifstream &file);
  bool Load_Dimension__Normalization(Layer *const layer_it,
                                     std::wifstream &file);
  bool Load_Dimension__Bias(Layer *const layer_it, std::wifstream &file);
  bool load_spec_params(std::wstring const &path_name);
  bool save_spec_params(std::wstring const &path_name);
  bool save_params(std::wstring const &path_name);
  bool Allocate__Sparse_K_Filter(void);
  bool Allocate__Parameter(void);
  bool Allocate__Parameter__Optimizer(void);
  bool Allocate__Parameter__Gradient_Descent(void);
  bool Allocate__Parameter__iRPROP_minus(void);
  bool Allocate__Parameter__iRPROP_plus(void);
  bool Allocate__Parameter__Adam(void);
  bool Allocate__Parameter__AMSGrad(void);
  bool Allocate__Parameter__Regularization(void);
  /* arguments:
      reallocate_received: Use in the load from a file function. */
  bool Allocate__Parameter__Normalization(void);
  bool Allocate__Basic_Units(void);
  bool Allocate__Basic_Indice_Units(void);
  bool Allocate__Neuron_Units(void);
  bool Allocate__Neuron__Mask_Dropout_Bernoulli(void);
  bool Allocate__Layer__Mask__Dropout__ShakeDrop(void);
  bool Allocate__AF_Units(void);
  bool Allocate__AF_Ind_Recurrent_Units(void);
  bool Allocate__Normalized_Unit(bool const organize_pointers_received);
  bool Allocate__Normalized_Unit__Batch_Normalization(void);
  bool Allocate__Normalized_Unit__Batch_Renormalization(void);
  bool Allocate__Block_Unit__Mask_Dropout_Zoneout(void);
  bool Allocate__LSTM_Layers(void);
  bool Allocate__Bidirectional__Layers(void);
  bool Allocate__Generator__Dropout_Bernoulli(void);
  bool Allocate__Generator__Dropout_Gaussian(void);
  bool Allocate__Generator__Dropout_ShakeDrop(void);
  bool Allocate__Generator__Dropout_Uout(void);
  bool Allocate__Generator__Dropout_Zoneout(void);
  bool weights_initialized(void) const;
  bool initialize_weights(DatasetV1 const *const ptr_Dataset_received);
  bool Set__Pre_Training_Level(size_t const pre_training_level_received);
  bool set_max_batch_size(size_t const max_batch_size);
  bool set_mp(bool const use_openmp_received);
  bool Set__Maximum_Thread_Usage(
      double const percentage_maximum_thread_usage_received);
  bool Set__Accurancy_Variance(real const accurancy_variance_received);
  bool set_seq_w(size_t const time_delays_received);
  bool set_dropout(size_t const index_layer_received,
                   LAYER_DROPOUT::TYPE const type_layer_dropout_received,
                   real const value_dropout_received[],
                   bool const scale_weights_received = true);
  bool set_dropout(Layer *const ptr_layer_received,
                   LAYER_DROPOUT::TYPE const type_layer_dropout_received,
                   real const value_dropout_received[],
                   bool const scale_weights_received = true);
  bool Set__Dropout_None(Layer *const ptr_layer_received);
  bool Set__Dropout_Alpha(Layer *const ptr_layer_received,
                          real const dropout_probability_received);
  bool Set__Dropout_Bernoulli(Layer *const ptr_layer_received,
                              real const retention_probability_received,
                              bool const scale_weights_received = true);
  bool Set__Dropout_Bernoulli_Inverted(
      Layer *const ptr_layer_received,
      real const retention_probability_received);
  bool Set__Dropout_Gaussian(Layer *const ptr_layer_received,
                             real const dropout_probability_received);
  bool Set__Dropout_ShakeDrop(Layer *const ptr_layer_received,
                              real const dropout_probability_received);
  bool Set__Dropout_Uout(Layer *const ptr_layer_received,
                         real const dropout_probability_received);
  bool Set__Dropout_Zoneout(Layer *const ptr_layer_received,
                            real const zoneout_cell_received,
                            real const zoneout_hidden_received);
  void Scale_Weight__Dropout(real const scale_factor_received,
                             Layer const *const layer_it);
  void Scale_Weight__FC__Forward__Dropout(real const scale_factor_received,
                                          Layer const *const layer_it);
  void Scale_Weight__FC__Recurrent__Dropout(real const scale_factor_received,
                                            Layer const *const layer_it);
  bool Prepare__Normalized__Layers(void);
  bool Prepare__Normalized__Layer(Layer *&layer_it);
  bool Prepare__Normalized__Residual_Block(Layer *&layer_it);
  bool Prepare__Normalized__Residual_Layer(Layer *&layer_it);
  /* arguments:
      reallocate: When loading from a file this value should be set to false.
      organize_pointers: When loading from a file this value should be set to
     false. */
  bool Set__Layer_Normalization(
      size_t const index_layer_received,
      LAYER_NORM::TYPE const type_layer_normalization_received,
      bool const reallocate_dimension_parameters_received = true,
      bool const organize_pointers_received = true);
  bool Set__Layer_Normalization(
      Layer *const ptr_layer_received,
      LAYER_NORM::TYPE const type_layer_normalization_received,
      bool const reallocate_dimension_parameters_received = true,
      bool const organize_pointers_received = true);
  bool Set__Normalization_None(Layer *const ptr_layer_received,
                               bool const organize_pointers_received);
  bool Set__Batch_Normalization(
      Layer *const ptr_layer_received,
      bool const use_batch_normalization_received = true,
      bool const reallocate_dimension_parameters_received = true,
      bool const organize_pointers_received = true);
  bool Set__Batch_Renormalization(
      Layer *const ptr_layer_received,
      bool const use_batch_renormalization_received = true,
      bool const reallocate_dimension_parameters_received = true,
      bool const organize_pointers_received = true);
  bool Set__Ghost_Batch_Normalization(
      Layer *const ptr_layer_received,
      bool const use_ghost_batch_normalization_received = true,
      bool const reallocate_dimension_parameters_received = true,
      bool const organize_pointers_received = true);
  bool set_clip_gradient(real const clip_gradient);
  bool Check__Use__Regularization__Constraint_Recurrent_Weight__Default(
      size_t const index_layer_received) const;
  bool Check__Use__Regularization__Constraint_Recurrent_Weight__Default(
      Layer *const ptr_layer_received) const;
  bool Set__Regularization__Constraint_Recurrent_Weight__Default(
      size_t const index_layer_received);
  bool Set__Regularization__Constraint_Recurrent_Weight__Default(
      Layer *const ptr_layer_received);
  bool Set__Regularization__Constraint_Recurrent_Weight(
      size_t const index_layer_received,
      real const constraint_recurrent_weight_lower_bound_received,
      real const constraint_recurrent_weight_upper_bound_received);
  bool Set__Regularization__Constraint_Recurrent_Weight(
      Layer *const ptr_layer_received,
      real const constraint_recurrent_weight_lower_bound_received,
      real const constraint_recurrent_weight_upper_bound_received);
  bool Set__Tied_Parameter(size_t const index_layer_received,
                           bool const use_tied_parameter_received,
                           bool const transpose_received = true);
  bool Set__Tied_Parameter(Layer *const ptr_layer_received,
                           bool const use_tied_parameter_received,
                           bool const transpose_received = true);
  // TODO: Backpropagate toward the K largest activation function (basic indice
  // unit).
  bool Set__K_Sparsity(size_t const index_layer_received,
                       size_t const k_sparsity_received);
  bool Set__K_Sparsity(Layer *const ptr_layer_received,
                       size_t const k_sparsity_received);
  bool Set__Alpha_Sparsity(size_t const index_layer_received,
                           real const alpha_sparsity_received);
  bool Set__Alpha_Sparsity(Layer *const ptr_layer_received,
                           real const alpha_sparsity_received);
  bool set_l1(real const val);
  bool set_l2(real const val);
  bool set_srip(real const val);
  bool set_weight_decay(real const val);
  bool Set__Regularization__Max_Norm_Constraints(
      real const regularization__max_norm_constraints_received);
  bool Set__Normalization_Momentum_Average(real const momentum_average_received);
  bool Set__Normalization_Epsilon(real const epsilon_received);
  bool Set__Batch_Renormalization_r_Correction_Maximum(
      real const r_correction_maximum_received);
  bool Set__Batch_Renormalization_d_Correction_Maximum(
      real const d_correction_maximum_received);
  bool set_layer_activation_function(
      size_t const index_layer_received,
      ACTIVATION::TYPE const type_activation_function_received);
  bool set_layer_activation_function(
      Layer *const layer_it,
      ACTIVATION::TYPE const type_activation_function_received);
  bool Set__Multi_Label(bool const use_multi_label_received);
  bool Set__Input_Mode(bool const use_first_layer_as_input_received);
  bool Set__Output_Mode(bool const use_last_layer_as_output_received);
  bool Use__Clip_Gradient(void) const;
  bool Use__Regularization_Parameter(void) const;
  bool Use__Normalization(void) const;
  bool Use__Batch_Normalization(void) const;
  bool Use__Batch_Renormalization(void) const;
  bool Use__Ghost_Batch_Normalization(void) const;
  bool Use__Streaming_Normalization(void) const;
  bool Use__Dropout__Alpha(void) const;
  bool Use__Dropout__Bernoulli(void) const;
  bool Use__Dropout__Bernoulli__Inverted(void) const;
  bool Use__Dropout__Gaussian(void) const;
  bool Use__Dropout__ShakeDrop(void) const;
  bool Use__Dropout__Uout(void) const;
  bool Use__Dropout__Zoneout(void) const;
  bool Use__K_Sparse(void) const;
  bool Use__Tied_Parameter(void) const;
  bool Use__Regularization__Constraint_Recurrent_Weight(void) const;
  bool Use__Multi_Label(void) const;
  bool Usable_Warm_Restarts(void) const;
  bool Compare(bool const use_metric_loss_received,
               bool const dataset_in_equal_less_dataset_out_accepted_received,
               ENV::TYPE const type_holdout_dataset_received,
               double const minimum_loss_holdout_dataset_accepted_received,
               Model const *const ptr_source_Neural_Network_received) const;
  bool *ptr_array_units_mask_dropout_bernoulli = nullptr;   // size[H].
  bool *ptr_array_layers_mask_dropout_shakedrop = nullptr;  // size[L, T, B].
  bool *ptr_array_cell_units_mask_dropout_zoneout = nullptr;
  bool use_mp = false;
  bool is_mp_initialized = false;
  bool use_warm_restarts = false;
  bool use_nesterov = false;
  bool use_normalized_weight_decay = true;
  bool use_adam_bias_correction = true;
  bool use_multi_label = false;
  bool use_clip_gradient = false;
  /* Use the first layer as input.
      Default:
      - Always true.
      Autoencoder:
      - true: Feed inputs into the input layer.
      - false: Feed inputs into the decoded layer. */
  bool use_first_layer_as_input = true;
  /* Use the last layer as output.
      Default:
      - Always true.
      Autoencoder:
      - true: Reconstruct the inputs as output(s).
      - false: Compress the inputs as output(s). */
  bool use_last_layer_as_output = true;

  std::pair<size_t, var> *ptr_array_k_sparse_activities = nullptr;

  size_t Prepare__Connections__FC(size_t const input_size_received,
                                  Layer *const layer_it);
  size_t Prepare__Connections__FC_Ind_RNN(size_t const input_size_received,
                                          Layer *const layer_it);
  size_t Prepare__Connections__LSTM(size_t const input_size_received,
                                    Layer *const layer_it);
  size_t Prepare__Bias__FC(size_t const shift_index_received,
                           Layer *const layer_it);
  size_t Prepare__Bias__LSTM(size_t const shift_index_received,
                             Layer *const layer_it);
  size_t Get__Total_Layers(void) const;
  size_t *ptr_array_number_loss = nullptr;      // size[N].
  size_t *ptr_array_number_bit_fail = nullptr;  // size[N].
  size_t n_acc_trial = 0_UZ;
  size_t number_threads = 1_UZ;
  size_t cache_number_threads = 1_UZ;
  size_t batch_size = 1_UZ;
  size_t cache_batch_size = 1_UZ;
  size_t maximum_batch_size = (std::numeric_limits<size_t>::max)();
  size_t total_basic_units = 0_UZ;
  size_t total_basic_units_allocated = 0_UZ;
  size_t total_basic_indice_units = 0_UZ;
  size_t total_basic_indice_units_allocated = 0_UZ;
  size_t total_neuron_units = 0_UZ;
  size_t total_neuron_units_allocated = 0_UZ;
  size_t total_AF_units = 0_UZ;
  size_t total_AF_units_allocated = 0_UZ;
  size_t total_AF_Ind_recurrent_units = 0_UZ;
  size_t total_AF_Ind_recurrent_units_allocated = 0_UZ;
  size_t total_block_units = 0_UZ;
  size_t total_block_units_allocated = 0_UZ;
  size_t total_cell_units = 0_UZ;
  size_t total_cell_units_allocated = 0_UZ;
  size_t total_normalized_units = 0_UZ;
  size_t total_normalized_units_allocated = 0_UZ;
  size_t total_parameters = 0_UZ;
  size_t total_parameters_allocated = 0_UZ;
  size_t total_weights = 0_UZ;
  size_t total_weights_allocated = 0_UZ;
  size_t total_bias = 0_UZ;
  size_t total_bias_allocated = 0_UZ;
  size_t n_inp = 0_UZ;
  size_t n_out = 0_UZ;
  size_t seq_w = 1_UZ;
  size_t n_time_delay = 0_UZ;
  size_t pre_training_level = 0_UZ;
  size_t *ptr_array_basic_indice_units_indices = nullptr;   // size[B, T, H].
  size_t *ptr_array_number_neurons_by_layer = nullptr;      // size[L].
  size_t *ptr_array_number_connections_by_layer = nullptr;  // size[L].
  size_t *ptr_array_neuron_units_first_forward_connection_index =
      nullptr;  // size[H].
  size_t *ptr_array_neuron_units_last_forward_connection_index =
      nullptr;  // size[H].
  size_t *ptr_array_neuron_units_number_forward_connections =
      nullptr;  // size[H].
  size_t *ptr_array_AF_Ind_recurrent_units_recurrent_connection_index =
      nullptr;                                                // size[H].
  size_t *ptr_array_layers_number_outputs = nullptr;          // size[L].
  size_t *ptr_array_layers_first_connection_index = nullptr;  // size[L].
  size_t *ptr_array_layers_last_connection_index = nullptr;   // size[L].
  size_t total_layers = 0_UZ;
  size_t total_batch_normalization_layers = 0_UZ;
  size_t total_batch_renormalization_layers = 0_UZ;
  size_t total_ghost_batch_normalization_layers = 0_UZ;
  size_t total_streaming_normalization_layers = 0_UZ;
  size_t total_dropout_alpha_layers = 0_UZ;
  size_t total_dropout_bernoulli_layers = 0_UZ;
  size_t total_dropout_bernoulli_inverted_layers = 0_UZ;
  size_t total_dropout_gaussian_layers = 0_UZ;
  size_t total_dropout_shakedrop_layers = 0_UZ;
  size_t total_dropout_uout_layers = 0_UZ;
  size_t total_dropout_zoneout_layers = 0_UZ;
  size_t total_k_sparse_layers = 0_UZ;
  size_t total_tied_parameter_layers = 0_UZ;
  size_t total_constraint_recurrent_weight_layers = 0_UZ;

  double get_accu(ENV::TYPE const env) const;
  double get_loss(ENV::TYPE const env) const;
  double get_me(void) const;
  double get_loss_l1(void) const;
  double get_loss_l2(void) const;
  double get_mae(void) const;
  double get_mse(void) const;
  double get_rmse(void) const;
  double get_mape(void) const;
  double get_smape(void) const;
  double get_mase(void) const;
  double get_ace(void) const;
  double get_bitfail(void) const;
  double pct_threads = 100.0;
  double pct_threads_cached = 0.0;
  double *ptr_array_loss_values = nullptr;           // size[N].
  double *ptr_array_accuracy_values[5] = {nullptr};  // size[N].
  double loss_train = HUGE_VAL;
  double loss_valid = HUGE_VAL;
  double loss_testg = HUGE_VAL;
  double loss_rprop = HUGE_VAL;
  double loss_rprop_tm1 = HUGE_VAL;
  double acc_train = 0.0;
  double acc_valid = 0.0;
  double acc_testg = 0.0;
  double bit_fail_limit = 0.35;

  real warm_restarts_decay(void);
  real normalized_wd(size_t const batch_size,
                     size_t const training_size);
  real Get__Regularization__Max_Norm_Constraints(void) const;
  real get_l1(void) const;
  real get_l2(void) const;
  real get_srip(void) const;
  var Activation_Function(ACTIVATION::TYPE const type, var x);
  real activation_fn_derivative(ACTIVATION::TYPE const type, real const x,
                                real const q, real const y = 0_r);
  real Initialization__Gain__Scale(ACTIVATION::TYPE const type);
  real Initialization__Gaussian__Variance(size_t const n_inp,
                                          size_t const n_out,
                                          LAYER_ACTIVATION::TYPE const type);
  real Initialization__Uniform__Variance(size_t const n_inp, size_t const n_out,
                                         LAYER_ACTIVATION::TYPE const type);
  void forward_pass(size_t const batch_size,
                    real const *const *const ptr_array_inputs_received,
                    long long int input_layer_index_received = -1ll,
                    long long int output_layer_index_received = -1ll);
  void Forward_Pass__Pre_Training(
      size_t const batch_size,
      real const *const *const ptr_array_inputs_received);
  void assign_inputs_fwp_st(size_t const batch_size,
                            real const *const *const Xm);
  void assign_inputs_fwp_mp(size_t const batch_size,
                            real const *const *const Xm);
  void assign_inputs_rec_st(size_t const batch_size,
                            real const *const *const Xm);
  void assign_inputs_rec_mp(size_t const batch_size,
                            real const *const *const Xm);
  void assign_inputs_pre_train_fwp_st(size_t const batch_size,
                                      real const *const *const Xm);
  void assign_inputs_pre_train_fwp_mp(size_t const batch_size,
                                      real const *const *const Xm);
  void assign_inputs_pre_train_rec_st(size_t const batch_size,
                                      real const *const *const Xm);
  void assign_inputs_pre_train_rec_mp(size_t const batch_size,
                                      real const *const *const Xm);
  void Clear_Outputs(void);
  std::pair<real, real>
  Compute__Regularization__Constraint_Recurrent_Weight__Default(
      size_t const index_layer_received) const;
  std::pair<real, real>
  Compute__Regularization__Constraint_Recurrent_Weight__Default(
      Layer *const ptr_layer_received) const;
  var const *get_out(size_t const data_index_received,
                     size_t const time_step_index_received = 0_UZ) const;
  var const *get_out(Layer const *const layer_it,
                     size_t const data_index_received,
                     size_t const time_step_index_received = 0_UZ) const;
  real get_layer_variance(size_t const layer_index_received,
                             size_t const max_batch_size) const;
  real get_layer_variance(Layer const *const ptr_layer_received,
                             size_t const max_batch_size) const;
  real acc_var = 0.49_r;
  real learning_rate = 0.01_r;
  real learning_rate_final = 0.1_r;
  real learning_momentum = 0.9_r;
  real learning_gamma = 1e-3_r;
  real regularization__max_norm_constraints = 0_r;
  real regularization__l1 = 0_r;
  real regularization__l2 = 0_r;
  real regularization__srip = 0_r;
  real weight_decay = 0_r;
  real adam_learning_rate = 0.001_r;
  real adam_beta1 = 0.9_r;
  real adam_beta2 = 0.999_r;  // {0.99, 0.999}
  real adam_previous_beta2 = 0_r;
  real adam_epsilon = 1e-8_r;
  real adam_gamma = 0.1_r;  // {0.05, 0.1}
  real optimizer_time_step = 0_r;
  real epoch_time_step = 1_r;
  real warm_restarts_decay_learning_rate = 1_r;
  real warm_restarts_initial_maximum_learning_rate = 1_r;
  real warm_restarts_maximum_learning_rate = 1_r;
  real warm_restarts_minimum_learning_rate = 1e-7_r;
  real warm_restarts_initial_T_i = 1_r;
  real warm_restarts_T_i = 1_r;
  real warm_restarts_multiplier = 2_r;
  real clip_gradient = 1_r;
  real normalization_momentum_average = 0.999_r;
  real normalization_epsilon = 1e-5_r;
  real batch_renormalization_r_correction_maximum = 1_r;
  real batch_renormalization_d_correction_maximum = 0_r;
  var *ptr_array_basic_indice_units_values = nullptr;      // size[B, T, H].
  real *ptr_array_basic_indice_units_errors = nullptr;      // size[B, T, H].
  var *ptr_array_basic_units_values = nullptr;             // size[B, T, H].
  real *ptr_array_basic_units_errors = nullptr;             // size[B, T, H].
  var *ptr_array_neuron_units_summations = nullptr;        // size[B, T, H].
  real *ptr_array_neuron_units_errors = nullptr;            // size[B, T, H].
  var *ptr_array_AF_units_values = nullptr;                // size[B, T, H].
  real *ptr_array_AF_units_errors = nullptr;                // size[B, T, H].
  var *ptr_array_AF_Ind_recurrent_units_pre_AFs = nullptr;  // size[B, T, H].
  var *ptr_array_AF_Ind_recurrent_units_AFs = nullptr;      // size[B, T, H].
  real *ptr_array_AF_Ind_recurrent_units_errors = nullptr;   // size[B, T, H].
  real *ptr_array_AF_Ind_recurrent_units_dAFs = nullptr;     // size[B, T, H].
  var *ptr_array_cells_summations_cells_inputs = nullptr;   // size[B, T, H].
  var *ptr_array_cells_summations_input_cells_inputs =
      nullptr;  // size[B, T, H].
  var *ptr_array_cells_summations_recurrent_cells_inputs =
      nullptr;                                              // size[B, T, H].
  var *ptr_array_blocks_summations_inputs_gates = nullptr;  // size[B, T, H].
  var *ptr_array_blocks_summations_input_inputs_gates =
      nullptr;  // size[B, T, H].
  var *ptr_array_blocks_summations_recurrent_inputs_gates =
      nullptr;                                               // size[B, T, H].
  var *ptr_array_blocks_summations_forgets_gates = nullptr;  // size[B, T, H].
  var *ptr_array_blocks_summations_input_forgets_gates =
      nullptr;  // size[B, T, H].
  var *ptr_array_blocks_summations_recurrent_forgets_gates =
      nullptr;                                               // size[B, T, H].
  var *ptr_array_blocks_summations_outputs_gates = nullptr;  // size[B, T, H].
  var *ptr_array_blocks_summations_input_outputs_gates =
      nullptr;  // size[B, T, H].
  var *ptr_array_blocks_summations_recurrent_outputs_gates =
      nullptr;                                               // size[B, T, H].
  var *ptr_array_cells_inputs = nullptr;                     // size[B, T, H].
  var *ptr_array_cells_states = nullptr;                     // size[B, T, H].
  var *ptr_array_cells_states_activates = nullptr;           // size[B, T, H].
  var *ptr_array_cells_outputs = nullptr;                    // size[B, T, H].
  var *ptr_array_blocks_inputs_gates = nullptr;              // size[B, T, H].
  var *ptr_array_blocks_forgets_gates = nullptr;             // size[B, T, H].
  var *ptr_array_blocks_outputs_gates = nullptr;             // size[B, T, H].
  real *ptr_array_cells_delta_inputs = nullptr;               // size[B, T, H].
  real *ptr_array_cells_delta_input_inputs = nullptr;         // size[B, T, H].
  real *ptr_array_cells_delta_recurrent_inputs = nullptr;     // size[B, T, H].
  real *ptr_array_cells_delta_states = nullptr;               // size[B, T, H].
  real *ptr_array_cells_delta_outputs = nullptr;              // size[B, T, H].
  real *ptr_array_blocks_delta_inputs_gates = nullptr;        // size[B, T, H].
  real *ptr_array_blocks_delta_input_inputs_gates = nullptr;  // size[B, T, H].
  real *ptr_array_blocks_delta_recurrent_inputs_gates =
      nullptr;                                                // size[B, T, H].
  real *ptr_array_blocks_delta_forgets_gates = nullptr;        // size[B, T, H].
  real *ptr_array_blocks_delta_input_forgets_gates = nullptr;  // size[B, T, H].
  real *ptr_array_blocks_delta_recurrent_forgets_gates =
      nullptr;                                                // size[B, T, H].
  real *ptr_array_blocks_delta_outputs_gates = nullptr;        // size[B, T, H].
  real *ptr_array_blocks_delta_input_outputs_gates = nullptr;  // size[B, T, H].
  real *ptr_array_blocks_delta_recurrent_outputs_gates =
      nullptr;                                                 // size[B, T, H].
  real *ptr_array_derivatives_parameters = nullptr;             // size[N, P].
  real *ptr_array_previous_steps = nullptr;                     // size[P].
  real *ptr_array_previous_delta_parameters = nullptr;          // size[P].
  real *ptr_array_previous_derivatives_parameters = nullptr;    // size[P].
  var *ptr_array_parameters = nullptr;                         // size[P].
  real *ptr_array_mask_regularized_parameters = nullptr;       // size[P].
  real *ptr_array_previous_biased_first_moment = nullptr;       // size[P].
  real *ptr_array_previous_biased_second_moment = nullptr;      // size[P].
  real *ptr_array_previous_biased_second_moment_hat = nullptr;  // size[P].
  var *ptr_array_normalized_batch_units_values_hats =
      nullptr;  // size[B, T, H]. Batch normalization variable.
  var *ptr_array_normalized_batch_units_values_normalizes =
      nullptr;  // size[B, T, H]. Batch normalization variable.
  var *ptr_array_normalized_batch_units_scales =
      nullptr;  // size[H]. Batch normalization variable.
  var *ptr_array_normalized_batch_units_shifts =
      nullptr;  // size[H]. Batch normalization variable.
  real *ptr_array_normalized_batch_units_derivatives_scales =
      nullptr;  // size[N, H]. Batch normalization variable.
  real *ptr_array_normalized_batch_units_derivatives_shifts =
      nullptr;  // size[N, H]. Batch normalization variable.
  var *ptr_array_normalized_batch_units_means =
      nullptr;  // size[N, T, H]. Batch normalization variable.
  var *ptr_array_normalized_batch_units_variances =
      nullptr;  // size[N, T, H]. Batch normalization variable.
  real *ptr_array_normalized_batch_units_derivatives_means =
      nullptr;  // size[N, T, H]. Batch normalization variable.
  real *ptr_array_normalized_batch_units_derivatives_variances =
      nullptr;  // size[N, T, H]. Batch normalization variable.
  var *ptr_array_normalized_batch_units_r_corrections =
      nullptr;  // size[T, H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_d_corrections =
      nullptr;  // size[T, H]. Batch renormalization variable.
  var *ptr_array_normalized_batch_units_means_averages =
      nullptr;  // size[T, H]. Batch normalization variable.
  var *ptr_array_normalized_batch_units_variances_averages =
      nullptr;  // size[T, H]. Batch normalization variable.
  real *ptr_array_normalized_batch_units_errors =
      nullptr;  // size[B, T, H]. Batch normalization variable.

  ACTIVATION::TYPE *ptr_array_AF_units_type_activation_function =
      nullptr;  // size[H].
  ACTIVATION::TYPE *ptr_array_AF_Ind_recurrent_units_type_activation_function =
      nullptr;  // size[H].
  MODEL::TYPE type = MODEL::NONE;
  OPTIMIZER::TYPE type_optimizer_function = OPTIMIZER::NONE;
  LOSS_FN::TYPE type_loss_function = LOSS_FN::NONE;
  ACCU_FN::TYPE type_accuracy_function = ACCU_FN::DISTANCE;
  PROPAGATION::TYPE type_state_propagation =
      PROPAGATION::INFERENCE;  // Dropout && Batch normalization
                               // variable
  LAYER_ACTIVATION::TYPE Activation_Function__To__Class_Activation_Function(
      ACTIVATION::TYPE const type_activation_function_received) const;

  Dist::Bernoulli *bernoulli = nullptr;
  Dist::Bernoulli *ptr_array_Class_Generator_Bernoulli_ShakeDrop = nullptr;
  Dist::Bernoulli *ptr_array_Class_Generator_Bernoulli_Zoneout_State = nullptr;
  Dist::Bernoulli *ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden = nullptr;
  Dist::Real *ptr_array_Class_Generator_Real_ShakeDrop = nullptr;
  Dist::Real *ptr_array_Class_Generator_Real_Uout = nullptr;
  Dist::Real real_gen;
  Dist::Gaussian *ptr_array_Class_Generator_Real_Gaussian = nullptr;
  Dist::Gaussian gaussian;

  bool Initialize__LSUV(size_t const max_n_trials = 10_UZ,
                        size_t const max_batch_size = 32_UZ,
                        real const bias = 0_r,
                        real const var_target = 1_r,
                        real const var_tolerance = 0.01_r);

  void Initialization__Glorot__Gaussian(real const bias = 0_r);
  void initialize_weights_with_glorot_uniform(real const bias = 0_r);
  void Initialization__Identity(real const bias = 0_r);
  bool Initialization__LSUV(DatasetV1 const *const ptr_Dataset_received);
  bool Initialization__LSUV__Loop(DatasetV1 const *const ptr_Dataset_received);
  bool Initialization__LSUV__OpenMP(DatasetV1 const *const ptr_Dataset_received);
  void layers_initialize_orthogonal(bool const pre_initialize_received = false,
                                  real const bias = 0_r);
  void Initialization__Uniform(real const bias = 0_r,
                               real const lower_bound = -1.0_r,
                               real const upper_bound = 1_r);
  void **ptr_array_ptr_connections = nullptr;

  Layer const *Get__Layer(size_t const index_received) const;
  Layer const *Get__End_Layer__Active(void) const;
  Layer *ptr_array_layers = nullptr;
  Layer *ptr_last_layer = nullptr;

  Bidirectional_Layer *ptr_array_bidirectional_layers = nullptr;
  Bidirectional_Layer *ptr_last_bidirectional_layer = nullptr;

  Basic_unit *ptr_array_basic_units = nullptr;
  Basic_unit *ptr_last_basic_unit = nullptr;

  Basic_indice_unit *ptr_array_basic_indice_units = nullptr;
  Basic_indice_unit *ptr_last_basic_indice_unit = nullptr;

  Neuron_unit *ptr_array_neuron_units = nullptr;
  Neuron_unit *ptr_last_neuron_unit = nullptr;

  AF_unit *ptr_array_AF_units = nullptr;
  AF_unit *ptr_last_AF_unit = nullptr;

  AF_Ind_recurrent_unit *ptr_array_AF_Ind_recurrent_units = nullptr;
  AF_Ind_recurrent_unit *ptr_last_AF_Ind_recurrent_unit = nullptr;

  union Normalized_unit *ptr_array_normalized_units = nullptr;
  union Normalized_unit *ptr_last_normalized_unit = nullptr;

  BlockUnit *ptr_array_block_units = nullptr;
  BlockUnit *ptr_last_block_unit = nullptr;

  CellUnit *ptr_array_cell_units = nullptr;
  CellUnit *ptr_last_cell_unit = nullptr;

  bool Use__CUDA(void) const;
  bool set_cu(bool const use_cuda, size_t const allowable_memory);
  bool Initialize__CUDA(size_t const allowable_memory);
  bool Initialize__CUDA__Thread(Datasets const *const datasets);
  bool Deinitialize__CUDA(void);

  bool is_update_from_device = true;
  bool is_cu_initialized = false;
  bool use_cu = false;

#ifdef COMPILE_CUDA
  void Clear_Training_Arrays__CUDA(void);
  void Copy__Parameters__Host_To_Device(void);

  bool Copy_Device_To_Host(bool const refresh_from_genetic_algorithm_received);
  bool Copy__Parameters__Device_To_Host(void);
  bool Copy__Optimizer_Paramaters__Device_To_Host(void);
  bool Copy__Optimizer_Gradient_Descent__Device_To_Host(void);
  bool Copy__Optimizer_RPROP_minus__Device_To_Host(void);
  bool Copy__Optimizer_RPROP_plus__Device_To_Host(void);
  bool Copy__Optimizer_Adam__Device_To_Host(void);
  bool Copy__Optimizer_AMSGrad__Device_To_Host(void);
  bool Copy__Batch_Normalization_Neurons__Device_To_Host(void);
  template <typename T>
  bool Copy__Optimizer_Gradient_Descent__Device_To_Host(
      T &ref_optimizer_time_step_received,
      T &ref_warm_maximum_learning_rate_received, T &ref_warm_T_i_received,
      T *const ptr_array_previous_delta_parameters_received) const;
  template <typename T>
  bool Copy__Optimizer_RPROP_minus__Device_To_Host(
      T *const ptr_array_previous_steps_received,
      T *const ptr_array_previous_derivates_parameters_received) const;
  template <typename T>
  bool Copy__Optimizer_RPROP_plus__Device_To_Host(
      T &ref_loss_received, T &ref_previous_loss_received,
      T *const ptr_array_previous_steps_received,
      T *const ptr_array_previous_derivates_parameters_received,
      T *const ptr_array_previous_delta_parameters_received) const;
  template <typename T>
  bool Copy__Optimizer_Adam__Device_To_Host(
      T &ref_optimizer_time_step_received,
      T &ref_warm_maximum_learning_rate_received, T &ref_warm_T_i_received,
      T *const ptr_array_previous_biased_first_moment_received,
      T *const ptr_array_previous_biased_second_moment_received) const;
  template <typename T>
  bool Copy__Optimizer_AMSGrad__Device_To_Host(
      T &ref_optimizer_time_step_received,
      T &ref_warm_maximum_learning_rate_received, T &ref_warm_T_i_received,
      T *const ptr_array_previous_biased_first_moment_received,
      T *const ptr_array_previous_biased_second_moment_received,
      T *const ptr_array_previous_biased_second_moment_hat_received) const;
  template <typename T>
  bool Copy__Batch_Normalization_Neurons__Device_To_Host(
      T *const ptr_array_neuron_units_scale_received,
      T *const ptr_array_neuron_units_shift_received,
      T *const ptr_array_neuron_units_mean_average_received,
      T *const ptr_array_neuron_units_variance_average_received) const;
  template <typename T>
  bool Copy__Optimizer_Gradient_Descent__Host_To_Device(
      T const optimizer_time_step_received,
      T const warm_restarts_maximum_learning_rate_received,
      T const warm_restarts_T_i_received,
      T const *const ptr_array_previous_delta_parameters_received);
  template <typename T>
  bool Copy__Optimizer_RPROP_minus__Host_To_Device(
      T const *const ptr_array_previous_steps_received,
      T const *const ptr_array_previous_derivates_parameters_received);
  template <typename T>
  bool Copy__Optimizer_RPROP_plus__Host_To_Device(
      T const loss, T const previous_loss_received,
      T const *const ptr_array_previous_steps_received,
      T const *const ptr_array_previous_derivates_parameters_received,
      T const *const ptr_array_previous_delta_parameters_received);
  template <typename T>
  bool Copy__Optimizer_Adam__Host_To_Device(
      T const optimizer_time_step_received,
      T const warm_restarts_maximum_learning_rate_received,
      T const warm_restarts_T_i_received,
      T const *const ptr_array_previous_biased_first_moment_received,
      T const *const ptr_array_previous_biased_second_moment_received);
  template <typename T>
  bool Copy__Optimizer_AMSGrad__Host_To_Device(
      T const optimizer_time_step_received,
      T const warm_restarts_maximum_learning_rate_received,
      T const warm_restarts_T_i_received,
      T const *const ptr_array_previous_biased_first_moment_received,
      T const *const ptr_array_previous_biased_second_moment_received,
      T const *const ptr_array_previous_biased_second_moment_hat_received);
  template <typename T>
  bool Copy__Batch_Normalization_Neurons__Host_To_Device(
      T const *const ptr_array_neuron_units_scale_received,
      T const *const ptr_array_neuron_units_shift_received,
      T const *const ptr_array_neuron_units_mean_average_received,
      T const *const ptr_array_neuron_units_variance_average_received) const;

  cuModel *cumodel = NULL;
#endif

  std::wstring Get__Parameters(bool const full_description_received = false);

  size_t Get__Sizeof(size_t number_threads_received = 0_UZ,
                     size_t batch_size = 0_UZ) const;
  size_t Get__Batch_Sizeof(size_t batch_size = 0_UZ) const;
  size_t Get__Threads_Sizeof(size_t number_threads_received = 0_UZ) const;
  size_t Get__Input_Size(void) const;
  size_t get_n_out(void) const;
  size_t maximum_allowable_memory_bytes = 0_UZ;

  real quickprop_decay = -0.0001_r;
  real quickprop_mu = 1.75_r;

  real rprop_increase_factor = 1.2_r;
  real rprop_decrease_factor = 0.5_r;
  real rprop_delta_min = 1e-6_r;
  real rprop_delta_max = 50.0_r;
  real rprop_delta_zero = 0.1_r;

  real sarprop_weight_decay_shift = -6.644_r;
  real sarprop_step_error_threshold_factor = 0.1_r;
  real sarprop_step_error_shift = 1.385_r;
  real sarprop_temperature = 0.015_r;
  size_t sarprop_epoch = 0_UZ;

 private:
  bool _initialized__weight = true;

  INITIALIZER::TYPE _type_weights_initializer = INITIALIZER::NONE;

  LSUV_Parameters _LSUV_Parameters;

  // Need to be call in a sequential layer [0, ..., L - 1].
  void Order__Layer__Normalization_Iterator(Layer *const layer_it);

  bool Strategy_Comparison__Loss(
      unsigned int const strategy_index_received,
      ENV::TYPE const type_dataset_in_received,
      ENV::TYPE const type_dataset_out_received,
      Model const *const ptr_source_Neural_Network_received) const;
  bool Strategy_Comparison__Accuracy(
      unsigned int const strategy_index_received,
      ENV::TYPE const type_dataset_in_received,
      ENV::TYPE const type_dataset_out_received,
      Model const *const ptr_source_Neural_Network_received) const;
  bool set_layer_activation_function__AF(
      Layer *const layer_it,
      ACTIVATION::TYPE const type_activation_function_received);
  bool set_layer_activation_function__AF_Ind_Recurrent(
      Layer *const layer_it,
      ACTIVATION::TYPE const type_activation_function_received);
  bool set_layer_activation_function__LSTM(
      Layer *const layer_it,
      ACTIVATION::TYPE const type_activation_function_received);

  Layer *Get__Input_Layer(void) const;
  Layer *Get__Output_Layer(void) const;
  void Organize__Previous_Layers_Connected(
      size_t &ref_state_layer_index_received, Layer *const ptr_layer_received,
      Layer const *&ptr_layer_state_received) const;
  void Organize__Next_Layers_Connected(
      size_t &ref_state_layer_index_received, Layer *const ptr_layer_received,
      Layer const *&ptr_layer_state_received) const;
  void Organize__Layer__Group(size_t &ref_state_layer_index_received,
                              Layer *const ptr_layer_received,
                              Layer const *&ptr_layer_state_received) const;
};
}  // namespace DL