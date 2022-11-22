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

#include "deep-learning-lib/pch.hpp"

#include "deep-learning-lib/v1/learner/model.hpp"
#include "deep-learning-lib/data/string.hpp"
#include "deep-learning-lib/data/time.hpp"
#include "deep-learning-lib/io/logger.hpp"

namespace DL::v1 {
Model::~Model(void) { this->Deallocate(); }

void Model::clear(void) {
  if (this->type != MODEL::NONE) {
    this->type = MODEL::NONE;
    this->type_optimizer_function = OPTIMIZER::NONE;

    this->Deallocate();
  }
}

void Model::Deallocate(void) {
  // Delete basic unit variable.
  SAFE_DELETE_ARRAY(this->ptr_array_basic_units);
  SAFE_DELETE_ARRAY(this->ptr_array_basic_units_values);
  SAFE_DELETE_ARRAY(this->ptr_array_basic_units_errors);

  // Delete basic indice unit variable.
  SAFE_DELETE_ARRAY(this->ptr_array_basic_indice_units);
  SAFE_DELETE_ARRAY(
      this->ptr_array_basic_indice_units_indices);
  SAFE_DELETE_ARRAY(
      this->ptr_array_basic_indice_units_values);
  SAFE_DELETE_ARRAY(
      this->ptr_array_basic_indice_units_errors);

  // Delete block(s)/cell(s) variable.
  SAFE_DELETE_ARRAY(this->ptr_array_cells_summations_cells_inputs);
  SAFE_DELETE_ARRAY(this->ptr_array_cells_summations_input_cells_inputs);
  SAFE_DELETE_ARRAY(this->ptr_array_cells_summations_recurrent_cells_inputs);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_inputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_input_inputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_recurrent_inputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_forgets_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_input_forgets_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_recurrent_forgets_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_outputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_input_outputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_summations_recurrent_outputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_cells_inputs);
  SAFE_DELETE_ARRAY(this->ptr_array_cells_states);
  SAFE_DELETE_ARRAY(this->ptr_array_cells_states_activates);
  SAFE_DELETE_ARRAY(this->ptr_array_cells_outputs);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_inputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_forgets_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_outputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_cells_delta_inputs);
  SAFE_DELETE_ARRAY(this->ptr_array_cells_delta_input_inputs);
  SAFE_DELETE_ARRAY(this->ptr_array_cells_delta_recurrent_inputs);
  SAFE_DELETE_ARRAY(this->ptr_array_cells_delta_states);
  SAFE_DELETE_ARRAY(this->ptr_array_cells_delta_outputs);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_inputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_input_inputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_recurrent_inputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_forgets_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_input_forgets_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_recurrent_forgets_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_outputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_input_outputs_gates);
  SAFE_DELETE_ARRAY(this->ptr_array_blocks_delta_recurrent_outputs_gates);

  SAFE_DELETE_ARRAY(this->ptr_array_block_units);
  SAFE_DELETE_ARRAY(this->ptr_array_cell_units);
  // |END| Delete block(s)/cell(s) variable. |END|

  // Delete neuron unit(s) variable.
  SAFE_DELETE_ARRAY(
      this->ptr_array_neuron_units_first_forward_connection_index);  // delete[]
                                                                     // array
                                                                     // size_t.
  SAFE_DELETE_ARRAY(
      this->ptr_array_neuron_units_last_forward_connection_index);  // delete[]
                                                                    // array
                                                                    // size_t.
  SAFE_DELETE_ARRAY(
      this->ptr_array_neuron_units_number_forward_connections);  // delete[]
                                                                 // array
                                                                 // size_t.

  SAFE_DELETE_ARRAY(
      this->ptr_array_neuron_units_summations);
  SAFE_DELETE_ARRAY(this->ptr_array_neuron_units_errors);

  SAFE_DELETE_ARRAY(this->ptr_array_neuron_units);
  // |END| Delete neuron unit(s) variable. |END|

  // Delete AF unit(s) variable.
  SAFE_DELETE_ARRAY(this->ptr_array_AF_units_values);
  SAFE_DELETE_ARRAY(this->ptr_array_AF_units_errors);

  SAFE_DELETE_ARRAY(
      this->ptr_array_AF_units_type_activation_function);
  SAFE_DELETE_ARRAY(this->ptr_array_AF_units);
  // |END| Delete AF unit(s) variable. |END|

  // Delete AF Ind unit(s) variable.
  SAFE_DELETE_ARRAY(
      this->ptr_array_AF_Ind_recurrent_units_recurrent_connection_index);
  SAFE_DELETE_ARRAY(
      this->ptr_array_AF_Ind_recurrent_units_pre_AFs);
  SAFE_DELETE_ARRAY(
      this->ptr_array_AF_Ind_recurrent_units_AFs);
  SAFE_DELETE_ARRAY(
      this->ptr_array_AF_Ind_recurrent_units_errors);
  SAFE_DELETE_ARRAY(
      this->ptr_array_AF_Ind_recurrent_units_dAFs);

  SAFE_DELETE_ARRAY(
      this->ptr_array_AF_Ind_recurrent_units_type_activation_function);
  SAFE_DELETE_ARRAY(this->ptr_array_AF_Ind_recurrent_units);
  // |END| Delete AF Ind unit(s) variable. |END|

  SAFE_DELETE_ARRAY(this->ptr_array_layers);
  SAFE_DELETE_ARRAY(
      this->ptr_array_layers_number_outputs);
  SAFE_DELETE_ARRAY(
      this->ptr_array_layers_first_connection_index);
  SAFE_DELETE_ARRAY(
      this->ptr_array_layers_last_connection_index);
  SAFE_DELETE_ARRAY(this->ptr_array_bidirectional_layers);

  if (this->Use__Dropout__Bernoulli() ||
      this->Use__Dropout__Bernoulli__Inverted()) {
    this->Deallocate__Generator__Dropout_Bernoulli();

    this->Deallocate__Neuron__Mask_Dropout_Bernoulli();
  }

  if (this->Use__Dropout__Gaussian()) {
    this->Deallocate__Generator__Dropout_Gaussian();
  }

  if (this->Use__Dropout__ShakeDrop()) {
    this->Deallocate__Generator__Dropout_ShakeDrop();

    this->Deallocate__Layer__Mask_Dropout_ShakeDrop();
  }

  if (this->Use__Dropout__Uout()) {
    this->Deallocate__Generator__Dropout_Uout();
  }

  if (this->Use__Dropout__Zoneout()) {
    this->Deallocate__Generator__Dropout_Zoneout();

    this->Deallocate__Cell_Unit__Mask_Dropout_Zoneout();
  }

  if (this->Use__Normalization()) {
    this->Deallocate__Normalized_Unit();

    this->Deallocate__Normalized_Unit__Batch_Normalization();
  }

  if (this->Use__Batch_Renormalization()) {
    this->Deallocate__Normalized_Unit__Batch_Renormalization();
  }

  SAFE_DELETE_ARRAY(this->ptr_array_parameters);

  this->Deallocate__Parameter__Regularization();

#ifdef COMPILE_CUDA
  if (this->Deinitialize__CUDA() == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Deinitialize__CUDA()\" function.",);
  }
#endif

  this->Deallocate__Sparse_K_Filter();

  // Deallocate optimizer array.
  this->Deallocate__Parameter__Optimizer();
  // |END| Deallocate optimizer array. |END|

  SAFE_DELETE_ARRAY(this->ptr_array_ptr_connections);
  SAFE_DELETE_ARRAY(this->ptr_array_derivatives_parameters);

  // Loss parameters.
  SAFE_DELETE_ARRAY(this->ptr_array_number_loss);
  SAFE_DELETE_ARRAY(this->ptr_array_number_bit_fail);
  SAFE_DELETE_ARRAY(this->ptr_array_loss_values);
  // |END| Loss parameters. |END|

  // Accuracy parameters.
  SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[0]);
  SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[1]);
  SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[2]);
  SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[3]);
  SAFE_DELETE_ARRAY(this->ptr_array_accuracy_values[4]);
  // |END| Accuracy parameters. |END|
}

void Model::Deallocate__Neuron__Mask_Dropout_Bernoulli(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_units_mask_dropout_bernoulli);
}

void Model::Deallocate__Layer__Mask_Dropout_ShakeDrop(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_layers_mask_dropout_shakedrop);
}

void Model::Deallocate__Cell_Unit__Mask_Dropout_Zoneout(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_cell_units_mask_dropout_zoneout);
}

void Model::Deallocate__Sparse_K_Filter(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_k_sparse_activities);
}

void Model::Deallocate__Parameter__Optimizer(void) {
  switch (this->type_optimizer_function) {
    case OPTIMIZER::NONE:
      break;
    case OPTIMIZER::GD:
      this->Deallocate__Parameter__Gradient_Descent();
      break;
    case OPTIMIZER::IRPROP_MINUS:
      this->Deallocate__Parameter__iRPROP_minus();
      break;
    case OPTIMIZER::IRPROP_PLUS:
      this->Deallocate__Parameter__iRPROP_plus();
      break;
    case OPTIMIZER::ADABOUND:
    case OPTIMIZER::ADAM:
    case OPTIMIZER::ADAMAX:
    case OPTIMIZER::NOSADAM:
      this->Deallocate__Parameter__Adam();
      break;
    case OPTIMIZER::AMSBOUND:
    case OPTIMIZER::AMSGRAD:
      this->Deallocate__Parameter__AMSGrad();
      break;
    default:
      ERR(
          L"Optimizer function type (%d | %ls) is not managed in",
          this->type_optimizer_function,
          OPTIMIZER_NAME[this->type_optimizer_function].c_str());
      break;
  }
}

void Model::Deallocate__Parameter__Gradient_Descent(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_previous_delta_parameters);
}

void Model::Deallocate__Parameter__iRPROP_minus(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_previous_steps);
  SAFE_DELETE_ARRAY(this->ptr_array_previous_derivatives_parameters);
}

void Model::Deallocate__Parameter__iRPROP_plus(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_previous_steps);
  SAFE_DELETE_ARRAY(this->ptr_array_previous_delta_parameters);
  SAFE_DELETE_ARRAY(this->ptr_array_previous_derivatives_parameters);
}

void Model::Deallocate__Parameter__Adam(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_first_moment);
  SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_second_moment);
}

void Model::Deallocate__Parameter__AMSGrad(void) {
  this->Deallocate__Parameter__Adam();

  SAFE_DELETE_ARRAY(this->ptr_array_previous_biased_second_moment_hat);
}

void Model::Deallocate__Generator__Dropout_Bernoulli(void) {
  SAFE_DELETE_ARRAY(this->bernoulli);
}

void Model::Deallocate__Generator__Dropout_Gaussian(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Real_Gaussian);
}

void Model::Deallocate__Generator__Dropout_ShakeDrop(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Bernoulli_ShakeDrop);
  SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Real_ShakeDrop);
}

void Model::Deallocate__Generator__Dropout_Uout(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Real_Uout);
}

void Model::Deallocate__Generator__Dropout_Zoneout(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Bernoulli_Zoneout_State);
  SAFE_DELETE_ARRAY(this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden);
}

void Model::Deallocate__Parameter__Batch_Normalization(void) {
  if (this->total_normalized_units_allocated != 0_UZ) {
    size_t const tmp_new_dimension_parameters(
        this->total_parameters_allocated -
        2_UZ * this->total_normalized_units_allocated);

    if (this->Reallocate__Parameter(tmp_new_dimension_parameters) == false) {
      ERR(
          L"An error has been triggered from the "
          "\"Reallocate__Parameter(%zu)\" function.",
          tmp_new_dimension_parameters);

      return;
    }

    this->total_normalized_units_allocated = this->total_normalized_units =
        0_UZ;
  }
}

void Model::Deallocate__Normalized_Unit(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_normalized_units);
}

void Model::Deallocate__Normalized_Unit__Batch_Normalization(void) {
  SAFE_DELETE_ARRAY(
      this->ptr_array_normalized_batch_units_values_hats);
  SAFE_DELETE_ARRAY(
      this->ptr_array_normalized_batch_units_values_normalizes);
  SAFE_DELETE_ARRAY(
      this->ptr_array_normalized_batch_units_means);
  SAFE_DELETE_ARRAY(
      this->ptr_array_normalized_batch_units_variances);
  SAFE_DELETE_ARRAY(
      this->ptr_array_normalized_batch_units_derivatives_means);
  SAFE_DELETE_ARRAY(
      this->ptr_array_normalized_batch_units_derivatives_variances);
  SAFE_DELETE_ARRAY(
      this->ptr_array_normalized_batch_units_means_averages);
  SAFE_DELETE_ARRAY(
      this->ptr_array_normalized_batch_units_variances_averages);
  SAFE_DELETE_ARRAY(
      this->ptr_array_normalized_batch_units_errors);
}

void Model::Deallocate__Normalized_Unit__Batch_Renormalization(void) {
  SAFE_DELETE_ARRAY(
      this->ptr_array_normalized_batch_units_r_corrections);
  SAFE_DELETE_ARRAY(
      this->ptr_array_normalized_batch_units_d_corrections);
}

void Model::Deallocate__Parameter__Regularization(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_mask_regularized_parameters);
}
}  // namespace DL