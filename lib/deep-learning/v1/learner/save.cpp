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

// PCH:
#include "pch.hpp"

// File header:
#include "deep-learning/v1/learner/model.hpp"

// Deep learning:
#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/logger.hpp"

// Standard:
#include <fstream>

using namespace DL::File;
using namespace DL::Str;

namespace DL::v1 {
bool Model::save_spec_params(std::wstring const &path_name) {
  if (create_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`create_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  std::wofstream file(CP_STR(path_name), std::ios::out | std::ios::binary);

  if (file.is_open() == false) {
    ERR(L"The file %ls can not be opened.", path_name.c_str());
    return false;
  }

  std::wstring inp;

  // clang-format off
  inp += L"|===| GRADIENT DESCENT PARAMETERS |===|" CRLF;
  inp += L"learning_rate " + to_wstring(this->learning_rate) + CRLF;
  inp += L"learning_rate_final " + to_wstring(this->learning_rate_final) + CRLF;
  inp += L"learning_momentum " + to_wstring(this->learning_momentum) + CRLF;
  inp += L"learning_gamma " + to_wstring(this->learning_gamma) + CRLF;
  inp += L"use_nesterov " + std::to_wstring(this->use_nesterov) + CRLF;
  inp += L"|END| GRADIENT DESCENT PARAMETERS |END|" CRLF;
  inp += CRLF;

  inp += L"|===| QUICKPROP PARAMETERS |===|" CRLF;
  inp += L"quickprop_decay " + to_wstring(this->quickprop_decay) + CRLF;
  inp += L"quickprop_mu " + to_wstring(this->quickprop_mu) + CRLF;
  inp += L"|END| QUICKPROP PARAMETERS |END|" CRLF;
  inp += CRLF;

  inp += L"|===| RESILLENT PROPAGATION PARAMETERS |===|" CRLF;
  inp += L"rprop_increase_factor " + to_wstring(this->rprop_increase_factor) + CRLF;
  inp += L"rprop_decrease_factor " + to_wstring(this->rprop_decrease_factor) + CRLF;
  inp += L"rprop_delta_min " + to_wstring(this->rprop_delta_min) + CRLF;
  inp += L"rprop_delta_max " + to_wstring(this->rprop_delta_max) + CRLF;
  inp += L"rprop_delta_zero " + to_wstring(this->rprop_delta_zero) + CRLF;
  inp += L"|END| RESILLENT PROPAGATION PARAMETERS |END|" CRLF;
  inp += CRLF;

  inp += L"|===| SARPROP PARAMETERS |===|" CRLF;
  inp += L"sarprop_weight_decay_shift " + to_wstring(this->sarprop_weight_decay_shift) + CRLF;
  inp += L"sarprop_step_error_threshold_factor " + to_wstring(this->sarprop_step_error_threshold_factor) + CRLF;
  inp += L"sarprop_step_error_shift " + to_wstring(this->sarprop_step_error_shift) + CRLF;
  inp += L"sarprop_temperature " + to_wstring(this->sarprop_temperature) + CRLF;
  inp += L"sarprop_epoch " + std::to_wstring(this->sarprop_epoch) + CRLF;
  inp += L"|END| SARPROP PARAMETERS |END|" CRLF;
  inp += CRLF;

  inp += L"|===| ADAM PARAMETERS |===|" CRLF;
  inp += L"adam_learning_rate " + to_wstring(this->adam_learning_rate) + CRLF;
  inp += L"adam_beta1 " + to_wstring(this->adam_beta1) + CRLF;
  inp += L"adam_beta2 " + to_wstring(this->adam_beta2) + CRLF;
  inp += L"adam_epsilon " + to_wstring(this->adam_epsilon) + CRLF;
  inp += L"adam_bias_correction " + std::to_wstring(this->use_adam_bias_correction) + CRLF;
  inp += L"adam_gamma " + to_wstring(this->adam_gamma) + CRLF;
  inp += L"|END| ADAM PARAMETERS |END|" CRLF;
  inp += CRLF;

  inp += L"|===| WARM RESTARTS PARAMETERS |===|" CRLF;
  inp += L"use_warm_restarts " + std::to_wstring(this->use_warm_restarts) + CRLF;
  inp += L"warm_restarts_decay_learning_rate " + to_wstring(this->warm_restarts_decay_learning_rate) + CRLF;
  inp += L"warm_restarts_maximum_learning_rate " + to_wstring(this->warm_restarts_initial_maximum_learning_rate) + CRLF;
  inp += L"warm_restarts_minimum_learning_rate " + to_wstring(this->warm_restarts_minimum_learning_rate) + CRLF;
  inp += L"warm_restarts_initial_T_i " + std::to_wstring(static_cast<size_t>(this->warm_restarts_initial_T_i)) + CRLF;
  inp += L"warm_restarts_multiplier " + std::to_wstring(static_cast<size_t>(this->warm_restarts_multiplier)) + CRLF;
  inp += L"|END| WARM RESTARTS PARAMETERS |END|" CRLF;
  inp += CRLF;

  inp += L"|===| TRAINING PARAMETERS |===|" CRLF;
  inp += L"type_optimizer_function " + std::to_wstring(this->type_optimizer_function) + CRLF;
  inp += L"type_loss_function " + std::to_wstring(this->type_loss_function) + CRLF;
  inp += L"type_accuracy_function " + std::to_wstring(this->type_accuracy_function) + CRLF;
  inp += L"bit_fail_limit " + to_wstring(this->bit_fail_limit) + CRLF;
  inp += L"pre_training_level " + std::to_wstring(this->pre_training_level) + CRLF;
  inp += L"use_clip_gradient " + std::to_wstring(this->use_clip_gradient) + CRLF;
  inp += L"clip_gradient " + to_wstring(this->clip_gradient) + CRLF;
  inp += L"|END| TRAINING PARAMETERS |END|" CRLF;
  inp += CRLF;

  inp += L"|===| REGULARIZATION PARAMETERS |===|" CRLF;
  inp += L"regularization__max_norm_constraints " + to_wstring(this->regularization__max_norm_constraints) + CRLF;
  inp += L"regularization__l1 " + to_wstring(this->regularization__l1) + CRLF;
  inp += L"regularization__l2 " + to_wstring(this->regularization__l2) + CRLF;
  inp += L"regularization__srip " + to_wstring(this->regularization__srip) + CRLF;
  inp += L"weight_decay " + to_wstring(this->weight_decay) + CRLF;
  inp += L"use_normalized_weight_decay " + std::to_wstring(this->use_normalized_weight_decay) + CRLF;
  inp += L"|END| REGULARIZATION PARAMETERS |END|" CRLF;
  inp += CRLF;

  inp += L"|===| NORMALIZATION PARAMETERS |===|" CRLF;
  inp += L"normalization_momentum_average " + to_wstring(this->normalization_momentum_average) + CRLF;
  inp += L"normalization_epsilon " + to_wstring(this->normalization_epsilon) + CRLF;
  inp += L"batch_renormalization_r_correction_maximum " + to_wstring(this->batch_renormalization_r_correction_maximum) + CRLF;
  inp += L"batch_renormalization_d_correction_maximum " + to_wstring(this->batch_renormalization_d_correction_maximum) + CRLF;
  inp += L"|END| NORMALIZATION PARAMETERS |END|" CRLF;
  inp += CRLF;

  inp += L"|===| LOSS PARAMETERS |===|" CRLF;
  inp += L"loss_train " + to_wstring(this->loss_train) + CRLF;
  inp += L"loss_valid " + to_wstring(this->loss_valid) + CRLF;
  inp += L"loss_testg " + to_wstring(this->loss_testg) + CRLF;
  inp += L"|END| LOSS PARAMETERS |END|" CRLF;
  inp += CRLF;

  inp += L"|===| ACCURANCY PARAMETERS |===|" CRLF;
  inp += L"acc_var " + to_wstring(this->acc_var) + CRLF;
  inp += L"acc_train " + to_wstring(this->acc_train) + CRLF;
  inp += L"acc_valid " + to_wstring(this->acc_valid) + CRLF;
  inp += L"acc_testg " + to_wstring(this->acc_testg) + CRLF;
  inp += L"|END| ACCURANCY PARAMETERS |END|" CRLF;
  inp += CRLF;

  inp += L"|===| COMPUTATION PARAMETERS |===|" CRLF;
  inp += L"use_cu " + std::to_wstring(this->use_cu) + CRLF;
  inp += L"use_mp " + std::to_wstring(this->use_mp) + CRLF;
  inp += L"pct_threads " + to_wstring(this->pct_threads, 9u) + CRLF;
  inp += L"maximum_batch_size " + std::to_wstring(this->maximum_batch_size) + CRLF;
  inp += L"|END| COMPUTATION PARAMETERS |END|" CRLF;
  // clang-format on

  file.write(inp.c_str(), static_cast<std::streamsize>(inp.size()));

  if (file.fail()) {
    ERR(L"An error has been triggered from the "
        L"`write(string, %zu)` function. "
        L"Logical error on i/o operation \"%ls\".",
        inp.size(), path_name.c_str());
    return false;
  }

  file.flush();

  if (file.fail()) {
    ERR(L"An error has been triggered from the "
        L"`flush()` function. "
        L"Logical error on i/o operation \"%ls\".",
        path_name.c_str());
    return false;
  }

  file.close();

  if (delete_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`delete_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  return true;
}

bool Model::save_params(std::wstring const &path_name) {
  if (create_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`create_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  std::wofstream file(CP_STR(path_name), std::ios::out | std::ios::binary);

  if (file.is_open() == false) {
    ERR(L"The file %ls can not be opened.", path_name.c_str());
    return false;
  }

  std::wstring inp;

  Layer const *const last_layer(this->ptr_last_layer -
                                1),  // Subtract output layer.
      *prev_layer, *layer_it(this->ptr_array_layers);

#ifdef COMPILE_CUDA
  if (this->is_cu_initialized && this->is_update_from_device == false) {
    if (this->Copy__Parameters__Device_To_Host() == false) {
      ERR(L"An error has been triggered from the "
          L"`Copy__Parameters__Device_To_Host(%ls)` function.");
    } else if (this->Use__Normalization() &&
               this->Copy__Batch_Normalization_Neurons__Device_To_Host() ==
                   false) {
      ERR(L"An error has been triggered from the "
          L"`Copy__Batch_Normalization_Neurons__Device_To_Host(%ls)` "
          L"function.");
    }
  }
#endif
  // clang-format off
  inp = L"|===| DIMENSION |===|" CRLF;
  inp += L"type " + std::to_wstring(this->type) + CRLF;
  inp += L"number_layers " + std::to_wstring(this->total_layers) + CRLF;
  inp += L"seq_w " + std::to_wstring(this->seq_w) + CRLF;
  inp += L"n_time_delay " + std::to_wstring(this->n_time_delay) + CRLF;
  inp += L"use_first_layer_as_input " + std::to_wstring(this->use_first_layer_as_input) + CRLF;
  inp += L"use_last_layer_as_output " + std::to_wstring(this->use_last_layer_as_output) + CRLF;
  inp += L"total_basic_units " + std::to_wstring(this->total_basic_units) + CRLF;
  inp += L"total_basic_indice_units " + std::to_wstring(this->total_basic_indice_units) + CRLF;
  inp += L"total_neuron_units " + std::to_wstring(this->total_neuron_units) + CRLF;
  inp += L"total_AF_units " + std::to_wstring(this->total_AF_units) + CRLF;
  inp += L"total_AF_Ind_recurrent_units " + std::to_wstring(this->total_AF_Ind_recurrent_units) + CRLF;
  inp += L"total_block_units " + std::to_wstring(this->total_block_units) + CRLF;
  inp += L"total_cell_units " + std::to_wstring(this->total_cell_units) + CRLF;
  inp += L"total_normalized_units " + std::to_wstring(this->total_normalized_units) + CRLF;
  inp += L"total_parameters " + std::to_wstring(this->total_parameters) + CRLF;
  inp += L"total_weights " + std::to_wstring(this->total_weights) + CRLF;
  inp += L"total_bias " + std::to_wstring(this->total_bias) + CRLF;
  // clang-format on

  auto get_dropout_params_fn([](Layer const *const layer,
                                bool const is_hidden_layer_received =
                                    true) -> std::wstring {
    // clang-format off
    std::wstring out(L"");

    out += L"    type_dropout " + std::to_wstring(layer->type_dropout) + CRLF;

    if (is_hidden_layer_received)
      out += L"      use_coded_dropout " + std::to_wstring(layer->use_coded_dropout) + CRLF;

    switch (layer->type_dropout) {
      case LAYER_DROPOUT::ALPHA:
        out += L"      dropout_values[0] " + to_wstring(1_r - layer->dropout_values[0]) + CRLF;
        out += L"      dropout_values[1] " + to_wstring(layer->dropout_values[1]) + CRLF;
        out += L"      dropout_values[2] " + to_wstring(layer->dropout_values[2]) + CRLF;
        break;
      case LAYER_DROPOUT::GAUSSIAN:
        out += L"      dropout_values[0] " + to_wstring(layer->dropout_values[0] / (layer->dropout_values[0] + 1_r)) + CRLF;
        out += L"      dropout_values[1] " + to_wstring(layer->dropout_values[1]) + CRLF;
        out += L"      dropout_values[2] " + to_wstring(layer->dropout_values[2]) + CRLF;
        break;
      default:
        out += L"      dropout_values[0] " + to_wstring(layer->dropout_values[0]) + CRLF;
        out += L"      dropout_values[1] " + to_wstring(layer->dropout_values[1]) + CRLF;
        out += L"      dropout_values[2] " + to_wstring(layer->dropout_values[2]) + CRLF;
        break;
    }

    // clang-format on
    return (out);
  });

  // clang-format off
  // Input layer.
  inp += L"  Input layer:" CRLF;
  inp += L"    type_layer " + std::to_wstring(layer_it->type_layer) + CRLF;
  inp += L"    type_activation " + std::to_wstring(layer_it->type_activation) + CRLF;
  inp += get_dropout_params_fn(layer_it, false);
  inp += L"    n_inp " + std::to_wstring(this->n_inp) + CRLF;
  // |END| Input layer. |END|

  // Hidden layer.
  for (++layer_it; layer_it != last_layer; ++layer_it) {
    inp += L"  Hidden layer " + std::to_wstring(static_cast<size_t>(layer_it - this->ptr_array_layers)) + L":" CRLF;
    inp += L"    type_layer " + std::to_wstring(layer_it->type_layer) + CRLF;
    inp += L"    use_bidirectional " + std::to_wstring(layer_it->use_bidirectional) + CRLF;

    switch (layer_it->type_layer) {
      case LAYER::AVERAGE_POOLING:
        inp += L"    kernel_size " + std::to_wstring(layer_it->pooling_values[0]) + CRLF;
        inp += L"    stride " + std::to_wstring(layer_it->pooling_values[1]) + CRLF;
        inp += L"    padding " + std::to_wstring(layer_it->pooling_values[2]) + CRLF;
        inp += L"    dilation " + std::to_wstring(layer_it->pooling_values[3]) + CRLF;
        inp += L"    number_basic_units " + std::to_wstring(*layer_it->ptr_number_outputs) + CRLF;
        break;
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_RECURRENT:
        prev_layer = layer_it->previous_connected_layers[0];

        inp += L"    type_activation " + std::to_wstring(layer_it->type_activation) + CRLF;
        inp += get_dropout_params_fn(layer_it);

        // Normalization.
        if (this->Information__Layer__Normalization(inp, layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__Normalization()` function.");
          return false;
        }
        // |END| Normalization. |END|

        inp += L"    use_tied_parameter " + std::to_wstring(layer_it->use_tied_parameter) + CRLF;
        inp += L"    k_sparsity " + std::to_wstring(layer_it->k_sparsity) + CRLF;
        inp += L"    alpha_sparsity " + to_wstring(layer_it->alpha_sparsity) + CRLF;
        inp += L"    constraint_recurrent_weight_lower_bound " + to_wstring(layer_it->constraint_recurrent_weight_lower_bound) + CRLF;
        inp += L"    constraint_recurrent_weight_upper_bound " + to_wstring(layer_it->constraint_recurrent_weight_upper_bound) + CRLF;

        // Neuron unit(s).
        if (this->Information__Layer__FC(inp, layer_it, prev_layer) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__FC()` function.");
          return false;
        }
        // |END| Neuron unit(s). |END|

        // AF unit(s).
        if (this->Information__Layer__AF(inp, layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__AF()` function.");
          return false;
        }
        // |END| AF unit(s). |END|

        // Bias parameter(s).
        if (this->Information__Layer__Bias(inp, layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__Bias()` function.");
          return false;
        }
        // |END| Bias parameter(s). |END|
        break;
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        prev_layer = layer_it->previous_connected_layers[0];

        inp += L"    type_activation " + std::to_wstring(layer_it->type_activation) + CRLF;
        inp += get_dropout_params_fn(layer_it);

        // Normalization.
        if (this->Information__Layer__Normalization(inp, layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__Normalization()` function.");
          return false;
        }
        // |END| Normalization. |END|

        inp += L"    use_tied_parameter " + std::to_wstring(layer_it->use_tied_parameter) + CRLF;
        inp += L"    k_sparsity " + std::to_wstring(layer_it->k_sparsity) + CRLF;
        inp += L"    alpha_sparsity " + to_wstring(layer_it->alpha_sparsity) + CRLF;
        inp += L"    constraint_recurrent_weight_lower_bound " + to_wstring(layer_it->constraint_recurrent_weight_lower_bound) + CRLF;
        inp += L"    constraint_recurrent_weight_upper_bound " + to_wstring(layer_it->constraint_recurrent_weight_upper_bound) + CRLF;

        // Neuron unit(s).
        if (this->Information__Layer__FC(inp, layer_it, prev_layer) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__FC()` function.");
          return false;
        }
        // |END| Neuron unit(s). |END|

        // AF Ind recurrent unit(s).
        if (this->Information__Layer__AF_Ind_Recurrent(inp, layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__AF_Ind_Recurrent()` function.");
          return false;
        }
        // |END| AF Ind recurrent unit(s). |END|

        // Bias parameter(s).
        if (this->Information__Layer__Bias(inp, layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__Bias()` function.");
          return false;
        }
        // |END| Bias parameter(s). |END|
        break;
      case LAYER::LSTM:
        prev_layer = layer_it->previous_connected_layers[0];

        inp += L"    type_activation " + std::to_wstring(layer_it->type_activation) + CRLF;
        inp += get_dropout_params_fn(layer_it);

        // Normalization.
        if (this->Information__Layer__Normalization(inp, layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__Normalization()` function.");
          return false;
        }
        // |END| Normalization. |END|

        inp += L"    use_tied_parameter " + std::to_wstring(layer_it->use_tied_parameter) + CRLF;
        inp += L"    k_sparsity " + std::to_wstring(layer_it->k_sparsity) + CRLF;
        inp += L"    alpha_sparsity " + to_wstring(layer_it->alpha_sparsity) + CRLF;
        inp += L"    constraint_recurrent_weight_lower_bound " + to_wstring(layer_it->constraint_recurrent_weight_lower_bound) + CRLF;
        inp += L"    constraint_recurrent_weight_upper_bound " + to_wstring(layer_it->constraint_recurrent_weight_upper_bound) + CRLF;

        // Blocks unit(s).
        if (this->Information__Layer__LSTM(inp, layer_it, prev_layer) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__LSTM()` function.");
          return false;
        }
        // |END| Blocks unit(s). |END|

        // Bias parameter(s).
        if (this->Information__Layer__Bias(inp, layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__Bias()` function.");
          return false;
        }
        // |END| Bias parameter(s). |END|
        break;
      case LAYER::MAX_POOLING:
        inp += L"    kernel_size " + std::to_wstring(layer_it->pooling_values[0]) + CRLF;
        inp += L"    stride " + std::to_wstring(layer_it->pooling_values[1]) + CRLF;
        inp += L"    padding " + std::to_wstring(layer_it->pooling_values[2]) + CRLF;
        inp += L"    dilation " + std::to_wstring(layer_it->pooling_values[3]) + CRLF;
        inp += L"    number_basic_indice_units " + std::to_wstring(*layer_it->ptr_number_outputs) + CRLF;
        break;
      case LAYER::RESIDUAL:
        inp += L"    block_depth " + std::to_wstring(layer_it->block_depth) + CRLF;
        inp += L"    padding " + std::to_wstring(layer_it->pooling_values[2]) + CRLF;
        inp += get_dropout_params_fn(layer_it);

        // Normalization.
        if (this->Information__Layer__Normalization(inp, layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`Information__Layer__Normalization()` function.");
          return false;
        }
        // |END| Normalization. |END|

        inp += L"    number_basic_units " + std::to_wstring(*layer_it->ptr_number_outputs) + CRLF;
        break;
      default:
        ERR(L"Layer type (%d | %ls) is not managed in the switch.", layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
        return false;
    }
  }
  // |END| Hidden layer. |END|

  // Output layer.
  prev_layer = layer_it->previous_connected_layers[0];

  inp += L"  Output layer:" CRLF;
  inp += L"    type_layer " + std::to_wstring(layer_it->type_layer) + CRLF;
  inp += L"    type_activation " + std::to_wstring(layer_it->type_activation) + CRLF;

  //  Neuron_unit.
  if (this->Information__Output_Layer(inp, layer_it, prev_layer) == false) {
    ERR(L"An error has been triggered from the "
        L"`Information__Output_Layer()` function.");
    return false;
  }
  //  |END| Neuron_unit. |END|

  //  AF unit(s).
  if (this->Information__Layer__AF(inp, layer_it) == false) {
    ERR(L"An error has been triggered from the "
        L"`Information__Layer__AF()` function.");
    return false;
  }
  //  |END| AF unit(s). |END|

  //  Bias parameter(s).
  if (this->Information__Layer__Bias(inp, layer_it) == false) {
    ERR(L"An error has been triggered from the "
        L"`Information__Layer__Bias()` function.");
    return false;
  }
  //  |END| Bias parameter(s). |END|
  // |END| Output layer. |END|

  inp += L"|END| DIMENSION |END|" CRLF;
  // clang-format on

  file.write(inp.c_str(), static_cast<std::streamsize>(inp.size()));

  if (file.fail()) {
    ERR(L"An error has been triggered from the "
        L"`write(string, %zu)` function. "
        L"Logical error on i/o operation \"%ls\".",
        inp.size(), path_name.c_str());
    return false;
  }

  file.flush();

  if (file.fail()) {
    ERR(L"An error has been triggered from the "
        L"`flush()` function. "
        L"Logical error on i/o operation \"%ls\".",
        path_name.c_str());
    return false;
  }

  file.close();

  if (delete_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`delete_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  return true;
}

bool Model::Information__Layer__Normalization(std::wstring &out,
                                              Layer const *const layer) {
  union Normalized_unit const *const first_unit(
      layer->ptr_array_normalized_units),
      *const last_unit(layer->ptr_last_normalized_unit), *unit_it(first_unit);

  size_t const n_units(static_cast<size_t>(last_unit - unit_it));

  // clang-format off
  out += L"    type_normalization " + std::to_wstring(layer->type_normalization) + CRLF;

  switch (layer->type_layer) {
    case LAYER::LSTM:
    case LAYER::RESIDUAL:
      break;
    case LAYER::FULLY_CONNECTED:
    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
    case LAYER::FULLY_CONNECTED_RECURRENT:
      out += L"      use_layer_normalization_before_activation " + std::to_wstring(layer->use_layer_normalization_before_activation) + CRLF;
      break;
    default:
      ERR(L"Layer type (%d | %ls) is not managed in the switch.", layer->type_layer, LAYER_NAME[layer->type_layer].c_str());
      return false;
  }

  out += L"      number_normalized_units " + std::to_wstring(n_units) + CRLF;

  for (; unit_it != last_unit; ++unit_it) {
    out += L"        NormU[" + std::to_wstring(static_cast<size_t>(unit_it - first_unit)) + L"]" CRLF;

    if (this->Information__Normalized_Unit(n_units, layer->type_normalization, unit_it, out) == false) {
      ERR(L"An error has been triggered from the "
          L"`Information__Normalized_Unit(%zu, ref, ref)` function.", n_units);
    }
  }
  // clang-format on

  return true;
}

bool Model::Information__Normalized_Unit(
    size_t const n_units, LAYER_NORM::TYPE const type,
    union Normalized_unit const *const unit, std::wstring &out) {
  size_t t, i;

  // clang-format off
  switch (type) {
    case LAYER_NORM::BATCH_NORMALIZATION:
    case LAYER_NORM::BATCH_RENORMALIZATION:
    case LAYER_NORM::GHOST_BATCH_NORMALIZATION:
      out += L"          scale " + to_wstring(cast(*unit->normalized_batch_units.ptr_scale)) + CRLF;
      out += L"          shift " + to_wstring(cast(*unit->normalized_batch_units.ptr_shift)) + CRLF;

      for (t = 0_UZ; t != this->seq_w; ++t) {
        i = n_units * t;

        out += L"          mean_average[" + std::to_wstring(t) + L"] " + to_wstring(cast(unit->normalized_batch_units.ptr_mean_average[i])) + CRLF;
        out += L"          variance_average[" + std::to_wstring(t) + L"] " + to_wstring(cast(unit->normalized_batch_units.ptr_variance_average[i])) + CRLF;
      }
      break;
    default:
      ERR(L"Layer normalization type (%d | %ls) is not managed in the switch.", type, LAYER_NORM_NAME[type].c_str());
      return false;
  }
  // clang-format on

  return true;
}

bool Model::Information__Layer__FC(std::wstring &out, Layer const *const layer,
                                   Layer const *const prev_layer) {
  Neuron_unit const *const first_unit(layer->ptr_array_neuron_units),
      *const last_unit(layer->ptr_last_neuron_unit), *unit_it;

  // clang-format off
  out += L"    number_neuron_units " + std::to_wstring(static_cast<size_t>(last_unit - first_unit)) + CRLF;

  for (unit_it = first_unit; unit_it != last_unit; ++unit_it) {
    out += L"      Neuron_unit[" + std::to_wstring(static_cast<size_t>(unit_it - first_unit)) + L"]" CRLF;
    out += L"        number_connections " + std::to_wstring(*unit_it->ptr_number_connections) + CRLF;

    if (*unit_it->ptr_number_connections != 0_UZ) {
      switch (prev_layer->type_layer) {
        case LAYER::AVERAGE_POOLING:
        case LAYER::RESIDUAL:
          this->Layer__Forward__Neuron_Information__Connection<Basic_unit, LAYER::AVERAGE_POOLING>(out, unit_it, this->ptr_array_basic_units);
          break;
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          this->Layer__Forward__Neuron_Information__Connection<Neuron_unit, LAYER::FULLY_CONNECTED>(out, unit_it, this->ptr_array_neuron_units);
          break;
        case LAYER::LSTM:
          this->Layer__Forward__Neuron_Information__Connection<CellUnit, LAYER::LSTM>(out, unit_it, this->ptr_array_cell_units);
          break;
        case LAYER::MAX_POOLING:
          this->Layer__Forward__Neuron_Information__Connection<Basic_indice_unit, LAYER::MAX_POOLING>(out, unit_it, this->ptr_array_basic_indice_units);
          break;
        default:
          ERR(L"Layer type (%d | %ls) is not managed in the switch.", prev_layer->type_layer, LAYER_NAME[prev_layer->type_layer].c_str());
          return false;
      }
    }
  }
  // clang-format on

  return true;
}

bool Model::Information__Output_Layer(std::wstring &out,
                                      Layer const *const layer,
                                      Layer const *const prev_layer) {
  Neuron_unit const *const first_unit(layer->ptr_array_neuron_units),
      *const last_unit(layer->ptr_last_neuron_unit), *unit_it;

  // clang-format off
  out += L"    n_out " + std::to_wstring(static_cast<size_t>(last_unit - first_unit)) + CRLF;

  for (unit_it = first_unit; unit_it != last_unit; ++unit_it) {
    out += L"      Neuron_unit[" + std::to_wstring(static_cast<size_t>(unit_it - first_unit)) + L"]" CRLF;
    out += L"        number_connections " + std::to_wstring(*unit_it->ptr_number_connections) + CRLF;

    if (*unit_it->ptr_number_connections != 0_UZ) {
      switch (prev_layer->type_layer) {
        case LAYER::AVERAGE_POOLING:
        case LAYER::RESIDUAL:
          this->Layer__Forward__Neuron_Information__Connection<Basic_unit, LAYER::AVERAGE_POOLING>(out, unit_it, this->ptr_array_basic_units);
          break;
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          this->Layer__Forward__Neuron_Information__Connection<Neuron_unit, LAYER::FULLY_CONNECTED>(out, unit_it, this->ptr_array_neuron_units);
          break;
        case LAYER::LSTM:
          this->Layer__Forward__Neuron_Information__Connection<CellUnit, LAYER::LSTM>(out, unit_it, this->ptr_array_cell_units);
          break;
        case LAYER::MAX_POOLING:
          this->Layer__Forward__Neuron_Information__Connection<Basic_indice_unit, LAYER::MAX_POOLING>(out, unit_it, this->ptr_array_basic_indice_units);
          break;
        default:
          ERR(L"Layer type (%d | %ls) is not managed in the switch.", prev_layer->type_layer, LAYER_NAME[prev_layer->type_layer].c_str());
          return false;
      }
    }
  }
  // clang-format on

  return true;
}

bool Model::Information__Layer__AF(std::wstring &out,
                                   Layer const *const layer) {
  AF_unit const *const first_unit(layer->ptr_array_AF_units),
      *const last_unit(layer->ptr_last_AF_unit), *unit_it;

  // clang-format off
  out += L"    number_AF_units " + std::to_wstring(static_cast<size_t>(last_unit - first_unit)) + CRLF;

  for (unit_it = first_unit; unit_it != last_unit; ++unit_it) {
    out += L"      AF[" + std::to_wstring(static_cast<size_t>(unit_it - first_unit)) + L"]" CRLF;
    out += L"        activation_function " + std::to_wstring(static_cast<size_t>(*unit_it->ptr_type_activation_function)) + CRLF;
  }
  // clang-format on

  return true;
}

bool Model::Information__Layer__AF_Ind_Recurrent(std::wstring &out,
                                                 Layer const *const layer) {
  AF_Ind_recurrent_unit const *const *const connections(
      reinterpret_cast<AF_Ind_recurrent_unit **>(
          this->ptr_array_ptr_connections)),
      *const first_unit(layer->ptr_array_AF_Ind_recurrent_units),
          *const last_unit(layer->ptr_last_AF_Ind_recurrent_unit), *unit_it;

  // clang-format off
  out += L"    number_AF_Ind_recurrent_units " + std::to_wstring(static_cast<size_t>(last_unit - first_unit)) + CRLF;

  for (unit_it = first_unit; unit_it != last_unit; ++unit_it) {
    out += L"      AF_Ind_R[" + std::to_wstring(static_cast<size_t>(unit_it - first_unit)) + L"]" CRLF;
    out += L"        activation_function " + std::to_wstring(static_cast<size_t>(*unit_it->ptr_type_activation_function)) + CRLF;
    out += L"        connected_to_AF_Ind_R " + std::to_wstring(connections[*unit_it->ptr_recurrent_connection_index] - this->ptr_array_AF_Ind_recurrent_units) + CRLF;
    out += L"        weight[" + std::to_wstring(*unit_it->ptr_recurrent_connection_index) + L"] " + to_wstring(cast(this->ptr_array_parameters[*unit_it->ptr_recurrent_connection_index])) + CRLF;
  }
  // clang-format on

  return true;
}

bool Model::Information__Layer__Bias(std::wstring &out,
                                     Layer const *const layer) {
  size_t const n_connections(layer->last_bias_connection_index -
                             layer->first_bias_connection_index);

  var const *const params(this->ptr_array_parameters +
                          layer->first_bias_connection_index);

  // clang-format off
  out += L"    number_bias_parameters " + std::to_wstring(n_connections) + CRLF;

  for (size_t i(0_UZ); i != n_connections; ++i)
    out += L"      weight[" + std::to_wstring(layer->first_bias_connection_index + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
  // clang-format on

  return true;
}

template <class U, LAYER::TYPE const E>
void Model::Layer__Forward__Neuron_Information__Connection(
    std::wstring &out, Neuron_unit const *const unit,
    U const *const first_inp_unit) {
  size_t const n_connections(*unit->ptr_number_connections);

  var const *const params(this->ptr_array_parameters +
                          *unit->ptr_first_connection_index);

  U **connections(reinterpret_cast<U **>(this->ptr_array_ptr_connections +
                                         *unit->ptr_first_connection_index));

  // clang-format off
  for (size_t i(0_UZ); i != n_connections; ++i) {
    out += L"          " + LAYER_CONN_NAME[E] + L" " + std::to_wstring(connections[i] - first_inp_unit) + CRLF;
    out += L"          weight[" + std::to_wstring(*unit->ptr_first_connection_index + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
  }
  // clang-format on
}

bool Model::Information__Layer__LSTM(std::wstring &out,
                                     Layer const *const layer,
                                     Layer const *const prev_layer) {
  BlockUnit const *const first_block(layer->ptr_array_block_units),
      *const last_block(layer->ptr_last_block_unit), *block_it(first_block);

  size_t const n_blocks(static_cast<size_t>(last_block - block_it)),
      n_cells(static_cast<size_t>(layer->ptr_last_cell_unit -
                                  layer->ptr_array_cell_units));

  // clang-format off
  out += L"    number_block_units " + std::to_wstring(n_blocks) + CRLF;
  out += L"    number_cell_units " + std::to_wstring(n_cells) + CRLF;

  for (; block_it != last_block; ++block_it) {
    out += L"      Block[" + std::to_wstring(static_cast<size_t>(block_it - first_block)) + L"]" CRLF;
    out += L"        activation_function " + std::to_wstring(static_cast<size_t>(block_it->activation_function_io)) + CRLF;
    out += L"        number_connections " + std::to_wstring(block_it->last_index_connection - block_it->first_index_connection) + CRLF;

    switch (prev_layer->type_layer) {
      case LAYER::AVERAGE_POOLING:
      case LAYER::RESIDUAL:
        this->Layer__LSTM_Information__Connection<Basic_unit, LAYER::AVERAGE_POOLING>(out, block_it, this->ptr_array_basic_units);
        break;
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_RECURRENT:
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        this->Layer__LSTM_Information__Connection<Neuron_unit, LAYER::FULLY_CONNECTED>(out, block_it, this->ptr_array_neuron_units);
        break;
      case LAYER::LSTM:
        this->Layer__LSTM_Information__Connection<CellUnit, LAYER::LSTM>(out, block_it, this->ptr_array_cell_units);
        break;
      case LAYER::MAX_POOLING:
        this->Layer__LSTM_Information__Connection<Basic_indice_unit, LAYER::MAX_POOLING>(out, block_it, this->ptr_array_basic_indice_units);
        break;
      default:
        ERR(L"Layer type (%d | %ls) is not managed in the switch.", prev_layer->type_layer, LAYER_NAME[prev_layer->type_layer].c_str());
        return false;
    }
  }
  // clang-format on

  return true;
}

template <class U, LAYER::TYPE const E>
void Model::Layer__LSTM_Information__Connection(std::wstring &out,
                                                BlockUnit const *const block_it,
                                                U const *const first_inp_unit) {
  size_t const n_peephole_conns(block_it->last_index_peephole_input_gate -
                                block_it->first_index_peephole_input_gate),
      n_input_conns(block_it->last_index_feedforward_connection_input_gate -
                    block_it->first_index_feedforward_connection_input_gate),
      n_state_conns(block_it->last_index_recurrent_connection_input_gate -
                    block_it->first_index_recurrent_connection_input_gate);
  size_t i;

  var const *params;

  U **connections(reinterpret_cast<U **>(
      this->ptr_array_ptr_connections +
      block_it->first_index_feedforward_connection_input_gate));

  CellUnit const *const *const state_connections(reinterpret_cast<CellUnit **>(
      this->ptr_array_ptr_connections +
      block_it->first_index_recurrent_connection_input_gate)),
      *const *const peephole_connections(reinterpret_cast<CellUnit **>(
          this->ptr_array_ptr_connections +
          block_it->first_index_peephole_input_gate)),
          *const last_unit(block_it->ptr_last_cell_unit),
      *unit_it(block_it->ptr_array_cell_units);

  // clang-format off
  // [0] Cell input.
  for (; unit_it != last_unit; ++unit_it) {
    //    [1] Input, cell.
    params = this->ptr_array_parameters + unit_it->first_index_feedforward_connection_cell_input;

    for (i = 0_UZ; i != n_input_conns; ++i) {
      out += L"          " + LAYER_CONN_NAME[E] + L" " + std::to_wstring(connections[i] - first_inp_unit) + CRLF;
      out += L"          weight[" + std::to_wstring(unit_it->first_index_feedforward_connection_cell_input + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
    }
    //    [1] |END| Input, cell. |END|

    //    [1] Recurrent, cell.
    params = this->ptr_array_parameters + unit_it->first_index_recurrent_connection_cell_input;

    for (i = 0_UZ; i != n_state_conns; ++i) {
      out += L"          connected_to_cell " + std::to_wstring(state_connections[i] - this->ptr_array_cell_units) + CRLF;
      out += L"          weight[" + std::to_wstring(unit_it->first_index_recurrent_connection_cell_input + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
    }
    //    [1] |END| Recurrent, cell. |END|
  }
  // [0] |END| Cell input. |END|

  // [0] Input, gates.
  //    [1] Input gate.
  params = this->ptr_array_parameters + block_it->first_index_feedforward_connection_input_gate;

  for (i = 0_UZ; i != n_input_conns; ++i) {
    out += L"          " + LAYER_CONN_NAME[E] + L" " + std::to_wstring(connections[i] - first_inp_unit) + CRLF;
    out += L"          weight[" + std::to_wstring(block_it->first_index_feedforward_connection_input_gate + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
  }
  //    [1] |END| Input gate. |END|

  //    [1] Forget gate.
  params = this->ptr_array_parameters + block_it->first_index_feedforward_connection_forget_gate;

  for (i = 0_UZ; i != n_input_conns; ++i) {
    out += L"          " + LAYER_CONN_NAME[E] + L" " + std::to_wstring(connections[i] - first_inp_unit) + CRLF;
    out += L"          weight[" + std::to_wstring(block_it->first_index_feedforward_connection_forget_gate + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
  }
  //    [1] |END| Forget gate. |END|

  //    [1] Output gate.
  params = this->ptr_array_parameters + block_it->first_index_feedforward_connection_output_gate;

  for (i = 0_UZ; i != n_input_conns; ++i) {
    out += L"          " + LAYER_CONN_NAME[E] + L" " + std::to_wstring(connections[i] - first_inp_unit) + CRLF;
    out += L"          weight[" + std::to_wstring(block_it->first_index_feedforward_connection_output_gate + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
  }
  //    [1] |END| Output gate. |END|
  // [0] |END| Input, gates. |END|

  // [0] Recurrent, gates.
  //    [1] Input gate.
  params = this->ptr_array_parameters + block_it->first_index_recurrent_connection_input_gate;

  for (i = 0_UZ; i != n_state_conns; ++i) {
    out += L"          connected_to_cell " + std::to_wstring(state_connections[i] - this->ptr_array_cell_units) + CRLF;
    out += L"          weight[" + std::to_wstring(block_it->first_index_recurrent_connection_input_gate + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
  }
  //    [1] |END| Input gate. |END|

  //    [1] Forget gate.
  params = this->ptr_array_parameters + block_it->first_index_recurrent_connection_forget_gate;

  for (i = 0_UZ; i != n_state_conns; ++i) {
    out += L"          connected_to_cell " + std::to_wstring(state_connections[i] - this->ptr_array_cell_units) + CRLF;
    out += L"          weight[" + std::to_wstring(block_it->first_index_recurrent_connection_forget_gate + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
  }
  //    [1] |END| Forget gate. |END|

  //    [1] Output gate.
  params = this->ptr_array_parameters + block_it->first_index_recurrent_connection_output_gate;

  for (i = 0_UZ; i != n_state_conns; ++i) {
    out += L"          connected_to_cell " + std::to_wstring(state_connections[i] - this->ptr_array_cell_units) + CRLF;
    out += L"          weight[" + std::to_wstring(block_it->first_index_recurrent_connection_output_gate + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
  }
  //    [1] |END| Output gate. |END|
  // [0] |END| Recurrent, gates. |END|

#ifndef NO_PEEPHOLE
  // [0] Peepholes.
  //    [1] Input gate.
  params = this->ptr_array_parameters + block_it->first_index_peephole_input_gate;

  for (i = 0_UZ; i != n_peephole_conns; ++i) {
    out += L"          connected_to_cell " + std::to_wstring(peephole_connections[i] - this->ptr_array_cell_units) + CRLF;
    out += L"          weight[" + std::to_wstring(block_it->first_index_peephole_input_gate + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
  }
  //    [1] |END| Input gate. |END|

  //    [1] Forget gate.
  params = this->ptr_array_parameters + block_it->first_index_peephole_forget_gate;

  for (i = 0_UZ; i != n_peephole_conns; ++i) {
    out += L"          connected_to_cell " + std::to_wstring(peephole_connections[i] - this->ptr_array_cell_units) + CRLF;
    out += L"          weight[" + std::to_wstring(block_it->first_index_peephole_forget_gate + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
  }
  //    [1] |END| Forget gate. |END|

  //    [1] Output gate.
  params = this->ptr_array_parameters + block_it->first_index_peephole_output_gate;

  for (i = 0_UZ; i != n_peephole_conns; ++i) {
    out += L"          connected_to_cell " + std::to_wstring(peephole_connections[i] - this->ptr_array_cell_units) + CRLF;
    out += L"          weight[" + std::to_wstring(block_it->first_index_peephole_output_gate + i) + L"] " + to_wstring(cast(params[i])) + CRLF;
  }
  //    [1] |END| Output gate. |END|
  // [0] |END| Peepholes. |END|
#endif
  // clang-format on
}
}  // namespace DL::v1
