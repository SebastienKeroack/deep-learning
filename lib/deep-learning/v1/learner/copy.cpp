/* Copyright 2016, 2022 Sébastien Kéroack. All Rights Reserved.

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
#include "deep-learning/io/logger.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"

using namespace DL::Str;

namespace DL::v1 {
Model &Model::operator=(Model const &ref_source_Neural_Network_received) {
  if (&ref_source_Neural_Network_received != this)
    this->copy(ref_source_Neural_Network_received);
  return *this;
}

bool Model::copy(Model const &ref_source_Neural_Network_received,
                 bool const initialize_parallel_computation_received,
                 bool const copy_delta_optimizer_received,
                 size_t const allowable_memory) {
  /*
#ifdef COMPILE_CUDA
  if(ref_source_Neural_Network_received.is_cu_initialized
     &&
     ref_source_Neural_Network_received.is_update_from_device == false)
  {
      if(ref_source_Neural_Network_received.Copy__Parameters__Device_To_Host()
== false)
      {
          ERR(L"An error has been triggered from the
              L"`Copy__Parameters__Device_To_Host()` function.");
          return false;
      }
      else if(ref_source_Neural_Network_received.Use__Normalization() &&
ref_source_Neural_Network_received.Copy__Batch_Normalization_Neurons__Device_To_Host()
== false)
      {
          ERR(L"An error has been triggered from the
              L"`Copy__Batch_Normalization_Neurons__Device_To_Host()` function.");
          return false;
      }
  }
#endif
  */

  this->clear();

  if (this->Allocate__Structure(ref_source_Neural_Network_received.total_layers,
                                allowable_memory != 0_UZ
                                    ? allowable_memory
                                    : ref_source_Neural_Network_received
                                          .maximum_allowable_memory_bytes) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Structure(%zu, %zu)` function.",
        ref_source_Neural_Network_received.total_layers,
        allowable_memory != 0_UZ ? allowable_memory
                                 : ref_source_Neural_Network_received
                                       .maximum_allowable_memory_bytes);

    return false;
  }

  // General parameters.
  this->type = ref_source_Neural_Network_received.type;
  this->seq_w = ref_source_Neural_Network_received.seq_w;
  // |END| General parameters. |END|

  // Loss parameters.
  *this->ptr_array_number_bit_fail =
      *ref_source_Neural_Network_received.ptr_array_number_bit_fail;
  *this->ptr_array_number_loss =
      *ref_source_Neural_Network_received.ptr_array_number_loss;
  *this->ptr_array_loss_values =
      *ref_source_Neural_Network_received.ptr_array_loss_values;

  this->Copy__Loss(&ref_source_Neural_Network_received);
  // |END| Loss parameters. |END|

  // Accuracy parameters.
  this->n_acc_trial = ref_source_Neural_Network_received.n_acc_trial;
  this->ptr_array_accuracy_values[0][0] =
      ref_source_Neural_Network_received.ptr_array_accuracy_values[0][0];
  this->ptr_array_accuracy_values[1][0] =
      ref_source_Neural_Network_received.ptr_array_accuracy_values[1][0];
  this->ptr_array_accuracy_values[2][0] =
      ref_source_Neural_Network_received.ptr_array_accuracy_values[2][0];
  this->ptr_array_accuracy_values[3][0] =
      ref_source_Neural_Network_received.ptr_array_accuracy_values[3][0];
  this->ptr_array_accuracy_values[4][0] =
      ref_source_Neural_Network_received.ptr_array_accuracy_values[4][0];

  this->Copy__Accuracy(&ref_source_Neural_Network_received);
  // |END| Accuracy parameters. |END|

  // Dimension.
  this->total_layers = ref_source_Neural_Network_received.total_layers;
  this->n_inp = ref_source_Neural_Network_received.n_inp;
  this->n_out = ref_source_Neural_Network_received.n_out;
  this->total_basic_units =
      ref_source_Neural_Network_received.total_basic_units;
  this->total_basic_indice_units =
      ref_source_Neural_Network_received.total_basic_indice_units;
  this->total_neuron_units =
      ref_source_Neural_Network_received.total_neuron_units;
  this->total_AF_units = ref_source_Neural_Network_received.total_AF_units;
  this->total_AF_Ind_recurrent_units =
      ref_source_Neural_Network_received.total_AF_Ind_recurrent_units;
  this->total_block_units =
      ref_source_Neural_Network_received.total_block_units;
  this->total_cell_units = ref_source_Neural_Network_received.total_cell_units;
  this->total_parameters = ref_source_Neural_Network_received.total_weights +
                           ref_source_Neural_Network_received.total_bias;
  this->total_weights = ref_source_Neural_Network_received.total_weights;
  this->total_bias = ref_source_Neural_Network_received.total_bias;

  if (this->Set__Input_Mode(
          ref_source_Neural_Network_received.use_first_layer_as_input) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Input_Mode(%ls)` function.",
        to_wstring(ref_source_Neural_Network_received.use_first_layer_as_input)
            .c_str());
    return false;
  } else if (this->Set__Output_Mode(
                 ref_source_Neural_Network_received.use_last_layer_as_output) ==
             false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Output_Mode(%ls)` function.",
        to_wstring(ref_source_Neural_Network_received.use_last_layer_as_output)
            .c_str());
    return false;
  }

  Layer const *const tmp_ptr_destination_last_layer(this->ptr_last_layer),
      *tmp_ptr_destination_previous_layer_it, *tmp_ptr_source_layer_it;
  Layer *tmp_ptr_destination_layer_it;

  for (tmp_ptr_source_layer_it =
           ref_source_Neural_Network_received.ptr_array_layers,
      tmp_ptr_destination_layer_it = this->ptr_array_layers;
       tmp_ptr_destination_layer_it != tmp_ptr_destination_last_layer;
       ++tmp_ptr_destination_layer_it, ++tmp_ptr_source_layer_it) {
    tmp_ptr_destination_layer_it->type_layer =
        tmp_ptr_source_layer_it->type_layer;

    // Pooling.
    tmp_ptr_destination_layer_it->pooling_values[0] =
        tmp_ptr_source_layer_it->pooling_values[0];
    tmp_ptr_destination_layer_it->pooling_values[1] =
        tmp_ptr_source_layer_it->pooling_values[1];
    tmp_ptr_destination_layer_it->pooling_values[2] =
        tmp_ptr_source_layer_it->pooling_values[2];
    tmp_ptr_destination_layer_it->pooling_values[3] =
        tmp_ptr_source_layer_it->pooling_values[3];
    tmp_ptr_destination_layer_it->pooling_values[4] =
        tmp_ptr_source_layer_it->pooling_values[4];
    // |END Pooling. |END|

    tmp_ptr_destination_layer_it->type_activation =
        tmp_ptr_source_layer_it->type_activation;

    tmp_ptr_destination_layer_it->block_depth =
        tmp_ptr_source_layer_it->block_depth;

    *tmp_ptr_destination_layer_it->ptr_number_outputs =
        *tmp_ptr_source_layer_it->ptr_number_outputs;

    *tmp_ptr_destination_layer_it->ptr_first_connection_index =
        *tmp_ptr_source_layer_it->ptr_first_connection_index;
    *tmp_ptr_destination_layer_it->ptr_last_connection_index =
        *tmp_ptr_source_layer_it->ptr_last_connection_index;

    tmp_ptr_destination_layer_it->first_bias_connection_index =
        tmp_ptr_source_layer_it->first_bias_connection_index;
    tmp_ptr_destination_layer_it->last_bias_connection_index =
        tmp_ptr_source_layer_it->last_bias_connection_index;

    // Basic unit(s).
    tmp_ptr_destination_layer_it->ptr_last_basic_unit =
        tmp_ptr_destination_layer_it->ptr_array_basic_units +
        static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_basic_unit -
                            tmp_ptr_source_layer_it->ptr_array_basic_units);
    // |END| Basic unit(s). |END|

    // Basic indice unit(s).
    tmp_ptr_destination_layer_it->ptr_last_basic_indice_unit =
        tmp_ptr_destination_layer_it->ptr_array_basic_indice_units +
        static_cast<size_t>(
            tmp_ptr_source_layer_it->ptr_last_basic_indice_unit -
            tmp_ptr_source_layer_it->ptr_array_basic_indice_units);
    // |END| Basic indice unit(s). |END|

    // Neuron unit(s).
    tmp_ptr_destination_layer_it->ptr_last_neuron_unit =
        tmp_ptr_destination_layer_it->ptr_array_neuron_units +
        static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_neuron_unit -
                            tmp_ptr_source_layer_it->ptr_array_neuron_units);
    // |END| Neuron unit(s). |END|

    // AF unit(s).
    tmp_ptr_destination_layer_it->ptr_last_AF_unit =
        tmp_ptr_destination_layer_it->ptr_array_AF_units +
        static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_AF_unit -
                            tmp_ptr_source_layer_it->ptr_array_AF_units);
    // |END| AF unit(s). |END|

    // AF Ind recurrent unit(s).
    tmp_ptr_destination_layer_it->ptr_last_AF_Ind_recurrent_unit =
        tmp_ptr_destination_layer_it->ptr_array_AF_Ind_recurrent_units +
        static_cast<size_t>(
            tmp_ptr_source_layer_it->ptr_last_AF_Ind_recurrent_unit -
            tmp_ptr_source_layer_it->ptr_array_AF_Ind_recurrent_units);
    // |END| AF Ind recurrent unit(s). |END|

    // Block unit(s).
    tmp_ptr_destination_layer_it->ptr_last_block_unit =
        tmp_ptr_destination_layer_it->ptr_array_block_units +
        static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_block_unit -
                            tmp_ptr_source_layer_it->ptr_array_block_units);
    // |END| Block unit(s). |END|

    // Cell unit(s).
    tmp_ptr_destination_layer_it->ptr_last_cell_unit =
        tmp_ptr_destination_layer_it->ptr_array_cell_units +
        static_cast<size_t>(tmp_ptr_source_layer_it->ptr_last_cell_unit -
                            tmp_ptr_source_layer_it->ptr_array_cell_units);
    // |END| Cell unit(s). |END|
  }
  // |END| Dimension. |END|

  // Layers, connections.
  this->Order__Layers__Connection();

  if (this->Allocate__Basic_Units() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Basic_Units()` function.");
    return false;
  } else if (this->Allocate__Basic_Indice_Units() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Basic_Indice_Units()` function.");
    return false;
  } else if (this->Allocate__Neuron_Units() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Neuron_Units()` function.");
    return false;
  } else if (this->Allocate__AF_Units() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__AF_Units()` function.");
    return false;
  } else if (this->Allocate__AF_Ind_Recurrent_Units() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__AF_Ind_Recurrent_Units()` function.");
    return false;
  } else if (this->Allocate__LSTM_Layers() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__LSTM_Layers()` function.");
    return false;
  } else if (this->Allocate__Bidirectional__Layers() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Bidirectional__Layers()` function.");
    return false;
  } else if (this->Allocate__Parameter() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Parameter()` function.");
    return false;
  }

  this->Copy__Neuron_Units(
      0_UZ, this->total_neuron_units,
      ref_source_Neural_Network_received.ptr_array_neuron_units);

  this->Copy__AF_Units(0_UZ, this->total_AF_units,
                       ref_source_Neural_Network_received.ptr_array_AF_units);

  this->Copy__AF_Ind_Recurrent_Units(
      0_UZ, this->total_AF_Ind_recurrent_units,
      ref_source_Neural_Network_received.ptr_array_AF_Ind_recurrent_units);

  this->Order__Layers__Output();

  // copy connections.
  for (tmp_ptr_source_layer_it =
           ref_source_Neural_Network_received.ptr_array_layers + 1,
      tmp_ptr_destination_layer_it = this->ptr_array_layers + 1;
       tmp_ptr_destination_layer_it != tmp_ptr_destination_last_layer;
       ++tmp_ptr_destination_layer_it, ++tmp_ptr_source_layer_it) {
    // If the current layer is a pooling/residual layer, continue.
    if (tmp_ptr_destination_layer_it->type_layer == LAYER::AVERAGE_POOLING ||
        tmp_ptr_destination_layer_it->type_layer == LAYER::MAX_POOLING ||
        tmp_ptr_destination_layer_it->type_layer == LAYER::RESIDUAL) {
      continue;
    }

    tmp_ptr_destination_previous_layer_it =
        tmp_ptr_destination_layer_it->previous_connected_layers[0];

    if (tmp_ptr_destination_layer_it->type_layer ==
        LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT) {
      this->Copy__Layer__AF_Ind_Recurrent(
          tmp_ptr_source_layer_it,
          ref_source_Neural_Network_received.ptr_array_AF_Ind_recurrent_units,
          this->ptr_array_AF_Ind_recurrent_units,
          reinterpret_cast<AF_Ind_recurrent_unit **>(
              ref_source_Neural_Network_received.ptr_array_ptr_connections),
          reinterpret_cast<AF_Ind_recurrent_unit **>(
              this->ptr_array_ptr_connections));
    }

    switch (tmp_ptr_destination_previous_layer_it->type_layer) {
      case LAYER::AVERAGE_POOLING:
      case LAYER::RESIDUAL:
        switch (tmp_ptr_destination_layer_it->type_layer) {
          case LAYER::FULLY_CONNECTED:
          case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          case LAYER::FULLY_CONNECTED_RECURRENT:
            this->Copy__Layer__FC<Basic_unit>(
                tmp_ptr_source_layer_it, tmp_ptr_destination_layer_it,
                ref_source_Neural_Network_received.ptr_array_basic_units,
                this->ptr_array_basic_units,
                reinterpret_cast<Basic_unit **>(
                    ref_source_Neural_Network_received
                        .ptr_array_ptr_connections),
                reinterpret_cast<Basic_unit **>(
                    this->ptr_array_ptr_connections));
            break;
          case LAYER::LSTM:
            this->Copy__Layer__LSTM<Basic_unit>(
                tmp_ptr_source_layer_it, tmp_ptr_destination_layer_it,
                ref_source_Neural_Network_received.ptr_array_cell_units,
                ref_source_Neural_Network_received.ptr_array_basic_units,
                this->ptr_array_basic_units,
                ref_source_Neural_Network_received.ptr_array_ptr_connections,
                this->ptr_array_ptr_connections);
            break;
          default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                tmp_ptr_destination_layer_it->type_layer,
                LAYER_NAME[tmp_ptr_destination_layer_it->type_layer].c_str());
            return false;
        }
        break;
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
      case LAYER::FULLY_CONNECTED_RECURRENT:
        switch (tmp_ptr_destination_layer_it->type_layer) {
          case LAYER::FULLY_CONNECTED:
          case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          case LAYER::FULLY_CONNECTED_RECURRENT:
            this->Copy__Layer__FC<Neuron_unit>(
                tmp_ptr_source_layer_it, tmp_ptr_destination_layer_it,
                ref_source_Neural_Network_received.ptr_array_neuron_units,
                this->ptr_array_neuron_units,
                reinterpret_cast<Neuron_unit **>(
                    ref_source_Neural_Network_received
                        .ptr_array_ptr_connections),
                reinterpret_cast<Neuron_unit **>(
                    this->ptr_array_ptr_connections));
            break;
          case LAYER::LSTM:
            this->Copy__Layer__LSTM<Neuron_unit>(
                tmp_ptr_source_layer_it, tmp_ptr_destination_layer_it,
                ref_source_Neural_Network_received.ptr_array_cell_units,
                ref_source_Neural_Network_received.ptr_array_neuron_units,
                this->ptr_array_neuron_units,
                ref_source_Neural_Network_received.ptr_array_ptr_connections,
                this->ptr_array_ptr_connections);
            break;
          default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                tmp_ptr_destination_layer_it->type_layer,
                LAYER_NAME[tmp_ptr_destination_layer_it->type_layer].c_str());
            return false;
        }
        break;
      case LAYER::LSTM:
        switch (tmp_ptr_destination_layer_it->type_layer) {
          case LAYER::FULLY_CONNECTED:
          case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          case LAYER::FULLY_CONNECTED_RECURRENT:
            this->Copy__Layer__FC<CellUnit>(
                tmp_ptr_source_layer_it, tmp_ptr_destination_layer_it,
                ref_source_Neural_Network_received.ptr_array_cell_units,
                this->ptr_array_cell_units,
                reinterpret_cast<CellUnit **>(ref_source_Neural_Network_received
                                                  .ptr_array_ptr_connections),
                reinterpret_cast<CellUnit **>(this->ptr_array_ptr_connections));
            break;
          case LAYER::LSTM:
            this->Copy__Layer__LSTM<CellUnit>(
                tmp_ptr_source_layer_it, tmp_ptr_destination_layer_it,
                ref_source_Neural_Network_received.ptr_array_cell_units,
                ref_source_Neural_Network_received.ptr_array_cell_units,
                this->ptr_array_cell_units,
                ref_source_Neural_Network_received.ptr_array_ptr_connections,
                this->ptr_array_ptr_connections);
            break;
          default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                tmp_ptr_destination_layer_it->type_layer,
                LAYER_NAME[tmp_ptr_destination_layer_it->type_layer].c_str());
            return false;
        }
        break;
      case LAYER::MAX_POOLING:
        switch (tmp_ptr_destination_layer_it->type_layer) {
          case LAYER::FULLY_CONNECTED:
          case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          case LAYER::FULLY_CONNECTED_RECURRENT:
            this->Copy__Layer__FC<Basic_indice_unit>(
                tmp_ptr_source_layer_it, tmp_ptr_destination_layer_it,
                ref_source_Neural_Network_received.ptr_array_basic_indice_units,
                this->ptr_array_basic_indice_units,
                reinterpret_cast<Basic_indice_unit **>(
                    ref_source_Neural_Network_received
                        .ptr_array_ptr_connections),
                reinterpret_cast<Basic_indice_unit **>(
                    this->ptr_array_ptr_connections));
            break;
          case LAYER::LSTM:
            this->Copy__Layer__LSTM<Basic_indice_unit>(
                tmp_ptr_source_layer_it, tmp_ptr_destination_layer_it,
                ref_source_Neural_Network_received.ptr_array_cell_units,
                ref_source_Neural_Network_received.ptr_array_basic_indice_units,
                this->ptr_array_basic_indice_units,
                ref_source_Neural_Network_received.ptr_array_ptr_connections,
                this->ptr_array_ptr_connections);
            break;
          default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                tmp_ptr_destination_layer_it->type_layer,
                LAYER_NAME[tmp_ptr_destination_layer_it->type_layer].c_str());
            return false;
        }
        break;
      default:
        ERR(L"Layer type (%d | %ls) is not managed in",
            tmp_ptr_destination_previous_layer_it->type_layer,
            LAYER_NAME[tmp_ptr_destination_previous_layer_it->type_layer]
                .c_str());
        return false;
    }
  }
  // |END| copy connections. |END|

  // Dropout.
  this->Copy__Dropout(
      ref_source_Neural_Network_received.ptr_array_layers,
      ref_source_Neural_Network_received.Get__End_Layer__Active() -
          1,  // Get last active layer.
      this->ptr_array_layers);
  // |END| Dropout. |END|

  // Normalization.
  this->Copy__Normalization(
      ref_source_Neural_Network_received.ptr_array_layers +
          1,  // Skip input layer.
      ref_source_Neural_Network_received.Get__End_Layer__Active() ==
              ref_source_Neural_Network_received.ptr_last_layer
          ? (ref_source_Neural_Network_received.ptr_last_layer - 1)
          : ref_source_Neural_Network_received.Get__End_Layer__Active(),
      this->ptr_array_layers + 1);  // Skip input layer.

  this->Copy__Normalization(&ref_source_Neural_Network_received);

  this->Copy__Normalized_Units(
      0_UZ, ref_source_Neural_Network_received.total_normalized_units_allocated,
      ref_source_Neural_Network_received.ptr_array_normalized_units);
  // |END| Normalization. |END|

  // Parameters.
  VARCOPY(this->ptr_array_parameters,
          ref_source_Neural_Network_received.ptr_array_parameters,
          ref_source_Neural_Network_received.total_parameters * sizeof(var));
  // |END| Parameters. |END|

  // Initializer weight parameters.
  this->Copy__Initializer__Weight_Parameter(ref_source_Neural_Network_received);
  // |END| Initializer weight parameters. |END|

  // Training parameters.
  this->Copy__Training_Parameters(&ref_source_Neural_Network_received);
  // |END| Training parameters. |END|

  // Optimizer parameters.
  if (this->Copy__Optimizer_Parameters(&ref_source_Neural_Network_received,
                                       copy_delta_optimizer_received) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`Copy__Optimizer_Parameters(ref, %ls)` function.",
        to_wstring(copy_delta_optimizer_received).c_str());
    return false;
  }
  // |END| Optimizer parameters. |END|

  // Regularization parameters.
  this->Copy__Regularization(&ref_source_Neural_Network_received);

  if (this->type == MODEL::AUTOENCODER) {
    this->Copy__Tied_Weight(
        ref_source_Neural_Network_received.ptr_array_layers +
            1,  // Skip input layer.
        ref_source_Neural_Network_received.Get__End_Layer__Active(),
        this->ptr_array_layers + 1);  // Skip input layer.
  }

  this->Copy__Sparse_K_Filters(
      ref_source_Neural_Network_received.ptr_array_layers +
          1,  // Skip input layer.
      ref_source_Neural_Network_received.Get__End_Layer__Active() ==
              ref_source_Neural_Network_received.ptr_last_layer
          ? (ref_source_Neural_Network_received.ptr_last_layer - 1)
          : ref_source_Neural_Network_received.Get__End_Layer__Active(),
      this->ptr_array_layers + 1);  // Skip input layer.

  this->Copy__Constraint_Recurrent_Weight(
      ref_source_Neural_Network_received.ptr_array_layers +
          1,  // Skip input layer.
      ref_source_Neural_Network_received.Get__End_Layer__Active() ==
              ref_source_Neural_Network_received.ptr_last_layer
          ? (ref_source_Neural_Network_received.ptr_last_layer - 1)
          : ref_source_Neural_Network_received.Get__End_Layer__Active(),
      this->ptr_array_layers + 1);  // Skip input layer.
  // |END| Regularization parameters. |END|

  // Compute parameters.
  this->maximum_allowable_memory_bytes =
      ref_source_Neural_Network_received.maximum_allowable_memory_bytes;
  this->pct_threads = ref_source_Neural_Network_received.pct_threads;

  this->maximum_batch_size =
      ref_source_Neural_Network_received.maximum_batch_size;

  if (initialize_parallel_computation_received) {
    this->set_mp(ref_source_Neural_Network_received.use_mp);

#ifdef COMPILE_CUDA
    this->set_cu(ref_source_Neural_Network_received.use_cu,
                 ref_source_Neural_Network_received.is_cu_initialized
                     ? ref_source_Neural_Network_received.cumodel
                           ->Get__Maximum_Allowable_Memory()
                     : 0_UZ);
#endif
  } else {
    this->use_mp = ref_source_Neural_Network_received.use_mp;
    this->use_cu = ref_source_Neural_Network_received.use_cu;
  }
  // |END| Compute parameters. |END|

  return true;
}

bool Model::update(Model const &ref_source_Neural_Network_received,
                   bool const initialize_parallel_computation_received,
                   bool const update_delta_optimizer_received) {
  // Lambda: Redirect to copy.
  auto tmp_Redirect_To_Copy(
      [self = this,
       &tmp_source_Neural_Network = ref_source_Neural_Network_received,
       tmp_initialize_parallel_computation =
           initialize_parallel_computation_received,
       tmp_update_delta_optimizer = update_delta_optimizer_received]() -> bool {
        if (self->copy(tmp_source_Neural_Network,
                       tmp_initialize_parallel_computation,
                       tmp_update_delta_optimizer) == false) {
          ERR(L"An error has been triggered from the "
              L"`copy(ref, %ls, %ls)` function.",
              to_wstring(tmp_initialize_parallel_computation).c_str(),
              to_wstring(tmp_update_delta_optimizer).c_str());
          return false;
        }

        return true;
      });

  // Compare network topology. If different redirect to "copy" function.
  if (this->total_layers != ref_source_Neural_Network_received.total_layers ||
      this->total_weights != ref_source_Neural_Network_received.total_weights ||
      this->total_bias != ref_source_Neural_Network_received.total_bias ||
      this->total_neuron_units !=
          ref_source_Neural_Network_received.total_neuron_units ||
      this->total_AF_units !=
          ref_source_Neural_Network_received.total_AF_units ||
      this->total_AF_Ind_recurrent_units !=
          ref_source_Neural_Network_received.total_AF_Ind_recurrent_units ||
      this->total_cell_units !=
          ref_source_Neural_Network_received.total_cell_units ||
      this->total_block_units !=
          ref_source_Neural_Network_received.total_block_units) {
    return (tmp_Redirect_To_Copy());
  }

  /*
#ifdef COMPILE_CUDA
  if(ref_source_Neural_Network_received.is_cu_initialized
    &&
    ref_source_Neural_Network_received.is_update_from_device == false)
  {
      if(ref_source_Neural_Network_received.Copy__Parameters__Device_To_Host()
== false)
      {
          ERR(L"An error has been triggered from the
              L"`Copy__Parameters__Device_To_Host()` function.");
          return false;
      }
      else if(ref_source_Neural_Network_received.Use__Normalization() &&
ref_source_Neural_Network_received.Copy__Batch_Normalization_Neurons__Device_To_Host()
== false)
      {
          ERR(L"An error has been triggered from the
              L"`Copy__Batch_Normalization_Neurons__Device_To_Host()` function.");
          return false;
      }
  }
#endif
  */

  // Dropout.
  this->Copy__Dropout(
      ref_source_Neural_Network_received.ptr_array_layers,
      ref_source_Neural_Network_received.Get__End_Layer__Active() -
          1,  // Get last active layer.
      this->ptr_array_layers);
  // |END| Dropout. |END|

  // Normalization.
  this->Copy__Normalization(
      ref_source_Neural_Network_received.ptr_array_layers +
          1,  // Skip input layer.
      ref_source_Neural_Network_received.Get__End_Layer__Active() ==
              ref_source_Neural_Network_received.ptr_last_layer
          ? (ref_source_Neural_Network_received.ptr_last_layer - 1)
          : ref_source_Neural_Network_received.Get__End_Layer__Active(),
      this->ptr_array_layers + 1);  // Skip input layer.

  this->Copy__Normalization(&ref_source_Neural_Network_received);

  this->Copy__Normalized_Units(
      0_UZ, ref_source_Neural_Network_received.total_normalized_units_allocated,
      ref_source_Neural_Network_received.ptr_array_normalized_units);
  // |END| Normalization. |END|

  // Loss parameters.
  this->Copy__Loss(&ref_source_Neural_Network_received);
  // |END| Loss parameters. |END|

  // Accuracy parameters.
  this->Copy__Accuracy(&ref_source_Neural_Network_received);
  // |END| Accuracy parameters. |END|

  // Compare total parameters (Can be modified by normalization).
  if (this->total_parameters !=
      ref_source_Neural_Network_received.total_parameters) {
    return (tmp_Redirect_To_Copy());
  }

  // Parameters.
  VARCOPY(this->ptr_array_parameters,
          ref_source_Neural_Network_received.ptr_array_parameters,
          ref_source_Neural_Network_received.total_parameters * sizeof(var));
  // |END| Parameters. |END|

  // Initializer weight parameters.
  this->Copy__Initializer__Weight_Parameter(ref_source_Neural_Network_received);
  // |END| Initializer weight parameters. |END|

  // Training parameters.
  this->Copy__Training_Parameters(&ref_source_Neural_Network_received);
  // |END| Training parameters. |END|

  // Optimizer parameters.
  if (this->Copy__Optimizer_Parameters(&ref_source_Neural_Network_received,
                                       update_delta_optimizer_received) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`Copy__Optimizer_Parameters(ref, %ls)` function.",
        to_wstring(update_delta_optimizer_received).c_str());
    return false;
  }
  // |END| Optimizer parameters. |END|

  // Regularization parameters.
  this->Copy__Regularization(&ref_source_Neural_Network_received);

  if (this->type == MODEL::AUTOENCODER) {
    this->Copy__Tied_Weight(
        ref_source_Neural_Network_received.ptr_array_layers +
            1,  // Skip input layer.
        ref_source_Neural_Network_received.Get__End_Layer__Active(),
        this->ptr_array_layers + 1);  // Skip input layer.
  }

  this->Copy__Sparse_K_Filters(
      ref_source_Neural_Network_received.ptr_array_layers +
          1,  // Skip input layer.
      ref_source_Neural_Network_received.Get__End_Layer__Active() ==
              ref_source_Neural_Network_received.ptr_last_layer
          ? (ref_source_Neural_Network_received.ptr_last_layer - 1)
          : ref_source_Neural_Network_received.Get__End_Layer__Active(),
      this->ptr_array_layers + 1);  // Skip input layer.

  this->Copy__Constraint_Recurrent_Weight(
      ref_source_Neural_Network_received.ptr_array_layers +
          1,  // Skip input layer.
      ref_source_Neural_Network_received.Get__End_Layer__Active() ==
              ref_source_Neural_Network_received.ptr_last_layer
          ? (ref_source_Neural_Network_received.ptr_last_layer - 1)
          : ref_source_Neural_Network_received.Get__End_Layer__Active(),
      this->ptr_array_layers + 1);  // Skip input layer.
  // |END| Regularization parameters. |END|

  // Compute parameters.
  this->maximum_allowable_memory_bytes =
      ref_source_Neural_Network_received.maximum_allowable_memory_bytes;

  if (this->Set__Maximum_Thread_Usage(
          ref_source_Neural_Network_received.pct_threads) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Maximum_Thread_Usage(%f)` function.",
        ref_source_Neural_Network_received.pct_threads);
    return false;
  } else if (this->set_max_batch_size(
                 ref_source_Neural_Network_received.maximum_batch_size) ==
             false) {
    ERR(L"An error has been triggered from the "
        L"`set_max_batch_size(%zu)` function.",
        ref_source_Neural_Network_received.maximum_batch_size);
    return false;
  }

  if (initialize_parallel_computation_received) {
    this->set_mp(ref_source_Neural_Network_received.use_mp);

#ifdef COMPILE_CUDA
    this->set_cu(ref_source_Neural_Network_received.use_cu,
                 ref_source_Neural_Network_received.is_cu_initialized
                     ? ref_source_Neural_Network_received.cumodel
                           ->Get__Maximum_Allowable_Memory()
                     : 0_UZ);
#endif
  } else {
    this->use_mp = ref_source_Neural_Network_received.use_mp;
    this->use_cu = ref_source_Neural_Network_received.use_cu;
  }
  // |END| Compute parameters. |END|

  return true;
}

void Model::Copy__Warm_Restarts_Parameters(Model const *const model) {
  this->use_warm_restarts = model->use_warm_restarts;
  this->warm_restarts_decay_learning_rate =
      model->warm_restarts_decay_learning_rate;
  this->warm_restarts_initial_maximum_learning_rate =
      model->warm_restarts_initial_maximum_learning_rate;
  this->warm_restarts_maximum_learning_rate =
      model->warm_restarts_maximum_learning_rate;
  this->warm_restarts_minimum_learning_rate =
      model->warm_restarts_minimum_learning_rate;
  this->warm_restarts_initial_T_i = model->warm_restarts_initial_T_i;
  this->warm_restarts_T_i = model->warm_restarts_T_i;
  this->warm_restarts_multiplier = model->warm_restarts_multiplier;
}

bool Model::Copy__Optimizer_Parameters(
    Model const *const model, bool const copy_delta_optimizer_received) {
  switch (this->type_optimizer_function) {
    case OPTIMIZER::GD:
      this->Copy__Gradient_Descent_Parameters(model);
      break;
    case OPTIMIZER::IRPROP_MINUS:
      this->Copy__RPROP_minus_Parameters(model);
      break;
    case OPTIMIZER::IRPROP_PLUS:
      this->Copy__RPROP_plus_Parameters(model);
      break;
    case OPTIMIZER::SARPROP:
      this->Copy__SARProp_Parameters(model);
      break;
    case OPTIMIZER::QUICKPROP:
      this->Copy__QuickProp_Parameters(model);
      break;
    case OPTIMIZER::ADAM:
    case OPTIMIZER::ADAMAX:
    case OPTIMIZER::AMSGRAD:
      this->Copy__Adam_Parameters(model);
      break;
    case OPTIMIZER::NOSADAM:
      this->Copy__NosAdam_Parameters(model);
      break;
    case OPTIMIZER::ADABOUND:
    case OPTIMIZER::AMSBOUND:
      this->Copy__AdaBound_Parameters(model);
      break;
    default:
      ERR(L"Can not copy parameters of the optimizer (%d | %ls).",
          this->type_optimizer_function,
          OPTIMIZER_NAME[this->type_optimizer_function].c_str());
      return false;
  }

  if (copy_delta_optimizer_received) {
    switch (this->type_optimizer_function) {
      case OPTIMIZER::GD:
        if (this->Copy__Delta__Gradient_Descent(model) == false) {
          ERR(L"An error has been triggered from the "
              L"`Copy__Delta__Gradient_Descent(ptr)` function.");
          return false;
        }
        break;
      case OPTIMIZER::IRPROP_MINUS:
        if (this->Copy__Delta__iRPROP_minus(model) == false) {
          ERR(L"An error has been triggered from the "
              L"`Copy__Delta__iRPROP_minus(ptr)` function.");
          return false;
        }
        break;
      case OPTIMIZER::IRPROP_PLUS:
        if (this->Copy__Delta__iRPROP_plus(model) == false) {
          ERR(L"An error has been triggered from the "
              L"`Copy__Delta__iRPROP_plus(ptr)` function.");
          return false;
        }
        break;
      case OPTIMIZER::ADABOUND:
      case OPTIMIZER::ADAM:
      case OPTIMIZER::ADAMAX:
      case OPTIMIZER::NOSADAM:
        if (this->Copy__Delta__Adam(model) == false) {
          ERR(L"An error has been triggered from the "
              L"`Copy__Delta__Adam(ptr)` function.");
          return false;
        }
        break;
      case OPTIMIZER::AMSGRAD:
      case OPTIMIZER::AMSBOUND:
        if (this->Copy__Delta__AMSGrad(model) == false) {
          ERR(L"An error has been triggered from the "
              L"`Copy__Delta__AMSGrad(ptr)` function.");
          return false;
        }
        break;
      default:
        ERR(L"Can not allocate parameters of the optimizer (%d | %ls).",
            this->type_optimizer_function,
            OPTIMIZER_NAME[this->type_optimizer_function].c_str());
        return false;
    }
  }

  this->Copy__Warm_Restarts_Parameters(model);

  this->optimizer_time_step = model->optimizer_time_step;
  this->epoch_time_step = model->epoch_time_step;

#ifdef COMPILE_CUDA
  if (this->is_cu_initialized)
    this->cumodel->Copy__Optimizer_Parameters(this);
#endif

  return true;
}

void Model::Copy__Gradient_Descent_Parameters(Model const *const model) {
  // Gradient descent parameters.
  real const lr_mom(this->learning_momentum);

  this->learning_rate = model->learning_rate;
  this->learning_momentum = model->learning_momentum;
  this->use_nesterov = model->use_nesterov;

  if (lr_mom == 0_r) {
    this->Allocate__Parameter__Gradient_Descent();
  } else if (this->learning_momentum == 0_r) {
    this->Deallocate__Parameter__Gradient_Descent();
  }
  // |END| Gradient descent parameters. |END|
}

void Model::Copy__QuickProp_Parameters(Model const *const model) {
  // Quickprop parameters.
  this->quickprop_decay = model->quickprop_decay;
  this->quickprop_mu = model->quickprop_mu;
  // |END| Quickprop parameters. |END|
}

void Model::Copy__RPROP_minus_Parameters(Model const *const model) {
  // Resillent propagation minus parameters.
  this->rprop_increase_factor = model->rprop_increase_factor;
  this->rprop_decrease_factor = model->rprop_decrease_factor;
  this->rprop_delta_min = model->rprop_delta_min;
  this->rprop_delta_max = model->rprop_delta_max;
  this->rprop_delta_zero = model->rprop_delta_zero;
  // |END| Resillent propagation minus parameters. |END|
}

void Model::Copy__RPROP_plus_Parameters(Model const *const model) {
  // Resillent propagation plus parameters.
  this->Copy__RPROP_minus_Parameters(model);

  this->loss_rprop = model->loss_rprop;
  this->loss_rprop_tm1 = model->loss_rprop_tm1;
  // |END| Resillent propagation plus parameters. |END|
}

void Model::Copy__SARProp_Parameters(Model const *const model) {
  // SARProp parameters.
  this->sarprop_weight_decay_shift = model->sarprop_weight_decay_shift;
  this->sarprop_step_error_threshold_factor =
      model->sarprop_step_error_threshold_factor;
  this->sarprop_step_error_shift = model->sarprop_step_error_shift;
  this->sarprop_temperature = model->sarprop_temperature;
  this->sarprop_epoch = model->sarprop_epoch;
  // |END| SARProp parameters. |END|
}

void Model::Copy__Adam_Parameters(Model const *const model) {
  // Adam parameters.
  this->adam_learning_rate = model->adam_learning_rate;
  this->adam_beta1 = model->adam_beta1;
  this->adam_beta2 = model->adam_beta2;
  this->adam_epsilon = model->adam_epsilon;
  this->use_adam_bias_correction = model->use_adam_bias_correction;
  // |END| Adam parameters. |END|
}

void Model::Copy__NosAdam_Parameters(Model const *const model) {
  // Adam parameters.
  this->Copy__Adam_Parameters(model);

  this->adam_gamma = model->adam_gamma;
  // |END| Adam parameters. |END|
}

void Model::Copy__AdaBound_Parameters(Model const *const model) {
  // Adam parameters.
  this->Copy__Adam_Parameters(model);

  this->learning_rate_final = model->learning_rate_final;

  this->learning_gamma = model->learning_gamma;
  // |END| Adam parameters. |END|
}

bool Model::Copy__Delta__Gradient_Descent(Model const *const model) {
  if (this->learning_momentum == 0_r) {
    return true;
  } else if (model->ptr_array_previous_delta_parameters == nullptr) {
    ERR(L"Source array `ptr_array_previous_delta_parameters` "
        L"is a nullptr.");
    return false;
  } else if (this->ptr_array_previous_delta_parameters == nullptr) {
    ERR(L"Destination array `ptr_array_previous_delta_parameters` "
        L"is a nullptr.");
    return false;
  }

  memcpy(this->ptr_array_previous_delta_parameters,
         model->ptr_array_previous_delta_parameters,
         this->total_parameters * sizeof(real));

  return true;
}

bool Model::Copy__Delta__iRPROP_minus(Model const *const model) {
  if (model->ptr_array_previous_steps == nullptr) {
    ERR(L"Source array `ptr_array_previous_steps` is a nullptr.");
    return false;
  } else if (this->ptr_array_previous_steps == nullptr) {
    ERR(L"Destination array `ptr_array_previous_steps` is a nullptr.");
    return false;
  } else {
    memcpy(this->ptr_array_previous_steps, model->ptr_array_previous_steps,
           this->total_parameters * sizeof(real));
  }

  if (model->ptr_array_previous_derivatives_parameters == nullptr) {
    ERR(L"Source array `ptr_array_previous_derivatives_parameters` "
        L"is a nullptr.");
    return false;
  } else if (this->ptr_array_previous_derivatives_parameters == nullptr) {
    ERR(L"Destination array `ptr_array_previous_derivatives_parameters` "
        L"is a nullptr.");
    return false;
  } else {
    memcpy(this->ptr_array_previous_derivatives_parameters,
           model->ptr_array_previous_derivatives_parameters,
           this->total_parameters * sizeof(real));
  }

  return true;
}

bool Model::Copy__Delta__iRPROP_plus(Model const *const model) {
  if (this->Copy__Delta__iRPROP_minus(model) == false) {
    ERR(L"An error has been triggered from the "
        L"`Copy__Delta__iRPROP_minus()` function.");
    return false;
  }

  if (model->ptr_array_previous_delta_parameters == nullptr) {
    ERR(L"Source array `ptr_array_previous_delta_parameters` "
        L"is a nullptr.");
    return false;
  } else if (this->ptr_array_previous_delta_parameters == nullptr) {
    ERR(L"Destination array `ptr_array_previous_delta_parameters` "
        L"is a nullptr.");
    return false;
  } else {
    memcpy(this->ptr_array_previous_delta_parameters,
           model->ptr_array_previous_delta_parameters,
           this->total_parameters * sizeof(real));
  }

  return true;
}

bool Model::Copy__Delta__Adam(Model const *const model) {
  if (model->ptr_array_previous_biased_first_moment == nullptr) {
    ERR(L"Source array `ptr_array_previous_biased_first_moment` "
        L"is a nullptr.");
    return false;
  } else if (this->ptr_array_previous_biased_first_moment == nullptr) {
    ERR(L"Destination array `ptr_array_previous_biased_first_moment` "
        L"is a nullptr.");
    return false;
  } else {
    memcpy(this->ptr_array_previous_biased_first_moment,
           model->ptr_array_previous_biased_first_moment,
           this->total_parameters * sizeof(real));
  }

  if (model->ptr_array_previous_biased_second_moment == nullptr) {
    ERR(L"Source array `ptr_array_previous_biased_second_moment` "
        L"is a nullptr.");
    return false;
  } else if (this->ptr_array_previous_biased_second_moment == nullptr) {
    ERR(L"Destination array `ptr_array_previous_biased_second_moment` "
        L"is a nullptr.");
    return false;
  } else {
    memcpy(this->ptr_array_previous_biased_second_moment,
           model->ptr_array_previous_biased_second_moment,
           this->total_parameters * sizeof(real));
  }

  return true;
}

bool Model::Copy__Delta__AMSGrad(Model const *const model) {
  if (this->Copy__Delta__Adam(model) == false) {
    ERR(L"An error has been triggered from the "
        L"`Copy__Delta__Adam()` function.");
    return false;
  }

  if (model->ptr_array_previous_biased_second_moment_hat == nullptr) {
    ERR(L"Source array `ptr_array_previous_biased_second_moment_hat` "
        L"is a nullptr.");
    return false;
  } else if (this->ptr_array_previous_biased_second_moment_hat == nullptr) {
    ERR(L"Destination array `ptr_array_previous_biased_second_moment_hat` "
        L"is a nullptr.");
    return false;
  } else {
    memcpy(this->ptr_array_previous_biased_second_moment_hat,
           model->ptr_array_previous_biased_second_moment_hat,
           this->total_parameters * sizeof(real));
  }

  return true;
}

void Model::Copy__Training_Parameters(Model const *const model) {
  this->set_optimizer(model->type_optimizer_function);
  this->set_loss_fn(model->type_loss_function);
  this->set_accu_fn(model->type_accuracy_function);
  this->set_bit_fail_limit(model->bit_fail_limit);

  if (this->set_clip_gradient(model->clip_gradient) == false) {
    ERR(L"An error has been triggered from the "
        L"`set_clip_gradient(%f)` function.",
        model->clip_gradient);
  } else if (this->Set__Pre_Training_Level(model->pre_training_level) ==
             false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Pre_Training_Level(%zu)` function.",
        model->pre_training_level);
  } else if (this->set_seq_w(model->n_time_delay) == false) {
    ERR(L"An error has been triggered from the `set_seq_w(%zu)` function.",
        model->n_time_delay);
  }
}

void Model::Copy__Initializer__Weight_Parameter(
    Model const &ref_source_Neural_Network_received) {
  this->_initialized__weight =
      ref_source_Neural_Network_received._initialized__weight;

  this->_type_weights_initializer =
      ref_source_Neural_Network_received._type_weights_initializer;

  this->_LSUV_Parameters = ref_source_Neural_Network_received._LSUV_Parameters;
}

void Model::Copy__Regularization(Model const *const model) {
  // Regularization parameters.
  if (this->Set__Regularization__Max_Norm_Constraints(
          model->regularization__max_norm_constraints) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Regularization__Max_Norm_Constraints(%f)` function.",
        model->regularization__max_norm_constraints);
    return;
  } else if (this->set_l1(model->regularization__l1) == false) {
    ERR(L"An error has been triggered from the `set_l1(%f)` function.",
        model->regularization__l1);
    return;
  } else if (this->set_l2(model->regularization__l2) == false) {
    ERR(L"An error has been triggered from the `set_l2(%f)` function.",
        model->regularization__l2);
    return;
  } else if (this->set_srip(model->regularization__srip) == false) {
    ERR(L"An error has been triggered from the `set_srip(%f)` function.",
        model->regularization__srip);
    return;
  } else if (this->set_weight_decay(model->weight_decay) == false) {
    ERR(L"An error has been triggered from the "
        L"`set_weight_decay(%f)` function.",
        model->weight_decay);
    return;
  }
  // |END| Regularization parameters. |END|
}

void Model::Copy__Tied_Weight(Layer const *ptr_array_source_layers_received,
                              Layer const *const ptr_last_source_layer_received,
                              Layer *ptr_array_destination_layers_received) {
  for (; ptr_array_source_layers_received != ptr_last_source_layer_received;
       ++ptr_array_source_layers_received,
       ++ptr_array_destination_layers_received) {
    if (this->Set__Tied_Parameter(
            ptr_array_destination_layers_received,
            ptr_array_source_layers_received->use_tied_parameter) == false) {
      ERR(L"An error has been triggered from the "
          L"`Set__Tied_Parameter(ptr, %ls)` function.",
          to_wstring(ptr_array_source_layers_received->use_tied_parameter)
              .c_str());
    }
  }
}

void Model::Copy__Sparse_K_Filters(
    Layer const *ptr_array_source_layers_received,
    Layer const *const ptr_last_source_layer_received,
    Layer *ptr_array_destination_layers_received) {
  for (; ptr_array_source_layers_received != ptr_last_source_layer_received;
       ++ptr_array_source_layers_received,
       ++ptr_array_destination_layers_received) {
    if (this->Set__K_Sparsity(ptr_array_destination_layers_received,
                              ptr_array_source_layers_received->k_sparsity) ==
        false) {
      ERR(L"An error has been triggered from the "
          L"`Set__K_Sparsity(ptr, %zu)` function.",
          ptr_array_source_layers_received->k_sparsity);
      return;
    } else if (this->Set__Alpha_Sparsity(
                   ptr_array_destination_layers_received,
                   ptr_array_source_layers_received->alpha_sparsity) == false) {
      ERR(L"An error has been triggered from the "
          L"`Set__Alpha_Sparsity(%f)` function.",
          ptr_array_source_layers_received->alpha_sparsity);
      return;
    }
  }
}

void Model::Copy__Constraint_Recurrent_Weight(
    Layer const *ptr_array_source_layers_received,
    Layer const *const ptr_last_source_layer_received,
    Layer *ptr_array_destination_layers_received) {
  for (; ptr_array_source_layers_received != ptr_last_source_layer_received;
       ++ptr_array_source_layers_received,
       ++ptr_array_destination_layers_received) {
    if (this->Set__Regularization__Constraint_Recurrent_Weight(
            ptr_array_destination_layers_received,
            ptr_array_source_layers_received
                ->constraint_recurrent_weight_lower_bound,
            ptr_array_source_layers_received
                ->constraint_recurrent_weight_upper_bound) == false) {
      ERR(L"An error has been triggered from the "
          L"`Set__Regularization__Constraint_Recurrent_Weight(ptr, %f, %f)` "
          L"function.",
          ptr_array_source_layers_received
              ->constraint_recurrent_weight_lower_bound,
          ptr_array_source_layers_received
              ->constraint_recurrent_weight_upper_bound);
      return;
    }
  }
}

void Model::Copy__Loss(Model const *const model) {
  // Loss parameters.
  this->loss_train = model->loss_train;
  this->loss_valid = model->loss_valid;
  this->loss_testg = model->loss_testg;
  // |END| Loss parameters. |END|
}

void Model::Copy__Accuracy(Model const *const model) {
  // Accuracy parameters.
  this->acc_var = model->acc_var;
  this->acc_train = model->acc_train;
  this->acc_valid = model->acc_valid;
  this->acc_testg = model->acc_testg;
  // |END| Accuracy parameters. |END|
}

void Model::Copy__Dropout(Layer const *ptr_array_source_layers_received,
                          Layer const *const ptr_last_source_layer_received,
                          Layer *ptr_array_destination_layers_received) {
#ifdef COMPILE_CUDA
  bool tmp_parameters_has_change(false);
#endif

  for (; ptr_array_source_layers_received != ptr_last_source_layer_received;
       ++ptr_array_source_layers_received,
       ++ptr_array_destination_layers_received) {
#ifdef COMPILE_CUDA
    if (ptr_array_source_layers_received->dropout_values[0] !=
            ptr_array_destination_layers_received->dropout_values[0] ||
        ptr_array_source_layers_received->type_dropout !=
            ptr_array_destination_layers_received->type_dropout)
      tmp_parameters_has_change = true;
#endif

    if (this->set_dropout(ptr_array_destination_layers_received,
                          ptr_array_source_layers_received->type_dropout,
                          ptr_array_source_layers_received->dropout_values,
                          false) == false) {
      ERR(L"An error has been triggered from the "
          L"`Set__Layer_Normalization(ptr, %ls, %f, %f, %f, false)` "
          L"function.",
          LAYER_DROPOUT_NAME[ptr_array_source_layers_received->type_dropout]
              .c_str(),
          ptr_array_source_layers_received->dropout_values[0],
          ptr_array_source_layers_received->dropout_values[1],
          ptr_array_source_layers_received->dropout_values[2]);
      return;
    }
  }

#ifdef COMPILE_CUDA
  if (this->is_cu_initialized && tmp_parameters_has_change)
    this->cumodel->Copy__Dropout(this);
#endif
}

void Model::Copy__Normalization(
    Model const *const ptr_source_Neural_Network_received) {
  if (this->Set__Normalization_Momentum_Average(
          ptr_source_Neural_Network_received->normalization_momentum_average) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Normalization_Momentum_Average(%f)` function.",
        ptr_source_Neural_Network_received->normalization_momentum_average);
    return;
  } else if (this->Set__Normalization_Epsilon(
                 ptr_source_Neural_Network_received->normalization_epsilon) ==
             false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Normalization_Epsilon(%f)` function.",
        ptr_source_Neural_Network_received->normalization_epsilon);
    return;
  } else if (this->Set__Batch_Renormalization_r_Correction_Maximum(
                 ptr_source_Neural_Network_received
                     ->batch_renormalization_r_correction_maximum) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Batch_Renormalization_r_Correction_Maximum(%f)` function.",
        ptr_source_Neural_Network_received
            ->batch_renormalization_r_correction_maximum);
    return;
  } else if (this->Set__Batch_Renormalization_d_Correction_Maximum(
                 ptr_source_Neural_Network_received
                     ->batch_renormalization_d_correction_maximum) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Batch_Renormalization_d_Correction_Maximum(%f)` function.",
        ptr_source_Neural_Network_received
            ->batch_renormalization_d_correction_maximum);
    return;
  }
}

void Model::Copy__Normalization(
    Layer const *ptr_array_source_layers_received,
    Layer const *const ptr_last_source_layer_received,
    Layer *ptr_array_destination_layers_received) {
#ifdef COMPILE_CUDA
  bool tmp_parameters_has_change(false);
#endif

  // Hidden layer(s).
  for (; ptr_array_source_layers_received != ptr_last_source_layer_received;
       ++ptr_array_source_layers_received,
       ++ptr_array_destination_layers_received) {
#ifdef COMPILE_CUDA
    if (ptr_array_source_layers_received->type_normalization !=
        ptr_array_destination_layers_received->type_normalization)
      tmp_parameters_has_change = true;
#endif

    if (this->Set__Layer_Normalization(
            ptr_array_destination_layers_received,
            ptr_array_source_layers_received->type_normalization) == false) {
      ERR(L"An error has been triggered from the "
          L"`Set__Layer_Normalization(ptr, %ls)` function.",
          LAYER_NORM_NAME[ptr_array_source_layers_received->type_normalization]
              .c_str());
      return;
    }
  }
  // |END| Hidden layer(s). |END|

#ifdef COMPILE_CUDA
  if (this->is_cu_initialized &&
      (tmp_parameters_has_change || this->Use__Normalization()))
    this->cumodel->Copy__Normalization(this);
#endif
}

void Model::Copy__Block(BlockUnit const *const ptr_source_block_unit_received,
                        BlockUnit *const ptr_destination_block_unit_received) {
  CellUnit const *tmp_ptr_source_cell_unit_it,
      *tmp_ptr_destination_block_ptr_last_unit;
  CellUnit *tmp_ptr_destination_block_ptr_cell_unit_it;

  ptr_destination_block_unit_received->first_index_connection =
      ptr_source_block_unit_received->first_index_connection;
  ptr_destination_block_unit_received->last_index_connection =
      ptr_source_block_unit_received->last_index_connection;
  ptr_destination_block_unit_received
      ->first_index_feedforward_connection_input_gate =
      ptr_source_block_unit_received
          ->first_index_feedforward_connection_input_gate;
  ptr_destination_block_unit_received
      ->last_index_feedforward_connection_input_gate =
      ptr_source_block_unit_received
          ->last_index_feedforward_connection_input_gate;
  ptr_destination_block_unit_received
      ->first_index_feedforward_connection_forget_gate =
      ptr_source_block_unit_received
          ->first_index_feedforward_connection_forget_gate;
  ptr_destination_block_unit_received
      ->last_index_feedforward_connection_forget_gate =
      ptr_source_block_unit_received
          ->last_index_feedforward_connection_forget_gate;
  ptr_destination_block_unit_received
      ->first_index_feedforward_connection_output_gate =
      ptr_source_block_unit_received
          ->first_index_feedforward_connection_output_gate;
  ptr_destination_block_unit_received
      ->last_index_feedforward_connection_output_gate =
      ptr_source_block_unit_received
          ->last_index_feedforward_connection_output_gate;
  ptr_destination_block_unit_received
      ->first_index_recurrent_connection_input_gate =
      ptr_source_block_unit_received
          ->first_index_recurrent_connection_input_gate;
  ptr_destination_block_unit_received
      ->last_index_recurrent_connection_input_gate =
      ptr_source_block_unit_received
          ->last_index_recurrent_connection_input_gate;
  ptr_destination_block_unit_received
      ->first_index_recurrent_connection_forget_gate =
      ptr_source_block_unit_received
          ->first_index_recurrent_connection_forget_gate;
  ptr_destination_block_unit_received
      ->last_index_recurrent_connection_forget_gate =
      ptr_source_block_unit_received
          ->last_index_recurrent_connection_forget_gate;
  ptr_destination_block_unit_received
      ->first_index_recurrent_connection_output_gate =
      ptr_source_block_unit_received
          ->first_index_recurrent_connection_output_gate;
  ptr_destination_block_unit_received
      ->last_index_recurrent_connection_output_gate =
      ptr_source_block_unit_received
          ->last_index_recurrent_connection_output_gate;

#ifndef NO_PEEPHOLE
  ptr_destination_block_unit_received->first_index_peephole_input_gate =
      ptr_source_block_unit_received->first_index_peephole_input_gate;
  ptr_destination_block_unit_received->last_index_peephole_input_gate =
      ptr_source_block_unit_received->last_index_peephole_input_gate;
  ptr_destination_block_unit_received->first_index_peephole_forget_gate =
      ptr_source_block_unit_received->first_index_peephole_forget_gate;
  ptr_destination_block_unit_received->last_index_peephole_forget_gate =
      ptr_source_block_unit_received->last_index_peephole_forget_gate;
  ptr_destination_block_unit_received->first_index_peephole_output_gate =
      ptr_source_block_unit_received->first_index_peephole_output_gate;
  ptr_destination_block_unit_received->last_index_peephole_output_gate =
      ptr_source_block_unit_received->last_index_peephole_output_gate;
#endif

  this->Copy__Block__AF(ptr_source_block_unit_received,
                        ptr_destination_block_unit_received);

  for (tmp_ptr_source_cell_unit_it =
           ptr_source_block_unit_received->ptr_array_cell_units,
      tmp_ptr_destination_block_ptr_last_unit =
           ptr_destination_block_unit_received->ptr_last_cell_unit,
      tmp_ptr_destination_block_ptr_cell_unit_it =
           ptr_destination_block_unit_received->ptr_array_cell_units;
       tmp_ptr_destination_block_ptr_cell_unit_it !=
       tmp_ptr_destination_block_ptr_last_unit;
       ++tmp_ptr_destination_block_ptr_cell_unit_it,
      ++tmp_ptr_source_cell_unit_it) {
    tmp_ptr_destination_block_ptr_cell_unit_it
        ->first_index_feedforward_connection_cell_input =
        tmp_ptr_source_cell_unit_it
            ->first_index_feedforward_connection_cell_input;
    tmp_ptr_destination_block_ptr_cell_unit_it
        ->last_index_feedforward_connection_cell_input =
        tmp_ptr_source_cell_unit_it
            ->last_index_feedforward_connection_cell_input;
    tmp_ptr_destination_block_ptr_cell_unit_it
        ->first_index_recurrent_connection_cell_input =
        tmp_ptr_source_cell_unit_it
            ->first_index_recurrent_connection_cell_input;
    tmp_ptr_destination_block_ptr_cell_unit_it
        ->last_index_recurrent_connection_cell_input =
        tmp_ptr_source_cell_unit_it->last_index_recurrent_connection_cell_input;

#ifndef NO_PEEPHOLE
    tmp_ptr_destination_block_ptr_cell_unit_it->index_peephole_input_gate =
        tmp_ptr_source_cell_unit_it->index_peephole_input_gate;
    tmp_ptr_destination_block_ptr_cell_unit_it->index_peephole_forget_gate =
        tmp_ptr_source_cell_unit_it->index_peephole_forget_gate;
    tmp_ptr_destination_block_ptr_cell_unit_it->index_peephole_output_gate =
        tmp_ptr_source_cell_unit_it->index_peephole_output_gate;
#endif
  }
}

void Model::Copy__Block__AF(
    BlockUnit const *const ptr_source_block_unit_received,
    BlockUnit *const ptr_destination_block_unit_received) {
  ptr_destination_block_unit_received->activation_function_gate =
      ptr_source_block_unit_received->activation_function_gate;
  ptr_destination_block_unit_received->activation_function_io =
      ptr_source_block_unit_received->activation_function_io;
}

void Model::Copy__Blocks(size_t const start_index_received,
                         size_t const end_index_received,
                         BlockUnit const *ptr_array_source_block_units_received,
                         bool const copy_connections_received) {
  if (start_index_received + end_index_received == 0_UZ) {
    return;
  } else if (start_index_received > end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        start_index_received, end_index_received);
    return;
  }

  BlockUnit *tmp_ptr_destination_block_it(this->ptr_array_block_units +
                                          start_index_received);

  if (this->use_mp && this->is_mp_initialized) {
    int const tmp_total_units__int(
        static_cast<int>(end_index_received - start_index_received));
    int tmp_unit_index__int;

    if (copy_connections_received) {
#pragma omp parallel for schedule(static)
      for (tmp_unit_index__int = static_cast<int>(start_index_received);
           tmp_unit_index__int < tmp_total_units__int; ++tmp_unit_index__int) {
        this->Copy__Block(
            ptr_array_source_block_units_received + tmp_unit_index__int,
            tmp_ptr_destination_block_it + tmp_unit_index__int);
      }
    } else {
#pragma omp parallel for schedule(static)
      for (tmp_unit_index__int = static_cast<int>(start_index_received);
           tmp_unit_index__int < tmp_total_units__int; ++tmp_unit_index__int) {
        this->Copy__Block__AF(
            ptr_array_source_block_units_received + tmp_unit_index__int,
            tmp_ptr_destination_block_it + tmp_unit_index__int);
      }
    }
  } else {
    BlockUnit const *const tmp_ptr_destination_last_block_unit(
        tmp_ptr_destination_block_it +
        (end_index_received - start_index_received));

    if (copy_connections_received) {
      for (;
           tmp_ptr_destination_block_it != tmp_ptr_destination_last_block_unit;
           ++tmp_ptr_destination_block_it,
           ++ptr_array_source_block_units_received) {
        this->Copy__Block(ptr_array_source_block_units_received,
                          tmp_ptr_destination_block_it);
      }
    } else {
      for (;
           tmp_ptr_destination_block_it != tmp_ptr_destination_last_block_unit;
           ++tmp_ptr_destination_block_it,
           ++ptr_array_source_block_units_received) {
        this->Copy__Block__AF(ptr_array_source_block_units_received,
                              tmp_ptr_destination_block_it);
      }
    }
  }
}

void Model::Copy__Blocks__AF(
    size_t const start_index_received, size_t const end_index_received,
    BlockUnit const *ptr_array_source_block_units_received) {
  if (start_index_received + end_index_received == 0_UZ) {
    return;
  } else if (start_index_received > end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        start_index_received, end_index_received);
    return;
  }

  BlockUnit *tmp_ptr_destination_block_it(this->ptr_array_block_units +
                                          start_index_received);

  if (this->use_mp && this->is_mp_initialized) {
    int const tmp_total_units__int(
        static_cast<int>(end_index_received - start_index_received));
    int tmp_unit_index__int;

#pragma omp parallel for schedule(static)
    for (tmp_unit_index__int = static_cast<int>(start_index_received);
         tmp_unit_index__int < tmp_total_units__int; ++tmp_unit_index__int) {
      this->Copy__Block__AF(
          ptr_array_source_block_units_received + tmp_unit_index__int,
          tmp_ptr_destination_block_it + tmp_unit_index__int);
    }
  } else {
    BlockUnit const *const tmp_ptr_destination_last_block_unit(
        tmp_ptr_destination_block_it +
        (end_index_received - start_index_received));

    for (; tmp_ptr_destination_block_it != tmp_ptr_destination_last_block_unit;
         ++tmp_ptr_destination_block_it,
         ++ptr_array_source_block_units_received) {
      this->Copy__Block__AF(ptr_array_source_block_units_received,
                            tmp_ptr_destination_block_it);
    }
  }
}

void Model::Copy__Neuron_Units(
    size_t const start_index_received, size_t const end_index_received,
    Neuron_unit const *ptr_array_source_neuron_units_received) {
  if (start_index_received + end_index_received == 0_UZ) {
    return;
  } else if (start_index_received > end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        start_index_received, end_index_received);
    return;
  }

  Neuron_unit *tmp_ptr_destination_neuron_it(this->ptr_array_neuron_units +
                                             start_index_received);

  if (this->use_mp && this->is_mp_initialized) {
    int const tmp_total_units__int(
        static_cast<int>(end_index_received - start_index_received));
    int tmp_unit_index__int;

#pragma omp parallel for schedule(static)
    for (tmp_unit_index__int = 0; tmp_unit_index__int < tmp_total_units__int;
         ++tmp_unit_index__int) {
      this->Copy__Neuron_Unit(
          ptr_array_source_neuron_units_received + tmp_unit_index__int,
          tmp_ptr_destination_neuron_it + tmp_unit_index__int);
    }
  } else {
    Neuron_unit const *const tmp_ptr_destination_last_neuron(
        tmp_ptr_destination_neuron_it +
        (end_index_received - start_index_received));

    for (; tmp_ptr_destination_neuron_it != tmp_ptr_destination_last_neuron;
         ++tmp_ptr_destination_neuron_it,
         ++ptr_array_source_neuron_units_received) {
      this->Copy__Neuron_Unit(ptr_array_source_neuron_units_received,
                              tmp_ptr_destination_neuron_it);
    }
  }
}

void Model::Copy__Neuron_Unit(
    Neuron_unit const *const ptr_source_neuron_unit_received,
    Neuron_unit *const ptr_destination_neuron_unit_received) {
  *ptr_destination_neuron_unit_received->ptr_first_connection_index =
      *ptr_source_neuron_unit_received->ptr_first_connection_index;
  *ptr_destination_neuron_unit_received->ptr_last_connection_index =
      *ptr_source_neuron_unit_received->ptr_last_connection_index;
  *ptr_destination_neuron_unit_received->ptr_number_connections =
      *ptr_source_neuron_unit_received->ptr_number_connections;
}

void Model::Copy__AF_Units(size_t const start_index_received,
                           size_t const end_index_received,
                           AF_unit const *ptr_array_source_AF_units_received) {
  if (start_index_received + end_index_received == 0_UZ) {
    return;
  } else if (start_index_received > end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        start_index_received, end_index_received);
    return;
  }

  AF_unit *tmp_ptr_destination_AF_it(this->ptr_array_AF_units +
                                     start_index_received);

  if (this->use_mp && this->is_mp_initialized) {
    int const tmp_total_units__int(
        static_cast<int>(end_index_received - start_index_received));
    int tmp_unit_index__int;

#pragma omp parallel for schedule(static)
    for (tmp_unit_index__int = 0; tmp_unit_index__int < tmp_total_units__int;
         ++tmp_unit_index__int) {
      this->Copy__AF_Unit(
          ptr_array_source_AF_units_received + tmp_unit_index__int,
          tmp_ptr_destination_AF_it + tmp_unit_index__int);
    }
  } else {
    AF_unit const *const tmp_ptr_destination_last_af(
        tmp_ptr_destination_AF_it +
        (end_index_received - start_index_received));

    for (; tmp_ptr_destination_AF_it != tmp_ptr_destination_last_af;
         ++tmp_ptr_destination_AF_it, ++ptr_array_source_AF_units_received) {
      this->Copy__AF_Unit(ptr_array_source_AF_units_received,
                          tmp_ptr_destination_AF_it);
    }
  }
}

void Model::Copy__AF_Unit(AF_unit const *const ptr_source_AF_unit_received,
                          AF_unit *const ptr_destination_AF_unit_received) {
  *ptr_destination_AF_unit_received->ptr_type_activation_function =
      *ptr_source_AF_unit_received->ptr_type_activation_function;
}

void Model::Copy__AF_Ind_Recurrent_Units(
    size_t const start_index_received, size_t const end_index_received,
    AF_Ind_recurrent_unit const
        *ptr_array_source_AF_Ind_recurrent_units_received,
    bool const copy_connections_received) {
  if (start_index_received + end_index_received == 0_UZ) {
    return;
  } else if (start_index_received > end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        start_index_received, end_index_received);
    return;
  }

  AF_Ind_recurrent_unit *tmp_ptr_destination_AF_Ind_it(
      this->ptr_array_AF_Ind_recurrent_units + start_index_received);

  if (this->use_mp && this->is_mp_initialized) {
    int const tmp_total_units__int(
        static_cast<int>(end_index_received - start_index_received));
    int tmp_unit_index__int;

#pragma omp parallel for schedule(static)
    for (tmp_unit_index__int = 0; tmp_unit_index__int < tmp_total_units__int;
         ++tmp_unit_index__int) {
      this->Copy__AF_Ind_Recurrent_Unit(
          ptr_array_source_AF_Ind_recurrent_units_received +
              tmp_unit_index__int,
          tmp_ptr_destination_AF_Ind_it + tmp_unit_index__int,
          copy_connections_received);
    }
  } else {
    AF_Ind_recurrent_unit const *const tmp_ptr_destination_last_AF_ind(
        tmp_ptr_destination_AF_Ind_it +
        (end_index_received - start_index_received));

    for (; tmp_ptr_destination_AF_Ind_it != tmp_ptr_destination_last_AF_ind;
         ++tmp_ptr_destination_AF_Ind_it,
         ++ptr_array_source_AF_Ind_recurrent_units_received) {
      this->Copy__AF_Ind_Recurrent_Unit(
          ptr_array_source_AF_Ind_recurrent_units_received,
          tmp_ptr_destination_AF_Ind_it, copy_connections_received);
    }
  }
}

void Model::Copy__AF_Ind_Recurrent_Unit(
    AF_Ind_recurrent_unit const
        *const ptr_source_AF_Ind_recurrent_unit_received,
    AF_Ind_recurrent_unit *const ptr_destination_AF_Ind_recurrent_unit_received,
    bool const copy_connections_received) {
  if (copy_connections_received) {
    *ptr_destination_AF_Ind_recurrent_unit_received
         ->ptr_recurrent_connection_index =
        *ptr_source_AF_Ind_recurrent_unit_received
             ->ptr_recurrent_connection_index;
  }

  *ptr_destination_AF_Ind_recurrent_unit_received
       ->ptr_type_activation_function =
      *ptr_source_AF_Ind_recurrent_unit_received->ptr_type_activation_function;
}

void Model::Copy__Normalized_Units(
    size_t const start_index_received, size_t const end_index_received,
    union Normalized_unit const *ptr_array_source_normalized_units_received) {
  if (start_index_received + end_index_received == 0_UZ) {
    return;
  } else if (start_index_received > end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        start_index_received, end_index_received);
    return;
  }

  if (this->Use__Normalization()) {
    size_t tmp_number_units[2] = {0};

    Layer const *const last_layer(this->ptr_last_layer);
    Layer *layer_it(this->ptr_array_layers);

    BlockUnit const *tmp_ptr_last_block_unit;
    BlockUnit *tmp_ptr_block_unit_it;

    CellUnit const *tmp_ptr_last_cell_unit;
    CellUnit *tmp_ptr_cell_unit_it;

    union Normalized_unit const *tmp_ptr_destination_last_normalized_unit;
    union Normalized_unit *tmp_ptr_destination_normalized_unit_it(nullptr);

    for (; layer_it != last_layer; ++layer_it) {
      if (static_cast<size_t>(layer_it->ptr_array_normalized_units -
                              this->ptr_array_normalized_units) <
          start_index_received) {
        continue;
      } else if (static_cast<size_t>(layer_it->ptr_array_normalized_units -
                                     this->ptr_array_normalized_units) >=
                 end_index_received) {
        break;
      }

      if ((tmp_number_units[0] = static_cast<size_t>(
               layer_it->ptr_last_normalized_unit -
               layer_it->ptr_array_normalized_units)) != 0_UZ) {
        switch (layer_it->type_layer) {
          case LAYER::FULLY_CONNECTED:
          case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          case LAYER::FULLY_CONNECTED_RECURRENT:
          case LAYER::RESIDUAL:
            switch (layer_it->type_normalization) {
              case LAYER_NORM::BATCH_NORMALIZATION:
              case LAYER_NORM::BATCH_RENORMALIZATION:
              case LAYER_NORM::GHOST_BATCH_NORMALIZATION:
                for (tmp_ptr_destination_last_normalized_unit =
                         this->ptr_array_normalized_units +
                         std::min<size_t>(
                             static_cast<size_t>(
                                 layer_it->ptr_last_normalized_unit -
                                 this->ptr_array_normalized_units),
                             end_index_received),
                    tmp_ptr_destination_normalized_unit_it =
                         layer_it->ptr_array_normalized_units;
                     tmp_ptr_destination_normalized_unit_it <
                     tmp_ptr_destination_last_normalized_unit;
                     ++tmp_ptr_destination_normalized_unit_it,
                    ++ptr_array_source_normalized_units_received) {
                  this->Copy__Normalized_Batch_Unit(
                      tmp_number_units[0],
                      ptr_array_source_normalized_units_received
                          ->normalized_batch_units,
                      tmp_ptr_destination_normalized_unit_it
                          ->normalized_batch_units);
                }
                break;
              default:
                ptr_array_source_normalized_units_received +=
                    tmp_number_units[0];
                break;
            }
            break;
          case LAYER::LSTM:
            switch (layer_it->type_normalization) {
              case LAYER_NORM::BATCH_NORMALIZATION:
              case LAYER_NORM::BATCH_RENORMALIZATION:
              case LAYER_NORM::GHOST_BATCH_NORMALIZATION:
                // Number block unit(s) in layer.
                tmp_number_units[0] =
                    static_cast<size_t>(layer_it->ptr_last_block_unit -
                                        layer_it->ptr_array_block_units);

                // Number cell unit(s) in layer.
                tmp_number_units[1] =
                    static_cast<size_t>(layer_it->ptr_last_cell_unit -
                                        layer_it->ptr_array_cell_units);

                // Loop through each block unit(s) in the layer.
                for (tmp_ptr_last_block_unit = layer_it->ptr_last_block_unit,
                    tmp_ptr_block_unit_it = layer_it->ptr_array_block_units;
                     tmp_ptr_block_unit_it != tmp_ptr_last_block_unit;
                     ++tmp_ptr_block_unit_it) {
                  // Loop through each cell unit(s) in the block.
                  for (tmp_ptr_last_cell_unit =
                           tmp_ptr_block_unit_it->ptr_last_cell_unit,
                      tmp_ptr_cell_unit_it =
                           tmp_ptr_block_unit_it->ptr_array_cell_units;
                       tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit;
                       ++tmp_ptr_cell_unit_it) {
                    // Loop through each normalized unit(s) in the cell.
                    for (tmp_ptr_destination_last_normalized_unit =
                             this->ptr_array_normalized_units +
                             std::min<size_t>(
                                 static_cast<size_t>(
                                     tmp_ptr_cell_unit_it
                                         ->ptr_last_normalized_unit -
                                     this->ptr_array_normalized_units),
                                 end_index_received),
                        tmp_ptr_destination_normalized_unit_it =
                             tmp_ptr_cell_unit_it->ptr_array_normalized_units;
                         tmp_ptr_destination_normalized_unit_it <
                         tmp_ptr_destination_last_normalized_unit;
                         ++tmp_ptr_destination_normalized_unit_it,
                        ++ptr_array_source_normalized_units_received) {
                      this->Copy__Normalized_Batch_Unit(
                          tmp_number_units[1],
                          ptr_array_source_normalized_units_received
                              ->normalized_batch_units,
                          tmp_ptr_destination_normalized_unit_it
                              ->normalized_batch_units);
                    }
                  }

                  // Loop through each normalized unit(s) in the block.
                  for (tmp_ptr_destination_last_normalized_unit =
                           this->ptr_array_normalized_units +
                           std::min<size_t>(
                               static_cast<size_t>(
                                   tmp_ptr_block_unit_it
                                       ->ptr_last_normalized_unit -
                                   this->ptr_array_normalized_units),
                               end_index_received),
                      tmp_ptr_destination_normalized_unit_it =
                           tmp_ptr_block_unit_it->ptr_array_normalized_units;
                       tmp_ptr_destination_normalized_unit_it <
                       tmp_ptr_destination_last_normalized_unit;
                       ++tmp_ptr_destination_normalized_unit_it,
                      ++ptr_array_source_normalized_units_received) {
                    this->Copy__Normalized_Batch_Unit(
                        tmp_number_units[0],
                        ptr_array_source_normalized_units_received
                            ->normalized_batch_units,
                        tmp_ptr_destination_normalized_unit_it
                            ->normalized_batch_units);
                  }
                }
                break;
              default:
                ptr_array_source_normalized_units_received +=
                    6_UZ *
                        static_cast<size_t>(layer_it->ptr_last_block_unit -
                                            layer_it->ptr_array_block_units) +
                    3_UZ * static_cast<size_t>(layer_it->ptr_last_cell_unit -
                                               layer_it->ptr_array_cell_units);
                break;
            }
            break;
          default:
            ERR(L"Layer type (%d | %ls) is not managed in",
                layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
            break;
        }
      }
    }
  }
}

void Model::Copy__Normalized_Batch_Unit(
    size_t const number_units_received,
    Normalized_batch_unit const &ref_source_normalized_batch_unit_received,
    Normalized_batch_unit &ref_destination_normalized_batch_unit_received) {
  size_t tmp_time_step_index, tmp_unit_timed_index;

  *ref_destination_normalized_batch_unit_received.ptr_scale =
      *ref_source_normalized_batch_unit_received.ptr_scale;
  *ref_destination_normalized_batch_unit_received.ptr_shift =
      *ref_source_normalized_batch_unit_received.ptr_shift;

  for (tmp_time_step_index = 0_UZ; tmp_time_step_index != this->seq_w;
       ++tmp_time_step_index) {
    tmp_unit_timed_index = number_units_received * tmp_time_step_index;

    ref_destination_normalized_batch_unit_received
        .ptr_mean_average[tmp_unit_timed_index] =
        ref_source_normalized_batch_unit_received
            .ptr_mean_average[tmp_unit_timed_index];
    ref_destination_normalized_batch_unit_received
        .ptr_variance_average[tmp_unit_timed_index] =
        ref_source_normalized_batch_unit_received
            .ptr_variance_average[tmp_unit_timed_index];
  }
}

template <class U>
void Model::Copy__Layer__FC(
    Layer const *const ptr_source_layer_received,
    Layer *const ptr_destination_layer_received,
    U *const ptr_source_first_U_received,
    U *const ptr_destination_first_U_received,
    U *const *ptr_source_array_ptr_connections_received,
    U **ptr_destination_array_ptr_connections_received) {
  Neuron_unit const *const tmp_ptr_source_last_neuron_unit(
      ptr_source_layer_received->ptr_last_neuron_unit),
      *tmp_ptr_source_neuron_unit_it(
          ptr_source_layer_received->ptr_array_neuron_units);
  Neuron_unit *tmp_ptr_destination_neuron_unit_it(
      ptr_destination_layer_received->ptr_array_neuron_units);

  size_t const tmp_number_forward_connections(
      *tmp_ptr_source_neuron_unit_it->ptr_number_connections);
  size_t tmp_connection_index;

  U *const *tmp_ptr_source_array_ptr_connection_U,
      **tmp_ptr_destination_array_ptr_connection_U;

  for (; tmp_ptr_source_neuron_unit_it != tmp_ptr_source_last_neuron_unit;
       ++tmp_ptr_source_neuron_unit_it, ++tmp_ptr_destination_neuron_unit_it) {
    tmp_ptr_source_array_ptr_connection_U =
        ptr_source_array_ptr_connections_received +
        *tmp_ptr_source_neuron_unit_it->ptr_first_connection_index;
    tmp_ptr_destination_array_ptr_connection_U =
        ptr_destination_array_ptr_connections_received +
        *tmp_ptr_source_neuron_unit_it->ptr_first_connection_index;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_forward_connections;
         ++tmp_connection_index) {
      tmp_ptr_destination_array_ptr_connection_U[tmp_connection_index] =
          ptr_destination_first_U_received +
          static_cast<size_t>(
              tmp_ptr_source_array_ptr_connection_U[tmp_connection_index] -
              ptr_source_first_U_received);
    }
  }
}

void Model::Copy__Layer__AF_Ind_Recurrent(
    Layer const *const ptr_source_layer_received,
    AF_Ind_recurrent_unit
        *const ptr_source_first_AF_Ind_recurrent_unit_received,
    AF_Ind_recurrent_unit
        *const ptr_destination_first_AF_Ind_recurrent_unit_received,
    AF_Ind_recurrent_unit *const *ptr_source_array_ptr_connections_received,
    AF_Ind_recurrent_unit **ptr_destination_array_ptr_connections_received) {
  AF_Ind_recurrent_unit const *const tmp_ptr_source_last_AF_Ind_recurrent_unit(
      ptr_source_layer_received->ptr_last_AF_Ind_recurrent_unit),
      *tmp_ptr_source_AF_Ind_recurrent_unit_it(
          ptr_source_layer_received->ptr_array_AF_Ind_recurrent_units);

  for (; tmp_ptr_source_AF_Ind_recurrent_unit_it !=
         tmp_ptr_source_last_AF_Ind_recurrent_unit;
       ++tmp_ptr_source_AF_Ind_recurrent_unit_it) {
    ptr_destination_array_ptr_connections_received
        [*tmp_ptr_source_AF_Ind_recurrent_unit_it
              ->ptr_recurrent_connection_index] =
            ptr_destination_first_AF_Ind_recurrent_unit_received +
            static_cast<size_t>(
                ptr_source_array_ptr_connections_received
                    [*tmp_ptr_source_AF_Ind_recurrent_unit_it
                          ->ptr_recurrent_connection_index] -
                ptr_source_first_AF_Ind_recurrent_unit_received);
  }
}

template <class U>
void Model::Copy__Layer__LSTM(
    Layer const *const ptr_source_layer_received,
    Layer *const ptr_destination_layer_received,
    CellUnit *const ptr_source_first_cell_unit_received,
    U *const ptr_source_first_U_received,
    U *const ptr_destination_first_U_received,
    void *const *ptr_source_array_ptr_connections_received,
    void **ptr_destination_array_ptr_connections_received) {
  BlockUnit const *const tmp_ptr_source_last_block_unit(
      ptr_source_layer_received->ptr_last_block_unit),
      *tmp_ptr_source_block_unit_it(
          ptr_source_layer_received->ptr_array_block_units);
  BlockUnit *tmp_ptr_destination_block_unit_it(
      ptr_destination_layer_received->ptr_array_block_units);

  size_t const tmp_number_inputs_connections(
      tmp_ptr_source_block_unit_it
          ->last_index_feedforward_connection_input_gate -
      tmp_ptr_source_block_unit_it
          ->first_index_feedforward_connection_input_gate),
      tmp_number_recurrents_connection(
          tmp_ptr_source_block_unit_it
              ->last_index_recurrent_connection_input_gate -
          tmp_ptr_source_block_unit_it
              ->first_index_recurrent_connection_input_gate);
  size_t tmp_connection_index;

  U *const *tmp_ptr_source_cell_input_array_ptr_connection_U,
      *const *tmp_ptr_source_input_gate_array_ptr_connection_U,
          *const *tmp_ptr_source_forget_gate_array_ptr_connection_U,
              *const *tmp_ptr_source_output_gate_array_ptr_connection_U,
      **tmp_ptr_destination_cell_input_array_ptr_connection_U,
      **tmp_ptr_destination_input_gate_array_ptr_connection_U,
      **tmp_ptr_destination_forget_gate_array_ptr_connection_U,
      **tmp_ptr_destination_output_gate_array_ptr_connection_U;

  CellUnit const *tmp_ptr_source_block_ptr_last_unit,
      *tmp_ptr_source_block_ptr_cell_unit_it;
  CellUnit *const *tmp_ptr_source_array_ptr_connection_cell_units,
      *const *tmp_ptr_source_cell_input_array_ptr_connection_cell_units,
          *const *tmp_ptr_source_input_gate_array_ptr_connection_cell_units,
              *const *tmp_ptr_source_forget_gate_array_ptr_connection_cell_units,
                  *const *
                      tmp_ptr_source_output_gate_array_ptr_connection_cell_units,
      **tmp_ptr_destination_array_ptr_connection_cell_units,
      **tmp_ptr_destination_cell_input_array_ptr_connection_cell_units,
      **tmp_ptr_destination_input_gate_array_ptr_connection_cell_units,
      **tmp_ptr_destination_forget_gate_array_ptr_connection_cell_units,
      **tmp_ptr_destination_output_gate_array_ptr_connection_cell_units,
      *tmp_ptr_destination_block_ptr_cell_unit_it;

  for (; tmp_ptr_source_block_unit_it != tmp_ptr_source_last_block_unit;
       ++tmp_ptr_source_block_unit_it, ++tmp_ptr_destination_block_unit_it) {
    this->Copy__Block(tmp_ptr_source_block_unit_it,
                      tmp_ptr_destination_block_unit_it);

    // [0] Cell input.
    tmp_ptr_source_block_ptr_last_unit =
        tmp_ptr_source_block_unit_it->ptr_last_cell_unit;

    for (tmp_ptr_destination_block_ptr_cell_unit_it =
             tmp_ptr_destination_block_unit_it->ptr_array_cell_units,
        tmp_ptr_source_block_ptr_cell_unit_it =
             tmp_ptr_source_block_unit_it->ptr_array_cell_units;
         tmp_ptr_source_block_ptr_cell_unit_it !=
         tmp_ptr_source_block_ptr_last_unit;
         ++tmp_ptr_source_block_ptr_cell_unit_it,
        ++tmp_ptr_destination_block_ptr_cell_unit_it) {
      //    [1] Input, cell input.
      tmp_ptr_source_cell_input_array_ptr_connection_U =
          reinterpret_cast<U *const *>(
              ptr_source_array_ptr_connections_received +
              tmp_ptr_source_block_ptr_cell_unit_it
                  ->first_index_feedforward_connection_cell_input);

      tmp_ptr_destination_cell_input_array_ptr_connection_U =
          reinterpret_cast<U **>(
              ptr_destination_array_ptr_connections_received +
              tmp_ptr_destination_block_ptr_cell_unit_it
                  ->first_index_feedforward_connection_cell_input);

      for (tmp_connection_index = 0_UZ;
           tmp_connection_index != tmp_number_inputs_connections;
           ++tmp_connection_index) {
        tmp_ptr_destination_cell_input_array_ptr_connection_U
            [tmp_connection_index] =
                ptr_destination_first_U_received +
                static_cast<size_t>(
                    tmp_ptr_source_cell_input_array_ptr_connection_U
                        [tmp_connection_index] -
                    ptr_source_first_U_received);
      }
      //    [1] |END| Input, cell input. |END|

      //    [1] Recurrent, cell input.
      tmp_ptr_source_cell_input_array_ptr_connection_cell_units =
          reinterpret_cast<CellUnit *const *>(
              ptr_source_array_ptr_connections_received +
              tmp_ptr_source_block_ptr_cell_unit_it
                  ->first_index_recurrent_connection_cell_input);

      tmp_ptr_destination_cell_input_array_ptr_connection_cell_units =
          reinterpret_cast<CellUnit **>(
              ptr_destination_array_ptr_connections_received +
              tmp_ptr_destination_block_ptr_cell_unit_it
                  ->first_index_recurrent_connection_cell_input);

      for (tmp_connection_index = 0_UZ;
           tmp_connection_index != tmp_number_recurrents_connection;
           ++tmp_connection_index) {
        tmp_ptr_destination_cell_input_array_ptr_connection_cell_units
            [tmp_connection_index] =
                this->ptr_array_cell_units +
                static_cast<size_t>(
                    tmp_ptr_source_cell_input_array_ptr_connection_cell_units
                        [tmp_connection_index] -
                    ptr_source_first_cell_unit_received);
      }
      //    [1] |END| Recurrent, cell input. |END|
    }
    // [0] |END| Cell input. |END|

    // [0] Input, gates.
    tmp_ptr_source_input_gate_array_ptr_connection_U =
        reinterpret_cast<U *const *>(
            ptr_source_array_ptr_connections_received +
            tmp_ptr_source_block_unit_it
                ->first_index_feedforward_connection_input_gate);
    tmp_ptr_source_forget_gate_array_ptr_connection_U =
        reinterpret_cast<U *const *>(
            ptr_source_array_ptr_connections_received +
            tmp_ptr_source_block_unit_it
                ->first_index_feedforward_connection_forget_gate);
    tmp_ptr_source_output_gate_array_ptr_connection_U =
        reinterpret_cast<U *const *>(
            ptr_source_array_ptr_connections_received +
            tmp_ptr_source_block_unit_it
                ->first_index_feedforward_connection_output_gate);

    tmp_ptr_destination_input_gate_array_ptr_connection_U =
        reinterpret_cast<U **>(
            ptr_destination_array_ptr_connections_received +
            tmp_ptr_destination_block_unit_it
                ->first_index_feedforward_connection_input_gate);
    tmp_ptr_destination_forget_gate_array_ptr_connection_U =
        reinterpret_cast<U **>(
            ptr_destination_array_ptr_connections_received +
            tmp_ptr_destination_block_unit_it
                ->first_index_feedforward_connection_forget_gate);
    tmp_ptr_destination_output_gate_array_ptr_connection_U =
        reinterpret_cast<U **>(
            ptr_destination_array_ptr_connections_received +
            tmp_ptr_destination_block_unit_it
                ->first_index_feedforward_connection_output_gate);

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_inputs_connections;
         ++tmp_connection_index) {
      tmp_ptr_destination_input_gate_array_ptr_connection_U
          [tmp_connection_index] =
              ptr_destination_first_U_received +
              static_cast<size_t>(
                  tmp_ptr_source_input_gate_array_ptr_connection_U
                      [tmp_connection_index] -
                  ptr_source_first_U_received);
      tmp_ptr_destination_forget_gate_array_ptr_connection_U
          [tmp_connection_index] =
              ptr_destination_first_U_received +
              static_cast<size_t>(
                  tmp_ptr_source_forget_gate_array_ptr_connection_U
                      [tmp_connection_index] -
                  ptr_source_first_U_received);
      tmp_ptr_destination_output_gate_array_ptr_connection_U
          [tmp_connection_index] =
              ptr_destination_first_U_received +
              static_cast<size_t>(
                  tmp_ptr_source_output_gate_array_ptr_connection_U
                      [tmp_connection_index] -
                  ptr_source_first_U_received);
    }
    // [0] |END| Input, gates. |END|

    // [0] Recurrent, gates.
    tmp_ptr_source_input_gate_array_ptr_connection_cell_units =
        reinterpret_cast<CellUnit *const *>(
            ptr_source_array_ptr_connections_received +
            tmp_ptr_source_block_unit_it
                ->first_index_recurrent_connection_input_gate);
    tmp_ptr_source_forget_gate_array_ptr_connection_cell_units =
        reinterpret_cast<CellUnit *const *>(
            ptr_source_array_ptr_connections_received +
            tmp_ptr_source_block_unit_it
                ->first_index_recurrent_connection_forget_gate);
    tmp_ptr_source_output_gate_array_ptr_connection_cell_units =
        reinterpret_cast<CellUnit *const *>(
            ptr_source_array_ptr_connections_received +
            tmp_ptr_source_block_unit_it
                ->first_index_recurrent_connection_output_gate);

    tmp_ptr_destination_input_gate_array_ptr_connection_cell_units =
        reinterpret_cast<CellUnit **>(
            ptr_destination_array_ptr_connections_received +
            tmp_ptr_destination_block_unit_it
                ->first_index_recurrent_connection_input_gate);
    tmp_ptr_destination_forget_gate_array_ptr_connection_cell_units =
        reinterpret_cast<CellUnit **>(
            ptr_destination_array_ptr_connections_received +
            tmp_ptr_destination_block_unit_it
                ->first_index_recurrent_connection_forget_gate);
    tmp_ptr_destination_output_gate_array_ptr_connection_cell_units =
        reinterpret_cast<CellUnit **>(
            ptr_destination_array_ptr_connections_received +
            tmp_ptr_destination_block_unit_it
                ->first_index_recurrent_connection_output_gate);

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_recurrents_connection;
         ++tmp_connection_index) {
      tmp_ptr_destination_input_gate_array_ptr_connection_cell_units
          [tmp_connection_index] =
              this->ptr_array_cell_units +
              static_cast<size_t>(
                  tmp_ptr_source_input_gate_array_ptr_connection_cell_units
                      [tmp_connection_index] -
                  ptr_source_first_cell_unit_received);
      tmp_ptr_destination_forget_gate_array_ptr_connection_cell_units
          [tmp_connection_index] =
              this->ptr_array_cell_units +
              static_cast<size_t>(
                  tmp_ptr_source_forget_gate_array_ptr_connection_cell_units
                      [tmp_connection_index] -
                  ptr_source_first_cell_unit_received);
      tmp_ptr_destination_output_gate_array_ptr_connection_cell_units
          [tmp_connection_index] =
              this->ptr_array_cell_units +
              static_cast<size_t>(
                  tmp_ptr_source_output_gate_array_ptr_connection_cell_units
                      [tmp_connection_index] -
                  ptr_source_first_cell_unit_received);
    }
    // [0] |END| Recurrent, gates. |END|

#ifndef NO_PEEPHOLE
    //    [1] Peepholes.
    tmp_ptr_source_array_ptr_connection_cell_units =
        reinterpret_cast<CellUnit *const *>(
            ptr_source_array_ptr_connections_received);

    tmp_ptr_destination_array_ptr_connection_cell_units =
        reinterpret_cast<CellUnit **>(
            ptr_destination_array_ptr_connections_received);

    for (tmp_ptr_destination_block_ptr_cell_unit_it =
             tmp_ptr_destination_block_unit_it->ptr_array_cell_units,
        tmp_ptr_source_block_ptr_cell_unit_it =
             tmp_ptr_source_block_unit_it->ptr_array_cell_units;
         tmp_ptr_source_block_ptr_cell_unit_it !=
         tmp_ptr_source_block_ptr_last_unit;
         ++tmp_ptr_source_block_ptr_cell_unit_it,
        ++tmp_ptr_destination_block_ptr_cell_unit_it) {
      tmp_ptr_destination_array_ptr_connection_cell_units
          [tmp_ptr_destination_block_ptr_cell_unit_it
               ->index_peephole_input_gate] =
              this->ptr_array_cell_units +
              static_cast<size_t>(tmp_ptr_source_array_ptr_connection_cell_units
                                      [tmp_ptr_source_block_ptr_cell_unit_it
                                           ->index_peephole_input_gate] -
                                  ptr_source_first_cell_unit_received);
      tmp_ptr_destination_array_ptr_connection_cell_units
          [tmp_ptr_destination_block_ptr_cell_unit_it
               ->index_peephole_forget_gate] =
              this->ptr_array_cell_units +
              static_cast<size_t>(tmp_ptr_source_array_ptr_connection_cell_units
                                      [tmp_ptr_source_block_ptr_cell_unit_it
                                           ->index_peephole_forget_gate] -
                                  ptr_source_first_cell_unit_received);
      tmp_ptr_destination_array_ptr_connection_cell_units
          [tmp_ptr_destination_block_ptr_cell_unit_it
               ->index_peephole_output_gate] =
              this->ptr_array_cell_units +
              static_cast<size_t>(tmp_ptr_source_array_ptr_connection_cell_units
                                      [tmp_ptr_source_block_ptr_cell_unit_it
                                           ->index_peephole_output_gate] -
                                  ptr_source_first_cell_unit_received);
    }
    //    [1] |END| Peepholes. |END|
#endif
  }
}
}  // namespace DL::v1
