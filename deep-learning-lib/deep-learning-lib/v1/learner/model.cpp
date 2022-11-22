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
#include "deep-learning-lib/io/term/input.hpp"
#include "deep-learning-lib/ops/math.hpp"
#include "deep-learning-lib/v1/ops/activations/functions.hpp"
#include "deep-learning-lib/v1/mem/reallocate.hpp"

#include <iostream>

using namespace DL::Str;
using namespace DL::Term;

namespace DL::v1 {
void Model::Organize__Previous_Layers_Connected(
    size_t &ref_state_layer_index_received,
    Layer *const ptr_layer_received,
    Layer const *&ptr_layer_state_received) const {
  ptr_layer_received->previous_connected_layers.clear();

  if (ptr_layer_received <= this->ptr_array_layers) {
    return;
  }

  Layer const *tmp_ptr_previous_layer_connected;

  /* If the previous layer was inside a residual block.
     Connect the layer to the previous residual unit (identity-mapping
     shortcut). */
  if (ptr_layer_state_received != nullptr &&
      ref_state_layer_index_received++ ==
          ptr_layer_state_received->block_depth) {
    tmp_ptr_previous_layer_connected = ptr_layer_state_received;

    ptr_layer_received->previous_connected_layers.push_back(
        tmp_ptr_previous_layer_connected);

    ref_state_layer_index_received = 0_UZ;

    ptr_layer_state_received = nullptr;
  } else {
    tmp_ptr_previous_layer_connected = ptr_layer_received - 1;

    /* If the previous layer is a residual block.
       Get the previously connected layer from the previous layer (residual
       block). */
    if (tmp_ptr_previous_layer_connected->type_layer == LAYER::RESIDUAL) {
      tmp_ptr_previous_layer_connected =
          tmp_ptr_previous_layer_connected->previous_connected_layers[0];
    }

    ptr_layer_received->previous_connected_layers.push_back(
        tmp_ptr_previous_layer_connected);
  }

  /* If the layer is a residual block.
     Keep track the following layers are inside a residual block. */
  if (ptr_layer_received->type_layer == LAYER::RESIDUAL) {
    ptr_layer_state_received = ptr_layer_received;
  }
}

void Model::Organize__Next_Layers_Connected(
    size_t &ref_state_layer_index_received,
    Layer *const ptr_layer_received,
    Layer const *&ptr_layer_state_received) const {
  ptr_layer_received->next_connected_layers.clear();

  Layer const *tmp_ptr_next_layer_connected;

  // If the layer is a residual block. Add the next layer after the current
  // residual block.
  if (ptr_layer_received->type_layer == LAYER::RESIDUAL) {
    tmp_ptr_next_layer_connected =
        ptr_layer_received + ptr_layer_received->block_depth + 1_UZ;
  } else {
    tmp_ptr_next_layer_connected = ptr_layer_received + 1;
  }

  /* If the layer are inside a residual block and is the last layer inside the
     block. Connect the layer to the residual unit (identity-mapping shortcut).
   */
  if (ptr_layer_state_received != nullptr &&
      ++ref_state_layer_index_received ==
          ptr_layer_state_received->block_depth) {
    tmp_ptr_next_layer_connected = ptr_layer_state_received;

    ptr_layer_received->next_connected_layers.push_back(
        tmp_ptr_next_layer_connected);

    ref_state_layer_index_received = 0_UZ;

    ptr_layer_state_received = nullptr;
  } else if (tmp_ptr_next_layer_connected < this->ptr_last_layer) {
    // Push back the next connected layer.
    ptr_layer_received->next_connected_layers.push_back(
        tmp_ptr_next_layer_connected);

    // If the next layer is a residual block. shift the next layer by plus one
    // and push it back to the vector.
    if (tmp_ptr_next_layer_connected->type_layer == LAYER::RESIDUAL) {
      ++tmp_ptr_next_layer_connected;

      // Push back the next connected layer.
      ptr_layer_received->next_connected_layers.push_back(
          tmp_ptr_next_layer_connected);
    }
  }

  // Keep track the following layers are inside a residual block.
  if (ptr_layer_received->type_layer == LAYER::RESIDUAL) {
    ptr_layer_state_received = ptr_layer_received;
  }
}

void Model::Organize__Layer__Group(
    size_t &ref_state_layer_index_received,
    Layer *const ptr_layer_received,
    Layer const *&ptr_layer_state_received) const {
  // If the layer are inside a residual block.
  if (ptr_layer_state_received != nullptr) {
    // If is the last layer inside the block.
    if (++ref_state_layer_index_received ==
        ptr_layer_state_received->block_depth) {
      ref_state_layer_index_received = 0_UZ;

      ptr_layer_state_received = nullptr;
    }

    ptr_layer_received->type_group = GROUP::RESIDUAL;
  } else {
    ptr_layer_received->type_group = GROUP::NONE;
  }

  // Keep track the following layers are inside a residual block.
  if (ptr_layer_received->type_layer == LAYER::RESIDUAL) {
    ptr_layer_state_received = ptr_layer_received;
  }
}

bool Model::operator==(Model const &ref_source_Neural_Network_received) {
  if (&ref_source_Neural_Network_received == this) {
    return true;
  }

  return (
      this->total_layers == ref_source_Neural_Network_received.total_layers &&
      this->total_weights == ref_source_Neural_Network_received.total_weights &&
      this->total_bias == ref_source_Neural_Network_received.total_bias &&
      this->total_parameters ==
          ref_source_Neural_Network_received.total_parameters &&
      this->total_basic_units ==
          ref_source_Neural_Network_received.total_basic_units &&
      this->total_basic_indice_units ==
          ref_source_Neural_Network_received.total_basic_indice_units &&
      this->total_neuron_units ==
          ref_source_Neural_Network_received.total_neuron_units &&
      this->total_AF_units ==
          ref_source_Neural_Network_received.total_AF_units &&
      this->total_AF_Ind_recurrent_units ==
          ref_source_Neural_Network_received.total_AF_Ind_recurrent_units &&
      this->total_cell_units ==
          ref_source_Neural_Network_received.total_cell_units &&
      this->total_block_units ==
          ref_source_Neural_Network_received.total_block_units &&
      this->total_normalized_units ==
          ref_source_Neural_Network_received.total_normalized_units &&
      this->total_dropout_alpha_layers ==
          ref_source_Neural_Network_received.total_dropout_alpha_layers &&
      this->total_dropout_bernoulli_layers ==
          ref_source_Neural_Network_received.total_dropout_bernoulli_layers &&
      this->total_dropout_bernoulli_inverted_layers ==
          ref_source_Neural_Network_received
              .total_dropout_bernoulli_inverted_layers &&
      this->total_dropout_gaussian_layers ==
          ref_source_Neural_Network_received.total_dropout_gaussian_layers &&
      this->total_dropout_shakedrop_layers ==
          ref_source_Neural_Network_received.total_dropout_shakedrop_layers &&
      this->total_dropout_uout_layers ==
          ref_source_Neural_Network_received.total_dropout_uout_layers &&
      this->total_dropout_zoneout_layers ==
          ref_source_Neural_Network_received.total_dropout_zoneout_layers &&
      this->total_batch_normalization_layers ==
          ref_source_Neural_Network_received.total_batch_normalization_layers &&
      this->total_batch_renormalization_layers ==
          ref_source_Neural_Network_received
              .total_batch_renormalization_layers &&
      this->total_ghost_batch_normalization_layers ==
          ref_source_Neural_Network_received
              .total_ghost_batch_normalization_layers &&
      this->total_streaming_normalization_layers ==
          ref_source_Neural_Network_received
              .total_streaming_normalization_layers &&
      this->total_k_sparse_layers ==
          ref_source_Neural_Network_received.total_k_sparse_layers &&
      this->total_tied_parameter_layers ==
          ref_source_Neural_Network_received.total_tied_parameter_layers &&
      this->total_constraint_recurrent_weight_layers ==
          ref_source_Neural_Network_received
              .total_constraint_recurrent_weight_layers);
}

bool Model::operator!=(Model const &ref_source_Neural_Network_received) {
  return (!(*this == ref_source_Neural_Network_received));
}

bool Layer::Compare__Dimensions(
    Layer const &ref_source_Layer_received) const {
  return (
      static_cast<size_t>(this->ptr_last_basic_unit -
                          this->ptr_array_basic_units) ==
          static_cast<size_t>(
              ref_source_Layer_received.ptr_last_basic_unit -
              ref_source_Layer_received.ptr_array_basic_units) &&
      static_cast<size_t>(this->ptr_last_basic_indice_unit -
                          this->ptr_array_basic_indice_units) ==
          static_cast<size_t>(
              ref_source_Layer_received.ptr_last_basic_indice_unit -
              ref_source_Layer_received.ptr_array_basic_indice_units) &&
      static_cast<size_t>(this->ptr_last_neuron_unit -
                          this->ptr_array_neuron_units) ==
          static_cast<size_t>(
              ref_source_Layer_received.ptr_last_neuron_unit -
              ref_source_Layer_received.ptr_array_neuron_units) &&
      static_cast<size_t>(this->ptr_last_AF_unit - this->ptr_array_AF_units) ==
          static_cast<size_t>(ref_source_Layer_received.ptr_last_AF_unit -
                              ref_source_Layer_received.ptr_array_AF_units) &&
      static_cast<size_t>(this->ptr_last_AF_Ind_recurrent_unit -
                          this->ptr_array_AF_Ind_recurrent_units) ==
          static_cast<size_t>(
              ref_source_Layer_received.ptr_last_AF_Ind_recurrent_unit -
              ref_source_Layer_received.ptr_array_AF_Ind_recurrent_units) &&
      static_cast<size_t>(this->ptr_last_block_unit -
                          this->ptr_array_block_units) ==
          static_cast<size_t>(
              ref_source_Layer_received.ptr_last_block_unit -
              ref_source_Layer_received.ptr_array_block_units) &&
      static_cast<size_t>(this->ptr_last_cell_unit -
                          this->ptr_array_cell_units) ==
          static_cast<size_t>(ref_source_Layer_received.ptr_last_cell_unit -
                              ref_source_Layer_received.ptr_array_cell_units) &&
      static_cast<size_t>(this->ptr_last_normalized_unit -
                          this->ptr_array_normalized_units) ==
          static_cast<size_t>(
              ref_source_Layer_received.ptr_last_normalized_unit -
              ref_source_Layer_received.ptr_array_normalized_units));
}

bool Neural_Network_Initializer::Build__Layer__FC(
    Layer_Parameters &ref_Layer_Parameters_received) {
  if (ref_Layer_Parameters_received.type_layer == LAYER::FULLY_CONNECTED ||
      ref_Layer_Parameters_received.type_layer ==
          LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT ||
      ref_Layer_Parameters_received.type_layer ==
          LAYER::FULLY_CONNECTED_RECURRENT) {
    ref_Layer_Parameters_received.unit_parameters[0] =
        parse_discrete<size_t>(1_UZ, L"Neuron(s): ");
  } else {
    ERR(
        L"Layer type (%d | %ls) is not a fully connected layer.",
        ref_Layer_Parameters_received.type_layer,
        LAYER_NAME[ref_Layer_Parameters_received.type_layer].c_str());

    return false;
  }

  return true;
}

bool Neural_Network_Initializer::Build__Layer__Pooling(
    Layer_Parameters &ref_Layer_Parameters_received) {
  if (ref_Layer_Parameters_received.type_layer == LAYER::AVERAGE_POOLING ||
      ref_Layer_Parameters_received.type_layer == LAYER::MAX_POOLING) {
    ref_Layer_Parameters_received.unit_parameters[0] =
        parse_discrete<size_t>(1_UZ, L"Kernel size: ");

    ref_Layer_Parameters_received.unit_parameters[1] =
        parse_discrete<size_t>(1_UZ, L"Stride: ");

    ref_Layer_Parameters_received.unit_parameters[2] =
        parse_discrete<size_t>(0_UZ, L"Padding: ");

    ref_Layer_Parameters_received.unit_parameters[3] =
        parse_discrete<size_t>(0_UZ, L"Dilation: ");

    ref_Layer_Parameters_received.unit_parameters[4] =
        static_cast<size_t>(accept(L"Ceil mode?"));
  } else {
    ERR(
        L"Layer type (%d | %ls) is not a pooling layer.",
        ref_Layer_Parameters_received.type_layer,
        LAYER_NAME[ref_Layer_Parameters_received.type_layer].c_str());

    return false;
  }

  return true;
}

bool Neural_Network_Initializer::Build__Layer__LSTM(
    Layer_Parameters &ref_Layer_Parameters_received) {
  if (ref_Layer_Parameters_received.type_layer == LAYER::LSTM) {
    ref_Layer_Parameters_received.use_bidirectional =
        accept(L"Bidirectional?");

    ref_Layer_Parameters_received.unit_parameters[0] =
        parse_discrete<size_t>(1_UZ, L"Block(s): ");

    ref_Layer_Parameters_received.unit_parameters[1] =
        parse_discrete<size_t>(1_UZ, L"Cells(s) per block: ");
  } else {
    ERR(
        L"Layer type (%d | %ls) is not a LSTM layer.",
        ref_Layer_Parameters_received.type_layer,
        LAYER_NAME[ref_Layer_Parameters_received.type_layer].c_str());

    return false;
  }

  return true;
}

bool Neural_Network_Initializer::Build__Layer__Residual(void) {
  unsigned int tmp_layer_type_index;

  size_t tmp_residual_unit_index, tmp_number_residual_units, tmp_layer_index,
      tmp_block_depth;

  Layer_Parameters tmp_Layer_Parameters;

  INFO(L"");
  INFO(L"Number residual unit(s).");
  INFO(L"Range[1, inf].");
  tmp_number_residual_units =
      parse_discrete<size_t>(1_UZ, L"Number residual unit(s): ");

  INFO(L"");
  INFO(L"Block width.");
  INFO(L"Range[2, inf].");
  INFO(L"default=2.0.");
  tmp_block_depth = parse_discrete<size_t>(2_UZ, L"Block width: ");

  INFO(L"");
  if (accept(L"Use widening factor, alpha?")) {
    bool tmp_while(true);

    double tmp_widening_factor_alpha, tmp_widening_factors[2] = {0},
                                      tmp_widening_factor_units[2] = {0};

    Layer_Parameters tmp_widening_Layer_Parameters;

    // Residual unit type.
    do {
      INFO(L"");
      INFO(L"Layer type.");
      for (tmp_layer_type_index = 1u; tmp_layer_type_index != LAYER::LENGTH;
           ++tmp_layer_type_index) {
        INFO(
            L"[%d]: %ls.",
            tmp_layer_type_index,
            LAYER_NAME[static_cast<LAYER::TYPE>(tmp_layer_type_index)]
                .c_str());
      }
      INFO(L"default=%ls.",
             LAYER_NAME[LAYER::FULLY_CONNECTED].c_str());

      if ((tmp_widening_Layer_Parameters.type_layer =
               static_cast<LAYER::TYPE>(
                   parse_discrete<int>(
                       1, LAYER::LENGTH - 1, L"Residual unit, type: "))) >=
          LAYER::LENGTH) {
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 1,
            LAYER::LENGTH - 1u);

        return false;
      }

      switch (tmp_widening_Layer_Parameters.type_layer) {
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          if (this->Build__Layer__FC(tmp_widening_Layer_Parameters) == false) {
            ERR(
                L"An error has been triggered from the "
                "\"Build__Layer__FC()\" function.",);

            return false;
          }
          break;
        case LAYER::LSTM:
          if (this->Build__Layer__LSTM(tmp_widening_Layer_Parameters) ==
              false) {
            ERR(
                L"An error has been triggered from the "
                "\"Build__Layer__LSTM()\" function.",);

            return false;
          }
          break;
        default:
          ERR(
              L"Layer type (%d | %ls) is not managed in the "
              "switch.",
              tmp_widening_Layer_Parameters.type_layer,
              LAYER_NAME[tmp_widening_Layer_Parameters.type_layer].c_str());
          continue;
      }

      tmp_widening_factor_units[0] = static_cast<double>(
          tmp_widening_Layer_Parameters.unit_parameters[0]);
      tmp_widening_factor_units[1] = static_cast<double>(
          tmp_widening_Layer_Parameters.unit_parameters[1]);

      // Widening factor #0.
      INFO(L"");
      INFO(L"Widening factor, alpha[0].");
      INFO(L"Range[-%zu, inf] 0=Fixed.",
             tmp_widening_Layer_Parameters.unit_parameters[0] - 1_UZ);
      tmp_widening_factor_alpha =
          static_cast<double>(parse_discrete<long long>(
              -static_cast<long long int>(
                  tmp_widening_Layer_Parameters.unit_parameters[0]) +
                  1ll,
              L"Widening factor, alpha[0]: "));
      tmp_widening_factors[0] = tmp_widening_factor_alpha /
                                 static_cast<double>(tmp_number_residual_units);
      // |END| Widening factor #0. |END|

      // Widening factor #1.
      if (tmp_widening_factor_units[1] != 0.0) {
        INFO(L"");
        INFO(L"Widening factor, alpha[1].");
        INFO(L"Range[-%zu, inf] 0=Fixed.",
               tmp_widening_Layer_Parameters.unit_parameters[1] - 1_UZ);
        tmp_widening_factor_alpha =
            static_cast<double>(parse_discrete<long long>(
                -static_cast<long long int>(
                    tmp_widening_Layer_Parameters.unit_parameters[1]) +
                    1ll,
                L"Widening factor, alpha[1]: "));
        tmp_widening_factors[1] =
            tmp_widening_factor_alpha /
            static_cast<double>(tmp_number_residual_units);
      }
      // |END| Widening factor #1. |END|

      tmp_while = false;
    } while (tmp_while);

    // Loop through each remaining residual unit(s).
    for (tmp_residual_unit_index = 0_UZ;
         tmp_residual_unit_index != tmp_number_residual_units;
         ++tmp_residual_unit_index) {
      // Residual unit.
      tmp_Layer_Parameters.type_layer = LAYER::RESIDUAL;

      tmp_Layer_Parameters.unit_parameters[0] = tmp_block_depth;

      this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
      // |END| Residual unit. |END|

      // Building block.
      tmp_Layer_Parameters.type_layer =
          tmp_widening_Layer_Parameters.type_layer;

      switch (tmp_Layer_Parameters.type_layer) {
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          tmp_widening_factor_units[0] += tmp_widening_factors[0];
          tmp_Layer_Parameters.unit_parameters[0] =
              static_cast<size_t>(tmp_widening_factor_units[0]);
          break;
        case LAYER::LSTM:
          tmp_Layer_Parameters.use_bidirectional =
              tmp_widening_Layer_Parameters.use_bidirectional;

          tmp_widening_factor_units[0] += tmp_widening_factors[0];
          tmp_Layer_Parameters.unit_parameters[0] =
              static_cast<size_t>(tmp_widening_factor_units[0]);

          tmp_widening_factor_units[1] += tmp_widening_factors[1];
          tmp_Layer_Parameters.unit_parameters[1] =
              static_cast<size_t>(tmp_widening_factor_units[1]);
          break;
        default:
          ERR(
              L"Layer type (%d | %ls) is not managed in the "
              "switch.",
              tmp_Layer_Parameters.type_layer,
              LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
          return false;
      }

      for (tmp_layer_index = 0_UZ; tmp_layer_index != tmp_block_depth;
           ++tmp_layer_index) {
        this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
      }
      // |END| Building block. |END|
    }
  } else {
    // Loop through each remaining residual unit(s).
    for (tmp_residual_unit_index = 0_UZ;
         tmp_residual_unit_index != tmp_number_residual_units;
         ++tmp_residual_unit_index) {
      // Residual unit.
      tmp_Layer_Parameters.type_layer = LAYER::RESIDUAL;

      tmp_Layer_Parameters.unit_parameters[0] = tmp_block_depth;

      this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
      // |END| Residual unit. |END|

      // Building block.
      for (tmp_layer_index = 0_UZ; tmp_layer_index != tmp_block_depth;
           ++tmp_layer_index) {
        INFO(L"");
        INFO(L"Layer type.");
        for (tmp_layer_type_index = 1u;
             tmp_layer_type_index != LAYER::LENGTH;
             ++tmp_layer_type_index) {
          INFO(
              L"[%d]: %ls.",
              tmp_layer_type_index,
              LAYER_NAME[static_cast<LAYER::TYPE>(tmp_layer_type_index)]
                  .c_str());
        }
        INFO(L"default=%ls.",
               LAYER_NAME[LAYER::FULLY_CONNECTED].c_str());

        if ((tmp_Layer_Parameters.type_layer = static_cast<LAYER::TYPE>(
                 parse_discrete<int>(
                     1, LAYER::LENGTH - 1,
                     (L"Residual[" + std::to_wstring(tmp_residual_unit_index) +
                      L"], layer[" + std::to_wstring(tmp_layer_index) +
                      L"], type: ")
                         .c_str()))) >= LAYER::LENGTH) {
          ERR(
              L"An error has been triggered from the "
              "\"parse_discrete<int>(%u, %u)\" function.", 1,
              LAYER::LENGTH - 1u);

          return false;
        }

        switch (tmp_Layer_Parameters.type_layer) {
          case LAYER::AVERAGE_POOLING:
          case LAYER::MAX_POOLING:
            if (this->Build__Layer__Pooling(tmp_Layer_Parameters) == false) {
              ERR(
                  L"An error has been triggered from the "
                  "\"Build__Layer__Pooling()\" function.",);

              return false;
            }
            break;
          case LAYER::FULLY_CONNECTED:
          case LAYER::FULLY_CONNECTED_RECURRENT:
          case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
            if (this->Build__Layer__FC(tmp_Layer_Parameters) == false) {
              ERR(
                  L"An error has been triggered from the "
                  "\"Build__Layer__FC()\" function.",);

              return false;
            }
            break;
          case LAYER::LSTM:
            if (this->Build__Layer__LSTM(tmp_Layer_Parameters) == false) {
              ERR(
                  L"An error has been triggered from the "
                  "\"Build__Layer__LSTM()\" function.",);

              return false;
            }
            break;
          default:
            ERR(
                L"Layer type (%d | %ls) is not managed in the "
                "switch.",
                tmp_Layer_Parameters.type_layer,
                LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
            continue;
        }

        this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
      }
    }
  }

  return true;
}

bool Neural_Network_Initializer::While__Push_Back__Layer(void) {
  unsigned int tmp_layer_type_index;

  size_t tmp_layer_index(this->vector_layers_parameters.size());

  Layer_Parameters tmp_Layer_Parameters;

  while (this->vector_layers_parameters.size() < 2_UZ ||
         accept(
             (L"Add another layer (" + std::to_wstring(tmp_layer_index) + L")?")
                 .c_str())) {
    INFO(L"");
    INFO(L"Layer type.");
    for (tmp_layer_type_index = 1u; tmp_layer_type_index != LAYER::LENGTH;
         ++tmp_layer_type_index) {
      INFO(L"[%d]: %ls.",
             tmp_layer_type_index,
             LAYER_NAME[static_cast<LAYER::TYPE>(tmp_layer_type_index)]
                 .c_str());
    }
    INFO(L"default=%ls.",
           LAYER_NAME[LAYER::FULLY_CONNECTED].c_str());

    if ((tmp_Layer_Parameters.type_layer = static_cast<LAYER::TYPE>(
             parse_discrete<int>(
                 1, LAYER::LENGTH - 1,
                 (L"Hidden layer " + std::to_wstring(tmp_layer_index) + L" type: ")
                     .c_str()))) >= LAYER::LENGTH) {
      ERR(
          L"An error has been triggered from the "
          "\"parse_discrete<int>(%u, %u)\" function.", 1,
          LAYER::LENGTH - 1u);

      return false;
    }

    switch (tmp_Layer_Parameters.type_layer) {
      case LAYER::AVERAGE_POOLING:
      case LAYER::MAX_POOLING:
        if (this->Build__Layer__Pooling(tmp_Layer_Parameters) == false) {
          ERR(
              L"An error has been triggered from the "
              "\"Build__Layer__Pooling()\" function.",);

          return false;
        }

        this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
        break;
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_RECURRENT:
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        if (this->Build__Layer__FC(tmp_Layer_Parameters) == false) {
          ERR(
              L"An error has been triggered from the "
              "\"Build__Layer__FC()\" function.",);

          return false;
        }

        this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
        break;
      case LAYER::LSTM:
        if (this->Build__Layer__LSTM(tmp_Layer_Parameters) == false) {
          ERR(
              L"An error has been triggered from the "
              "\"Build__Layer__LSTM()\" function.",);

          return false;
        }

        this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
        break;
      case LAYER::RESIDUAL:
        if (tmp_layer_index <= 1_UZ) {
          ERR(
              L"Layer type (%d | %ls) is not managed as the first "
              "hidden/input layer.",
              tmp_Layer_Parameters.type_layer,
              LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());

          continue;
        } else if (this->type_neural_network == MODEL::AUTOENCODER) {
          ERR(
              L"The autoencoder network can not use residual "
              "layer.",);

          continue;
        }

        if (this->Build__Layer__Residual() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"Build_Residual_Block()\" function.",);

          return false;
        }
        break;
      default:
        ERR(
            L"Layer type (%d | %ls) is not managed in the switch.",
            tmp_Layer_Parameters.type_layer,
            LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
        continue;
    }

    tmp_layer_index = this->vector_layers_parameters.size();
  }

  return true;
}

bool Neural_Network_Initializer::Input_Initialize(void) {
  size_t tmp_layer_index, tmp_layer_length;

  INFO(L"");
  INFO(L"Network type.");
  for (unsigned int tmp_network_type_index(1u);
       tmp_network_type_index != MODEL::LENGTH; ++tmp_network_type_index) {
    INFO(L"[%d]: %ls.",
           tmp_network_type_index,
           MODEL_NAME[static_cast<MODEL::TYPE>(tmp_network_type_index)]
               .c_str());
  }
  INFO(L"default=%ls.",
         MODEL_NAME[MODEL::FEEDFORWARD].c_str());

  if ((this->type_neural_network =
           static_cast<MODEL::TYPE>(parse_discrete<int>(
               1, MODEL::LENGTH - 1, L"Type: "))) >= MODEL::LENGTH) {
    ERR(
        L"An error has been triggered from the "
        "\"parse_discrete<int>(%u, %u)\" function.", 1,
        MODEL::LENGTH - 1u);

    return false;
  }

  INFO(L"");
  this->seq_w =
      parse_discrete<size_t>(1_UZ, L"Recurrent depth: ");

  if (this->seq_w > 1_UZ) {
    INFO(L"");
    INFO(L"Time delays.");
    INFO(L"Range[0, %zu].",
           this->seq_w - 1_UZ);
    this->n_time_delay = parse_discrete<size_t>(
        0_UZ, this->seq_w - 1_UZ, L"Time delays: ");
  }

  Layer_Parameters tmp_Layer_Parameters;

  INFO(L"");
  INFO(L"Input layer:");
  tmp_Layer_Parameters.type_layer = LAYER::FULLY_CONNECTED;
  tmp_Layer_Parameters.unit_parameters[0] =
      parse_discrete<size_t>(1_UZ, L"Number of inputs: ");
  this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

  switch (this->type_neural_network) {
    case MODEL::AUTOENCODER:
      if (this->While__Push_Back__Layer() == false) {
        ERR(
            L"An error has been triggered from the "
            "\"While__Push_Back__Layer()\" function.",);

        return false;
      }

      tmp_layer_length = this->vector_layers_parameters.size() - 1_UZ;

      for (tmp_layer_index = 1_UZ; tmp_layer_index != tmp_layer_length;
           ++tmp_layer_index) {
        tmp_Layer_Parameters.type_layer =
            this->vector_layers_parameters[tmp_layer_length - tmp_layer_index]
                .type_layer;

        tmp_Layer_Parameters.use_bidirectional =
            this->vector_layers_parameters[tmp_layer_length - tmp_layer_index]
                .use_bidirectional;

        tmp_Layer_Parameters.unit_parameters[0] =
            this->vector_layers_parameters[tmp_layer_length - tmp_layer_index]
                .unit_parameters[0];
        tmp_Layer_Parameters.unit_parameters[1] =
            this->vector_layers_parameters[tmp_layer_length - tmp_layer_index]
                .unit_parameters[0];

        this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
      }

      tmp_Layer_Parameters.type_layer = LAYER::FULLY_CONNECTED;
      tmp_Layer_Parameters.unit_parameters[0] =
          this->vector_layers_parameters[0].unit_parameters[0];
      this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
      break;
    default:
      if (this->While__Push_Back__Layer() == false) {
        ERR(
            L"An error has been triggered from the "
            "\"While__Push_Back__Layer()\" function.",);

        return false;
      }

      INFO(L"");
      INFO(L"Output layer:");
      tmp_Layer_Parameters.type_layer = LAYER::FULLY_CONNECTED;
      tmp_Layer_Parameters.unit_parameters[0] =
          parse_discrete<size_t>(1_UZ, L"Number of output(s): ");
      this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
      break;
  }

  return true;
}

bool Neural_Network_Initializer::Template_Initialize(void) {
  this->type_neural_network = MODEL::RECURRENT;

  INFO(L"");
  this->seq_w =
      parse_discrete<size_t>(1_UZ, L"Recurrent depth: ");

  if (this->seq_w > 1_UZ) {
    INFO(L"");
    INFO(L"Time delays.");
    INFO(L"Range[0, %zu].",
           this->seq_w - 1_UZ);
    this->n_time_delay = parse_discrete<size_t>(
        0_UZ, this->seq_w - 1_UZ, L"Time delays: ");
  }

  bool tmp_use_pooling, tmp_use_bottleneck;

  double tmp_widening_factor_alpha, tmp_widening_factors[2] = {0},
                                    tmp_widening_factor_units[2] = {0};

  size_t tmp_residual_unit_index, tmp_number_residual_units, tmp_layer_index,
      tmp_block_depth, tmp_pooling_layer_mod;

  Layer_Parameters tmp_Layer_Parameters, tmp_pooling_layer_Parameters,
      tmp_widening_Layer_Parameters;

  // Input layer.
  INFO(L"");
  INFO(L"Input layer:");
  tmp_Layer_Parameters.type_layer = LAYER::FULLY_CONNECTED;
  tmp_Layer_Parameters.unit_parameters[0] =
      parse_discrete<size_t>(1_UZ, L"Number of inputs: ");
  this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
  // |END| Input layer. |END|

  // #0: Fully connected, independently recurrent.
  INFO(L"");
  tmp_Layer_Parameters.type_layer =
      LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT;
  tmp_Layer_Parameters.unit_parameters[0] = parse_discrete<size_t>(
      1_UZ, L"First hidden layer, number units: ");
  this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

  INFO(L"Layer[%zu]: Type: %ls.",
         this->vector_layers_parameters.size() - 1_UZ,
         LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
  INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
         this->vector_layers_parameters.size() - 1_UZ,
         tmp_Layer_Parameters.unit_parameters[0]);
  // |END| #0: Fully connected, independently recurrent. |END|

  // #1: Residual, Fully connected, independently recurrent.
  INFO(L"");
  INFO(L"Residual block.");
  INFO(L"Range[1, inf].");
  tmp_number_residual_units =
      parse_discrete<size_t>(1_UZ, L"Number residual units: ");

  INFO(L"");
  INFO(L"Residual block, depth.");
  INFO(L"Range[2, inf].");
  INFO(L"default=3.");
  tmp_block_depth = parse_discrete<size_t>(2_UZ, L"Block depth: ");

  //  Pooling.
  INFO(L"");
  if ((tmp_use_pooling = accept(L"Do you want to use pooling?"))) {
    if (this->Build__Layer__Pooling(tmp_pooling_layer_Parameters) == false) {
      ERR(
          L"An error has been triggered from the "
          "\"Build__Layer__Pooling()\" function.",);

      return false;
    }

    INFO(L"");
    INFO(L"Pooling layer, frequency.");
    INFO(L"Range[1, %zu].",
           tmp_number_residual_units);
    INFO(L"default=%zu.",
           static_cast<size_t>(
               ceil(static_cast<double>(tmp_number_residual_units) / 3.0)));
    tmp_pooling_layer_mod =
        parse_discrete<size_t>(1_UZ, L"Pooling layer, frequency: ");
  }

  //  Bottleneck.
  if (tmp_block_depth > 2_UZ) {
    INFO(L"");
    INFO(L"default=true.");
    tmp_use_bottleneck = accept(L"Do you want to use bottleneck?");
  }

  tmp_widening_Layer_Parameters.type_layer =
      LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT;
  tmp_widening_Layer_Parameters.unit_parameters[0] =
      tmp_Layer_Parameters.unit_parameters[0];

  //  Widening factors.
  tmp_widening_factor_units[0] =
      static_cast<double>(tmp_widening_Layer_Parameters.unit_parameters[0]);
  tmp_widening_factor_units[1] =
      static_cast<double>(tmp_widening_Layer_Parameters.unit_parameters[1]);

  //      Widening factor #0.
  INFO(L"");
  INFO(L"Widening factor, alpha[0].");
  INFO(L"Range[-%zu, inf] 0=Fixed.",
         tmp_widening_Layer_Parameters.unit_parameters[0] - 1_UZ);
  tmp_widening_factor_alpha =
      static_cast<double>(parse_discrete<long long int>(
          -static_cast<long long>(
              tmp_widening_Layer_Parameters.unit_parameters[0]) +
              1ll,
          L"Widening factor, alpha[0]: "));
  tmp_widening_factors[0] = tmp_widening_factor_alpha /
                             static_cast<double>(tmp_number_residual_units);
  //      |END| Widening factor #0. |END|

  //      Widening factor #1.
  if (tmp_widening_factor_units[1] != 0.0) {
    INFO(L"");
    INFO(L"Widening factor, alpha[1].");
    INFO(L"Range[-%zu, inf] 0=Fixed.",
           tmp_widening_Layer_Parameters.unit_parameters[1] - 1_UZ);
    tmp_widening_factor_alpha =
        static_cast<double>(parse_discrete<long long int>(
            -static_cast<long long>(
                tmp_widening_Layer_Parameters.unit_parameters[1]) +
                1ll,
            L"Widening factor, alpha[1]: "));
    tmp_widening_factors[1] = tmp_widening_factor_alpha /
                               static_cast<double>(tmp_number_residual_units);
  }
  //      |END| Widening factor #1. |END|
  //  |END| Widening factors. |END|

  //  Loop through each remaining residual unit(s).
  for (tmp_residual_unit_index = 0_UZ;
       tmp_residual_unit_index != tmp_number_residual_units;
       ++tmp_residual_unit_index) {
    // Residual unit.
    tmp_Layer_Parameters.type_layer = LAYER::RESIDUAL;
    tmp_Layer_Parameters.unit_parameters[0] = tmp_block_depth;
    this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

    INFO(L"Layer[%zu]: Type: %ls.",
           this->vector_layers_parameters.size() - 1_UZ,
           LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
    INFO(L"Layer[%zu]: Block depth: %zu.",
           this->vector_layers_parameters.size() - 1_UZ,
           tmp_Layer_Parameters.unit_parameters[0]);
    // |END| Residual unit. |END|

    // Building block.
    tmp_Layer_Parameters.type_layer = tmp_widening_Layer_Parameters.type_layer;

    switch (tmp_Layer_Parameters.type_layer) {
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_RECURRENT:
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        tmp_layer_index = 0_UZ;

        if (tmp_use_bottleneck) {
          // First hidden layer inside the residual block.
          tmp_Layer_Parameters.unit_parameters[0] =
              static_cast<size_t>(tmp_widening_factor_units[0]);

          this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

          INFO(L"Layer[%zu]: Type: %ls.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
          INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 tmp_Layer_Parameters.unit_parameters[0]);

          ++tmp_layer_index;
          // |END| First hidden layer inside the residual block. |END|

          // Second hidden layer inside the residual block.
          tmp_Layer_Parameters.unit_parameters[0] = static_cast<size_t>(
              std::max<double>(
                  tmp_widening_factor_units[0],
                  tmp_widening_factor_units[0] + tmp_widening_factors[0]) /
              2.0);

          this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

          INFO(L"Layer[%zu]: Type: %ls.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
          INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 tmp_Layer_Parameters.unit_parameters[0]);

          ++tmp_layer_index;
          // |END| Second hidden layer inside the residual block. |END|
        }

        tmp_widening_factor_units[0] += tmp_widening_factors[0];
        tmp_Layer_Parameters.unit_parameters[0] =
            static_cast<size_t>(tmp_widening_factor_units[0]);

        for (; tmp_layer_index != tmp_block_depth; ++tmp_layer_index) {
          this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

          INFO(L"Layer[%zu]: Type: %ls.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
          INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 tmp_Layer_Parameters.unit_parameters[0]);
        }
        break;
      case LAYER::LSTM:
        tmp_Layer_Parameters.use_bidirectional =
            tmp_widening_Layer_Parameters.use_bidirectional;

        tmp_layer_index = 0_UZ;

        if (tmp_use_bottleneck) {
          // First hidden layer inside the residual block.
          tmp_Layer_Parameters.unit_parameters[0] =
              static_cast<size_t>(tmp_widening_factor_units[0]);
          tmp_Layer_Parameters.unit_parameters[1] =
              static_cast<size_t>(tmp_widening_factor_units[1]);

          this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

          INFO(L"Layer[%zu]: Type: %ls.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
          INFO(L"Layer[%zu]: Number block unit(s): %zu.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 tmp_Layer_Parameters.unit_parameters[0]);
          INFO(L"Layer[%zu]: Number cell unit(s) per block: %zu.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 tmp_Layer_Parameters.unit_parameters[1]);

          ++tmp_layer_index;
          // |END| First hidden layer inside the residual block. |END|

          // Second hidden layer inside the residual block.
          tmp_Layer_Parameters.unit_parameters[0] = static_cast<size_t>(
              std::max<double>(
                  tmp_widening_factor_units[0],
                  tmp_widening_factor_units[0] + tmp_widening_factors[0]) /
              2.0);
          tmp_Layer_Parameters.unit_parameters[1] = static_cast<size_t>(
              std::max<double>(
                  tmp_widening_factor_units[1],
                  tmp_widening_factor_units[1] + tmp_widening_factors[1]) /
              2.0);

          this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

          INFO(L"Layer[%zu]: Type: %ls.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
          INFO(L"Layer[%zu]: Number block unit(s): %zu.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 tmp_Layer_Parameters.unit_parameters[0]);
          INFO(L"Layer[%zu]: Number cell unit(s) per block: %zu.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 tmp_Layer_Parameters.unit_parameters[1]);

          ++tmp_layer_index;
          // |END| Second hidden layer inside the residual block. |END|
        }

        tmp_widening_factor_units[0] += tmp_widening_factors[0];
        tmp_Layer_Parameters.unit_parameters[0] =
            static_cast<size_t>(tmp_widening_factor_units[0]);

        tmp_widening_factor_units[1] += tmp_widening_factors[1];
        tmp_Layer_Parameters.unit_parameters[1] =
            static_cast<size_t>(tmp_widening_factor_units[1]);

        for (; tmp_layer_index != tmp_block_depth; ++tmp_layer_index) {
          this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

          INFO(L"Layer[%zu]: Type: %ls.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
          INFO(L"Layer[%zu]: Number block unit(s): %zu.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 tmp_Layer_Parameters.unit_parameters[0]);
          INFO(L"Layer[%zu]: Number cell unit(s) per block: %zu.",
                 this->vector_layers_parameters.size() - 1_UZ,
                 tmp_Layer_Parameters.unit_parameters[1]);
        }
        break;
      default:
        ERR(
            L"Layer type (%d | %ls) is not managed in the switch.",
            tmp_Layer_Parameters.type_layer,
            LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
        return false;
    }
    // |END| Building block. |END|

    //  Pooling layer.
    if (tmp_use_pooling &&
        (tmp_residual_unit_index + 1_UZ) % tmp_pooling_layer_mod == 0_UZ) {
      this->vector_layers_parameters.push_back(tmp_pooling_layer_Parameters);

      INFO(L"Layer[%zu]: Type: %ls.",
             this->vector_layers_parameters.size() - 1_UZ,
             LAYER_NAME[tmp_pooling_layer_Parameters.type_layer].c_str());
      INFO(L"Layer[%zu]: Kernel size: %zu.",
             this->vector_layers_parameters.size() - 1_UZ,
             tmp_pooling_layer_Parameters.unit_parameters[0]);
      INFO(L"Layer[%zu]: Stride: %zu.",
             this->vector_layers_parameters.size() - 1_UZ,
             tmp_pooling_layer_Parameters.unit_parameters[1]);
      INFO(L"Layer[%zu]: Padding: %zu.",
             this->vector_layers_parameters.size() - 1_UZ,
             tmp_pooling_layer_Parameters.unit_parameters[2]);
      INFO(L"Layer[%zu]: Dilation: %zu.",
             this->vector_layers_parameters.size() - 1_UZ,
             tmp_pooling_layer_Parameters.unit_parameters[3]);
      INFO(L"Layer[%zu]: Ceil mode: %ls.",
             this->vector_layers_parameters.size() - 1_UZ,
             tmp_pooling_layer_Parameters.unit_parameters[4] != 0_UZ
                 ? "true"
                 : "false");
    }
    //  |END| Pooling layer. |END|
  }
  // |END| #1: Residual, Fully connected, independently recurrent. |END|

  // #2: Fully connected, independently recurrent.
  tmp_Layer_Parameters.type_layer =
      LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT;
  tmp_Layer_Parameters.unit_parameters[0] =
      static_cast<size_t>(tmp_widening_factor_units[0]);
  this->vector_layers_parameters.push_back(tmp_Layer_Parameters);

  INFO(L"Layer[%zu]: Type: %ls.",
         this->vector_layers_parameters.size() - 1_UZ,
         LAYER_NAME[tmp_Layer_Parameters.type_layer].c_str());
  INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
         this->vector_layers_parameters.size() - 1_UZ,
         tmp_Layer_Parameters.unit_parameters[0]);
  // |END| #2: Fully connected, independently recurrent. |END|

  // Output layer.
  INFO(L"");
  INFO(L"Output layer:");
  tmp_Layer_Parameters.type_layer = LAYER::FULLY_CONNECTED;
  tmp_Layer_Parameters.unit_parameters[0] =
      parse_discrete<size_t>(1_UZ, L"Number of output(s): ");
  this->vector_layers_parameters.push_back(tmp_Layer_Parameters);
  // |END| Output layer. |END|

  return true;
}

Model *Neural_Network_Initializer::Output_Initialize(
    size_t const allowable_memory) const {
  if (this->vector_layers_parameters.empty()) {
    ERR(
        L"The vector \"layers parameters\" is empty.",);

    return nullptr;
  }

  if (sizeof(Model) > allowable_memory) {
    ERR(L"Can not allocate %zu bytes.", sizeof(Model));

    return nullptr;
  }

  Model *tmp_ptr_Neural_Network(new Model);
  if (tmp_ptr_Neural_Network == nullptr) {
    ERR(L"Can not allocate %zu bytes.", sizeof(Model));

    return nullptr;
  }

  if (tmp_ptr_Neural_Network->compile(
          this->vector_layers_parameters.size(), this->seq_w,
          this->type_neural_network, this->vector_layers_parameters.data(),
          allowable_memory) == false) {
    ERR(
        L"An error has been triggered from the \"compile(%zu, "
        "ptr, %zu)\" function.",
        this->vector_layers_parameters.size(), allowable_memory);

    SAFE_DELETE(tmp_ptr_Neural_Network);
  } else if (tmp_ptr_Neural_Network->set_seq_w(
                 this->n_time_delay) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"set_seq_w(%zu)\" function.",
        this->n_time_delay);

    SAFE_DELETE(tmp_ptr_Neural_Network);
  }

  return (tmp_ptr_Neural_Network);
}

Neural_Network_Initializer::~Neural_Network_Initializer(void) {}

bool Activation_Function_Initializer::Input_Initialize(
    size_t const n_layers,
    MODEL::TYPE const type_network_received) {
  size_t tmp_layer_index;

  if (this->Allocate__Layers_Activation_Function(n_layers) ==
      false) {
    ERR(
        L"An error has been triggered from the "
        "\"Allocate__Layers_Activation_Function(%zu)\" function.", n_layers);

    return false;
  }

  INFO(L"");
  INFO(L"Activation function initializer:");

  INFO(L"Activation functions:");
  for (unsigned int tmp_activation_function_index(1u);
       tmp_activation_function_index != ACTIVATION::LENGTH;
       ++tmp_activation_function_index) {
    INFO(L"[%d]: %ls.",
           tmp_activation_function_index,
           ACTIVATION_NAME[static_cast<ACTIVATION::TYPE>(
                                   tmp_activation_function_index)]
               .c_str());
  }
  INFO(L"default=%ls.",
         ACTIVATION_NAME[ACTIVATION::LEAKY_RELU].c_str());

  // Input layer.
  this->ptr_array_type_layers_activation_function[0] = ACTIVATION::LINEAR;

  switch (type_network_received) {
    case MODEL::AUTOENCODER:
      // Encoded layer(s).
      for (tmp_layer_index = 1_UZ;
           tmp_layer_index != (n_layers - 3_UZ) / 2_UZ + 1_UZ;
           ++tmp_layer_index) {
        if ((this->ptr_array_type_layers_activation_function[tmp_layer_index] =
                 static_cast<ACTIVATION::TYPE>(
                     parse_discrete<int>(
                         1, ACTIVATION::LENGTH - 1,
                         (L"Encoded layer " + std::to_wstring(tmp_layer_index) +
                          L", activation function: ")
                             .c_str()))) >= ACTIVATION::LENGTH) {
          ERR(
              L"An error has been triggered from the "
              "\"parse_discrete<int>(%u, %u)\" function.", 1,
              ACTIVATION::LENGTH - 1u);

          return false;
        }
      }

      // Coded layer.
      if ((this->ptr_array_type_layers_activation_function[tmp_layer_index] =
               static_cast<ACTIVATION::TYPE>(
                   parse_discrete<int>(
                       1, ACTIVATION::LENGTH - 1,
                       L"Coded layer, activation function: "))) >=
          ACTIVATION::LENGTH) {
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 1,
            ACTIVATION::LENGTH - 1u);

        return false;
      }

      // Decoded layer(s).
      for (++tmp_layer_index; tmp_layer_index != n_layers;
           ++tmp_layer_index) {
        this->ptr_array_type_layers_activation_function[tmp_layer_index] =
            this->ptr_array_type_layers_activation_function
                [n_layers - tmp_layer_index -
                 1_UZ];  // Subtract coded layer.
      }
      break;
    default:
      // Hidden layer(s).
      for (tmp_layer_index = 1_UZ;
           tmp_layer_index != n_layers - 1_UZ;
           ++tmp_layer_index) {
        if ((this->ptr_array_type_layers_activation_function[tmp_layer_index] =
                 static_cast<ACTIVATION::TYPE>(
                     parse_discrete<int>(
                         1, ACTIVATION::LENGTH - 1,
                         (L"Hidden layer " + std::to_wstring(tmp_layer_index) +
                          L", activation function: ")
                             .c_str()))) >= ACTIVATION::LENGTH) {
          ERR(
              L"An error has been triggered from the "
              "\"parse_discrete<int>(%u, %u)\" function.", 1,
              ACTIVATION::LENGTH - 1u);

          return false;
        }
      }

      // Output layer.
      INFO(L"");
      INFO(L"Output layer:");
      INFO(L"default=%ls.",
             ACTIVATION_NAME[ACTIVATION::SIGMOID].c_str());

      if ((this->ptr_array_type_layers_activation_function
               [n_layers - 1_UZ] =
               static_cast<ACTIVATION::TYPE>(
                   parse_discrete<int>(
                       1, ACTIVATION::LENGTH - 1,
                       L"Output layer, activation function: "))) >=
          ACTIVATION::LENGTH) {
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 1,
            ACTIVATION::LENGTH - 1u);

        return false;
      }
      // |END| Output layer. |END|
      break;
  }

  return true;
}

bool Activation_Function_Initializer::Output_Initialize(
    Model *const ptr_Neural_Network_received) const {
  if (this->ptr_array_type_layers_activation_function == nullptr) {
    ERR(
        L"\"ptr_array_type_layers_activation_function\" is a "
        "nullptr.",);

    return false;
  }

  size_t const tmp_number_layers(std::min<size_t>(
      this->number_layers, ptr_Neural_Network_received->total_layers));
  size_t tmp_layer_index;

  for (tmp_layer_index = 0_UZ; tmp_layer_index != tmp_number_layers;
       ++tmp_layer_index) {
    if (ptr_Neural_Network_received->set_layer_activation_function(
            tmp_layer_index,
            this->ptr_array_type_layers_activation_function[tmp_layer_index]) ==
        false) {
      ERR(
          L"An error has been triggered from the "
          "\"set_layer_activation_function(%zu, %u)\" function.", tmp_layer_index,
          this->ptr_array_type_layers_activation_function[tmp_layer_index]);

      return false;
    }
  }

  return true;
}

void Activation_Function_Initializer::Deallocate_Layers_Activation_Function(
    void) {
  SAFE_DELETE_ARRAY(this->ptr_array_type_layers_activation_function);
}

bool Activation_Function_Initializer::Allocate__Layers_Activation_Function(
    size_t const n_layers) {
  if (this->number_layers == 0_UZ) {
    if (this->ptr_array_type_layers_activation_function == nullptr) {
      ACTIVATION::TYPE *tmp_ptr_array_type_layers_activation_function(
          new ACTIVATION::TYPE[n_layers]);
      if (tmp_ptr_array_type_layers_activation_function == nullptr) {
        ERR(L"Can not allocate %zu bytes.",
               n_layers * sizeof(ACTIVATION::TYPE));

        return false;
      }
      memset(tmp_ptr_array_type_layers_activation_function, 0,
             n_layers * sizeof(ACTIVATION::TYPE));

      this->ptr_array_type_layers_activation_function =
          tmp_ptr_array_type_layers_activation_function;
    }

    this->number_layers = n_layers;
  }

  return true;
}

Activation_Function_Initializer::~Activation_Function_Initializer(void) {
  this->Deallocate_Layers_Activation_Function();
}

bool Loss_Function_Initializer::Input_Initialize(void) {
  INFO(L"");
  INFO(L"Loss functions:");
  for (unsigned int tmp_loss_function_index(1u);
       tmp_loss_function_index != LOSS_FN::LENGTH;
       ++tmp_loss_function_index) {
    INFO(L"[%d]: %ls.",
           tmp_loss_function_index,
           LOSS_FN_NAME[static_cast<LOSS_FN::TYPE>(
                                tmp_loss_function_index)]
               .c_str());
  }
  INFO(L"default=%ls.",
         LOSS_FN_NAME[LOSS_FN::RMSE].c_str());

  if ((this->type_loss_function = static_cast<LOSS_FN::TYPE>(
           parse_discrete<int>(1, LOSS_FN::LENGTH - 1,
                                                  L"Choose: "))) >=
      LOSS_FN::LENGTH) {
    ERR(
        L"An error has been triggered from the "
        "\"parse_discrete<int>(%u, %u)\" function.", 1,
        LOSS_FN::LENGTH - 1u);

    return false;
  }

  if (this->type_loss_function == LOSS_FN::BIT) {
    INFO(L"Loss function, BIT.");
    INFO(L"Range[0.0, 1.0].");
    INFO(L"default=0.5.");

    this->bit_fail_limit =
        parse_real(0_r, 1_r, L"Bit fail limit: ");
  }

  return true;
}

void Loss_Function_Initializer::Output_Initialize(
    Model *const ptr_Neural_Network_received) const {
  ptr_Neural_Network_received->set_loss_fn(this->type_loss_function);

  if (this->type_loss_function == LOSS_FN::BIT) {
    ptr_Neural_Network_received->set_bit_fail_limit(this->bit_fail_limit);
  }
}

bool Accuracy_Function_Initializer::Input_Initialize(void) {
  INFO(L"");
  INFO(L"Accuracy functions:");
  for (unsigned int tmp_type_accuracy_function_index(1u);
       tmp_type_accuracy_function_index != ACCU_FN::LENGTH;
       ++tmp_type_accuracy_function_index) {
    INFO(L"[%d]: %ls.",
           tmp_type_accuracy_function_index,
           ACC_FN_NAME[static_cast<ACCU_FN::TYPE>(
                               tmp_type_accuracy_function_index)]
               .c_str());
  }
  INFO(L"default=%ls.",
         ACC_FN_NAME[ACCU_FN::DISTANCE].c_str());

  if ((this->type_accuracy_function =
           static_cast<ACCU_FN::TYPE>(parse_discrete<int>(
               1, ACCU_FN::LENGTH - 1, L"Choose: "))) >=
      ACCU_FN::LENGTH) {
    ERR(
        L"An error has been triggered from the "
        "\"parse_discrete<int>(%u, %u)\" function.", 1,
        ACCU_FN::LENGTH - 1u);

    return false;
  }

  return true;
}

void Accuracy_Function_Initializer::Output_Initialize(
    Model *const ptr_Neural_Network_received) const {
  ptr_Neural_Network_received->set_accu_fn(
      this->type_accuracy_function);
}

bool Optimizer_Function_Initializer::Input_Initialize(void) {
  INFO(L"");
  INFO(L"Optimizer functions:");
  for (unsigned int tmp_optimizer_function_index(1u);
       tmp_optimizer_function_index != OPTIMIZER::LENGTH;
       ++tmp_optimizer_function_index) {
    INFO(L"[%d]: %ls.",
           tmp_optimizer_function_index,
           OPTIMIZER_NAME[static_cast<OPTIMIZER::TYPE>(
                                  tmp_optimizer_function_index)]
               .c_str());
  }
  INFO(L"default=%ls.",
         OPTIMIZER_NAME[OPTIMIZER::AMSGRAD].c_str());

  if ((this->type_optimizer_function = static_cast<OPTIMIZER::TYPE>(
           parse_discrete<int>(
               1, OPTIMIZER::LENGTH - 1, L"Choose: "))) >=
      OPTIMIZER::LENGTH) {
    ERR(
        L"An error has been triggered from the "
        "\"parse_discrete<int>(%u, %u)\" function.", 1,
        OPTIMIZER::LENGTH - 1u);

    return false;
  }

  switch (this->type_optimizer_function) {
    case OPTIMIZER::GD:
      INFO(L"");
      INFO(L"Learning rate.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=0.01.");
      this->values[0] = parse_real(0_r, L"Learning rate: ");

      INFO(L"");
      INFO(L"Learning momentum.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=0.9");
      this->values[1] =
          parse_real(0_r, L"Learning momentum: ");

      if (this->values[1] != 0_r) {
        INFO(L"");
        INFO(L"Use Nesterov.");
        INFO(L"default=Yes");
        this->values[2] =
            static_cast<real>(accept(L"Do you want to use Nesterov?"));
        break;
      }
      break;
    case OPTIMIZER::IRPROP_PLUS:
      INFO(L"");
      INFO(L"Increase factor.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=1.2.");
      this->values[0] =
          parse_real(0_r, L"Increase factor: ");

      INFO(L"");
      INFO(L"Decrease factor.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=0.5.");
      this->values[1] =
          parse_real(0_r, L"Decrease factor: ");

      INFO(L"");
      INFO(L"Delta maximum.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=50.0.");
      this->values[2] = parse_real(0_r, L"Delta maximum: ");

      INFO(L"");
      INFO(L"Delta minimum.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=1e-6.");
      this->values[3] = parse_real(0_r, L"Delta minimum: ");

      INFO(L"");
      INFO(L"Delta zero.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=0.1.");
      this->values[4] = parse_real(0_r, L"Delta zero: ");
      break;
    case OPTIMIZER::ADAM:
    case OPTIMIZER::AMSGRAD:
      INFO(L"");
      INFO(L"Learning rate.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=0.001.");
      this->values[0] = parse_real(0_r, L"Learning rate: ");

      INFO(L"");
      INFO(L"Beta1.");
      INFO(L"Range[0.0, 0.99...9].");
      INFO(L"default=0.9.");
      this->values[1] =
          parse_real(0_r, 1_r - 1e-7_r, L"Beta1: ");

      INFO(L"");
      INFO(L"Beta2.");
      INFO(L"Range[0.0, 0.99...9].");
      INFO(L"default=0.999.");
      this->values[2] =
          parse_real(0_r, 1_r - 1e-7_r, L"Beta2: ");

      INFO(L"");
      INFO(L"Epsilon.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=1e-8.");
      this->values[3] = parse_real(0_r, L"Epsilon: ");

      INFO(L"");
      INFO(L"Bias correction.");
      INFO(L"default=true.");
      this->values[4] = accept(L"Bias correction: ");
      break;
    case OPTIMIZER::NOSADAM:
      INFO(L"");
      INFO(L"Learning rate.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=0.001.");
      this->values[0] = parse_real(0_r, L"Learning rate: ");

      INFO(L"");
      INFO(L"Beta1.");
      INFO(L"Range[0.0, 0.99...9].");
      INFO(L"default=0.9.");
      this->values[1] =
          parse_real(0_r, 1_r - 1e-7_r, L"Beta1: ");

      INFO(L"");
      INFO(L"Beta2.");
      INFO(L"Range[0.0, 0.99...9].");
      INFO(L"default=0.999.");
      this->values[2] =
          parse_real(0_r, 1_r - 1e-7_r, L"Beta2: ");

      INFO(L"");
      INFO(L"Epsilon.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=1e-8.");
      this->values[3] = parse_real(0_r, L"Epsilon: ");

      INFO(L"");
      INFO(L"Bias correction.");
      INFO(L"default=true.");
      this->values[4] = accept(L"Bias correction: ");

      INFO(L"");
      INFO(L"Gamma.");
      INFO(L"Range[1e-7, inf].");
      INFO(L"default=0.1.");
      this->values[5] = parse_real<real>(1e-7_r, L"Gamma: ");
      break;
    case OPTIMIZER::ADABOUND:
    case OPTIMIZER::AMSBOUND:
      INFO(L"");
      INFO(L"Learning rate.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=0.001.");
      this->values[0] = parse_real(0_r, L"Learning rate: ");

      INFO(L"");
      INFO(L"Learning rate, final.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=0.1.");
      this->values[1] = parse_real<real>(
          this->values[0], L"Learning rate, final: ");

      INFO(L"");
      INFO(L"Beta1.");
      INFO(L"Range[0.0, 0.99...9].");
      INFO(L"default=0.9.");
      this->values[2] =
          parse_real(0_r, 1_r - 1e-7_r, L"Beta1: ");

      INFO(L"");
      INFO(L"Beta2.");
      INFO(L"Range[0.0, 0.99...9].");
      INFO(L"default=0.999.");
      this->values[3] =
          parse_real(0_r, 1_r - 1e-7_r, L"Beta2: ");

      INFO(L"");
      INFO(L"Epsilon.");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=1e-8.");
      this->values[4] = parse_real(0_r, L"Epsilon: ");

      INFO(L"");
      INFO(L"Bias correction.");
      INFO(L"default=true.");
      this->values[5] = accept(L"Bias correction: ");

      INFO(L"");
      INFO(L"Gamma.");
      INFO(L"Range[0.0, 0.99...9].");
      INFO(L"default=1e-3.");
      this->values[6] =
          parse_real(0_r, 1_r - 1e-7_r, L"Gamma: ");
      break;
    default:
      ERR(
          L"Type optimizer function (%u | %ls) is not managed in "
          "the switch.",
          this->type_optimizer_function,
          OPTIMIZER_NAME[this->type_optimizer_function].c_str());
      return false;
  }

  if (this->type_optimizer_function != OPTIMIZER::IRPROP_PLUS &&
      this->type_optimizer_function != OPTIMIZER::IRPROP_MINUS) {
    INFO(L"");
    INFO(L"Weight decay:");
    INFO(L"Range[0.0, 1.0]. Off = 0.");
    INFO(L"default=1e-5.");
    this->weight_decay =
        parse_real(0_r, 1_r, L"Weight decay: ");
  }

  return true;
}

bool Optimizer_Function_Initializer::Output_Initialize(
    Model *const ptr_Neural_Network_received) const {
  ptr_Neural_Network_received->set_optimizer(
      this->type_optimizer_function);

  switch (this->type_optimizer_function) {
    case OPTIMIZER::GD:
      ptr_Neural_Network_received->learning_rate = this->values[0];

      ptr_Neural_Network_received->learning_momentum = this->values[1];

      if (ptr_Neural_Network_received->learning_momentum != 0_r &&
          ptr_Neural_Network_received
                  ->Allocate__Parameter__Gradient_Descent() == false) {
        ERR(
            L"An error has been triggered from the "
            "\"Allocate__Parameter__Gradient_Descent()\" function.",);

        return false;
      }

      ptr_Neural_Network_received->use_nesterov = this->values[2] != 0_r;
      break;
    case OPTIMIZER::IRPROP_MINUS:
    case OPTIMIZER::IRPROP_PLUS:
      ptr_Neural_Network_received->rprop_increase_factor = this->values[0];

      ptr_Neural_Network_received->rprop_decrease_factor = this->values[1];

      ptr_Neural_Network_received->rprop_delta_max = this->values[2];

      ptr_Neural_Network_received->rprop_delta_min = this->values[3];

      ptr_Neural_Network_received->rprop_delta_zero = this->values[4];
      break;
    case OPTIMIZER::ADAM:
    case OPTIMIZER::ADAMAX:
    case OPTIMIZER::AMSGRAD:
      ptr_Neural_Network_received->adam_learning_rate = this->values[0];

      ptr_Neural_Network_received->adam_beta1 = this->values[1];

      ptr_Neural_Network_received->adam_beta2 = this->values[2];

      ptr_Neural_Network_received->adam_epsilon = this->values[3];

      ptr_Neural_Network_received->use_adam_bias_correction =
          this->values[4] != 0_r;
      break;
    case OPTIMIZER::NOSADAM:
      ptr_Neural_Network_received->adam_learning_rate = this->values[0];

      ptr_Neural_Network_received->adam_beta1 = this->values[1];

      ptr_Neural_Network_received->adam_beta2 = this->values[2];

      ptr_Neural_Network_received->adam_epsilon = this->values[3];

      ptr_Neural_Network_received->use_adam_bias_correction =
          this->values[4] != 0_r;

      ptr_Neural_Network_received->adam_gamma = this->values[5];
      break;
    case OPTIMIZER::ADABOUND:
    case OPTIMIZER::AMSBOUND:
      ptr_Neural_Network_received->adam_learning_rate = this->values[0];

      ptr_Neural_Network_received->learning_rate_final = this->values[1];

      ptr_Neural_Network_received->adam_beta1 = this->values[2];

      ptr_Neural_Network_received->adam_beta2 = this->values[3];

      ptr_Neural_Network_received->adam_epsilon = this->values[4];

      ptr_Neural_Network_received->use_adam_bias_correction =
          this->values[5] != 0_r;

      ptr_Neural_Network_received->learning_gamma = this->values[6];
      break;
    default:
      ERR(
          L"Type optimizer function (%u | %ls) is not managed in "
          "the switch.",
          this->type_optimizer_function,
          OPTIMIZER_NAME[this->type_optimizer_function].c_str());
      return false;
  }

  if (ptr_Neural_Network_received->Set__Regularization__Weight_Decay(
          this->weight_decay) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Set__Regularization__Weight_Decay(%f)\" function.",
        this->weight_decay);

    return false;
  }

#ifdef COMPILE_CUDA
  if (ptr_Neural_Network_received->is_cu_initialized) {
    ptr_Neural_Network_received->cumodel->Copy__Optimizer_Parameters(
        ptr_Neural_Network_received);
  }
#endif

  return true;
}

void Warm_Restarts_Initializer::Input_Initialize(void) {
  INFO(L"");
  INFO(L"Warm restarts:");
  INFO(L"default=Yes.");
  this->use_warm_restarts = accept(L"Use warm restarts: ");

  if (this->use_warm_restarts) {
    INFO(L"");
    INFO(L"Learning rate, decay:");
    INFO(L"Range[1e-5, 1.0].");
    INFO(L"default=0.95.");
    this->warm_restarts_decay_learning_rate =
        parse_real<real>(1e-5_r, 1_r, L"Learning rate, decay: ");

    INFO(L"");
    INFO(L"Maximum learning rate:");
    INFO(L"Range[0.0, 1.0].");
    INFO(L"default=1.");
    this->warm_restarts_maximum_learning_rate =
        parse_real(0_r, 1_r, L"Maximum learning rate: ");

    INFO(L"");
    INFO(L"Minimum learning rate:");
    INFO(L"Range[0.0, %f].",
           this->warm_restarts_maximum_learning_rate);
    INFO(L"default=0.");
    this->warm_restarts_minimum_learning_rate = parse_real<real>(
        0_r, this->warm_restarts_maximum_learning_rate,
        L"Minimum learning rate: ");
    if (this->warm_restarts_minimum_learning_rate == 0_r) {
      this->warm_restarts_minimum_learning_rate =
          this->warm_restarts_maximum_learning_rate / 1e+7_r;
    }

    INFO(L"");
    INFO(L"Initial Ti:");
    INFO(L"Range[0, inf].");
    INFO(L"default=1.");
    this->warm_restarts_initial_T_i =
        static_cast<real>(parse_discrete<size_t>(0_UZ, L"Initial Ti: "));

    INFO(L"");
    INFO(L"Warm restarts multiplier:");
    INFO(L"Range[0, inf].");
    INFO(L"default=2.");
    this->warm_restarts_multiplier = static_cast<real>(
        parse_discrete<size_t>(0_UZ, L"Warm restarts multiplier: "));
  }
}

bool Warm_Restarts_Initializer::Output_Initialize(
    Model *const ptr_Neural_Network_received) const {
  ptr_Neural_Network_received->use_warm_restarts = this->use_warm_restarts;

  if (this->use_warm_restarts) {
    ptr_Neural_Network_received->warm_restarts_decay_learning_rate =
        this->warm_restarts_decay_learning_rate;
    ptr_Neural_Network_received->warm_restarts_maximum_learning_rate =
        ptr_Neural_Network_received
            ->warm_restarts_initial_maximum_learning_rate =
            this->warm_restarts_maximum_learning_rate;
    ptr_Neural_Network_received->warm_restarts_minimum_learning_rate =
        this->warm_restarts_minimum_learning_rate;
    ptr_Neural_Network_received->warm_restarts_T_i =
        ptr_Neural_Network_received->warm_restarts_initial_T_i =
            this->warm_restarts_initial_T_i;
    ptr_Neural_Network_received->warm_restarts_multiplier =
        this->warm_restarts_multiplier;
  }

#ifdef COMPILE_CUDA
  if (ptr_Neural_Network_received->is_cu_initialized) {
    ptr_Neural_Network_received->cumodel->Copy__Warm_Restarts_Parameters(
        ptr_Neural_Network_received);
  }
#endif

  return true;
}

bool Weights_Initializer::Input_Initialize(void) {
  INFO(L"");
  INFO(L"Weights initializer:");
  for (unsigned int tmp_weights_initializer_type_index(1u);
       tmp_weights_initializer_type_index != INITIALIZER::LENGTH;
       ++tmp_weights_initializer_type_index) {
    INFO(L"[%d]: %ls.",
           tmp_weights_initializer_type_index,
           INITIALIZER_NAME[static_cast<INITIALIZER::TYPE>(
                                    tmp_weights_initializer_type_index)]
               .c_str());
  }
  INFO(L"default=%ls.",
         INITIALIZER_NAME[INITIALIZER::ORTHOGONAL].c_str());

  if ((this->type_weights_initializer = static_cast<INITIALIZER::TYPE>(
           parse_discrete<int>(
               1, INITIALIZER::LENGTH - 1, L"Type: "))) >=
      INITIALIZER::LENGTH) {
    ERR(
        L"An error has been triggered from the "
        "\"parse_discrete<int>(%u, %u)\" function.", 1,
        INITIALIZER::LENGTH - 1u);

    return false;
  }

  switch (this->type_weights_initializer) {
    case INITIALIZER::GLOROT_GAUSSIAN:
    case INITIALIZER::GLOROT_UNIFORM:
    case INITIALIZER::IDENTITY:
    case INITIALIZER::ORTHOGONAL:
      INFO(L"");
      INFO(L"Initial bias:");
      INFO(L"Range[0.0, 1.0].");
      INFO(L"default=0.0.");

      this->initial_bias =
          parse_real(0_r, 1_r, L"Initial bias: ");
      break;
    case INITIALIZER::LSUV:
      INFO(L"");
      INFO(L"Initial bias:");
      INFO(L"Range[0.0, 1.0].");
      INFO(L"default=0.0.");

      this->initial_bias =
          parse_real(0_r, 1_r, L"Initial bias: ");

      INFO(L"");
      INFO(L"Maximum number of tials:");
      INFO(L"Range[0, inf].");
      INFO(L"default=10.");

      this->values[0] = static_cast<real>(
          parse_discrete<size_t>(0_UZ, L"Maximum number trials: "));

      INFO(L"");
      INFO(L"Maximum batch size:");
      INFO(L"Range[1, inf].");
      INFO(L"default=32.");

      this->values[1] = static_cast<real>(
          parse_discrete<size_t>(1_UZ, L"Maximum batch size: "));

      INFO(L"");
      INFO(L"Variance target:");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=1.0.");

      this->values[2] =
          parse_real(0_r, L"Variance target: ");

      INFO(L"");
      INFO(L"Variance tolerance:");
      INFO(L"Range[0.0, inf].");
      INFO(L"default=0.01.");

      this->values[3] =
          parse_real(0_r, L"Variance tolerance: ");
      break;
    case INITIALIZER::UNIFORM:
      INFO(L"");
      INFO(L"Initial bias:");
      INFO(L"Range[0.0, 1.0].");
      INFO(L"default=0.0.");

      this->initial_bias =
          parse_real(0_r, 1_r, L"Initial bias: ");

      INFO(L"");
      INFO(L"Lower bound:");
      INFO(L"Range[-1.0, 1.0].");
      INFO(L"default=-1.0.");

      this->values[0] =
          parse_real<real>(-1_r, 1_r, L"Lower bound: ");

      INFO(L"");
      INFO(L"Upper bound:");
      INFO(L"Range[%f, 1.0].",
             this->values[0]);
      INFO(L"default=1.0.");

      this->values[1] = parse_real<real>(this->values[0], 1_r,
                                                        L"Upper bound: ");
      break;
    default:
      ERR(
          L"Type weights initializer (%u | %ls) is not managed in "
          "the switch.",
          this->type_weights_initializer,
          INITIALIZER_NAME[this->type_weights_initializer].c_str());
      return false;
  }

  return true;
}

bool Weights_Initializer::Output_Initialize(
    Model *const ptr_Neural_Network_received) const {
  switch (this->type_weights_initializer) {
    case INITIALIZER::GLOROT_GAUSSIAN:
      ptr_Neural_Network_received->Initialization__Glorot__Gaussian(
          this->initial_bias);
      break;
    case INITIALIZER::GLOROT_UNIFORM:
      ptr_Neural_Network_received->initialize_weights_with_glorot_uniform(
          this->initial_bias);
      break;
    case INITIALIZER::IDENTITY:
      ptr_Neural_Network_received->Initialization__Identity(this->initial_bias);
      break;
    case INITIALIZER::LSUV:
      if (ptr_Neural_Network_received->Initialize__LSUV(
              static_cast<size_t>(this->values[0]),
              static_cast<size_t>(this->values[1]), this->initial_bias,
              this->values[2], this->values[3]) == false) {
        ERR(
            L"An error has been triggered from the "
            "\"Initialize__LSUV(%zu, %zu, %f, %f, %f)\" function.",
            static_cast<size_t>(this->values[0]),
            static_cast<size_t>(this->values[1]), this->initial_bias,
            this->values[2], this->values[3]);

        return false;
      }
      break;
    case INITIALIZER::ORTHOGONAL:
      ptr_Neural_Network_received->layers_initialize_orthogonal(
          false, this->initial_bias);
      break;
    case INITIALIZER::UNIFORM:
      ptr_Neural_Network_received->Initialization__Uniform(
          this->initial_bias, this->values[0], this->values[1]);
      break;
    default:
      ERR(
          L"Type weights initializer (%u | %ls) is not managed in "
          "the switch.",
          this->type_weights_initializer,
          INITIALIZER_NAME[this->type_weights_initializer].c_str());
      return false;
  }

  #ifdef COMPILE_CUDA
  if (ptr_Neural_Network_received->is_cu_initialized)
    ptr_Neural_Network_received->Copy__Parameters__Host_To_Device();
  #endif

  return true;
}

bool Dropout_Initializer::Input_Initialize(
    size_t const n_layers,
    MODEL::TYPE const type_network_received) {
  bool tmp_use_dropout(false);

  unsigned int tmp_type_dropout_layer_index;

  size_t const tmp_option_end(type_network_received == MODEL::AUTOENCODER
                                  ? n_layers / 2_UZ + 1_UZ
                                  : n_layers - 1_UZ);
  size_t tmp_option, tmp_layer_index;

  std::wstring tmp_layer_name;

  if (this->Allocate__Layers_Using_Dropout(n_layers - 1_UZ) ==
      false)  // Subtract output layer.
  {
    ERR(
        L"An error has been triggered from the "
        "\"Allocate__Layers_Using_Dropout(%zu)\" function.",
        n_layers - 1_UZ);

    return false;
  }

  while (true) {
    INFO(L"");
    INFO(L"Dropout initializer:");
    INFO(L"[0]: Input layer: (%f, %f), %ls.",
           this->ptr_array_layers_dropout_array_values[0][0],
           this->ptr_array_layers_dropout_array_values[0][1],
           LAYER_DROPOUT_NAME[this->ptr_array_layers_type_dropout[0]]
               .c_str());
    for (tmp_layer_index = 1_UZ; tmp_layer_index != tmp_option_end;
         ++tmp_layer_index) {
      INFO(
          L"[%zu]: Hidden layer[%zu]: (%f, %f), %ls.", tmp_layer_index,
          tmp_layer_index - 1_UZ,
          this->ptr_array_layers_dropout_array_values[tmp_layer_index][0],
          this->ptr_array_layers_dropout_array_values[tmp_layer_index][1],
          LAYER_DROPOUT_NAME
              [this->ptr_array_layers_type_dropout[tmp_layer_index]]
                  .c_str());
    }
    INFO(L"[%zu]: Quit.",
           tmp_option_end);

    tmp_option =
        parse_discrete<size_t>(0_UZ, tmp_option_end, L"Option: ");

    if (tmp_option < tmp_option_end) {
      tmp_layer_name =
          tmp_option == 0_UZ
              ? L"Input"
              : L"Hidden[" + std::to_wstring(tmp_option - 1_UZ) + L"]";

      INFO(L"");
      INFO(L"Dropout layer:");
      for (tmp_type_dropout_layer_index = 0u;
           tmp_type_dropout_layer_index != LAYER_DROPOUT::LENGTH;
           ++tmp_type_dropout_layer_index) {
        INFO(L"[%d]: %ls.",
               tmp_type_dropout_layer_index,
               LAYER_DROPOUT_NAME[static_cast<LAYER_DROPOUT::TYPE>(
                                          tmp_type_dropout_layer_index)]
                   .c_str());
      }
      INFO(L"default=%ls.",
             LAYER_DROPOUT_NAME[LAYER_DROPOUT::BERNOULLI].c_str());

      switch ((this->ptr_array_layers_type_dropout[tmp_option] =
                   static_cast<LAYER_DROPOUT::TYPE>(
                       parse_discrete<size_t>(
                           0_UZ, LAYER_DROPOUT::LENGTH - 1,
                           (tmp_layer_name + L" layer, type: ").c_str())))) {
        case LAYER_DROPOUT::NONE:
          this->ptr_array_layers_dropout_array_values[tmp_option][0] = 0_r;
          this->ptr_array_layers_dropout_array_values[tmp_option][1] = 0_r;
          break;
        case LAYER_DROPOUT::ALPHA:
          INFO(L"");
          INFO(L"Alpha dropout: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          this->ptr_array_layers_dropout_array_values[tmp_option][0] =
              parse_real<real>(
                  0_r, 1_r,
                  (tmp_layer_name + L" layer, dropout probability: ").c_str());

          this->ptr_array_layers_dropout_array_values[tmp_option][1] = 0_r;

          if (this->ptr_array_layers_dropout_array_values[tmp_option][0] !=
              0_r) {
            tmp_use_dropout = true;
          }
          break;
        case LAYER_DROPOUT::BERNOULLI:
          INFO(L"");
          INFO(L"Dropout bernoulli: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          this->ptr_array_layers_dropout_array_values[tmp_option][0] =
              parse_real<real>(
                  0_r, 1_r,
                  (tmp_layer_name + L" layer, retention probability: ").c_str());

          this->ptr_array_layers_dropout_array_values[tmp_option][1] = 0_r;

          if (this->ptr_array_layers_dropout_array_values[tmp_option][0] !=
              1_r) {
            tmp_use_dropout = true;
          }
          break;
        case LAYER_DROPOUT::BERNOULLI_INVERTED:
          INFO(L"");
          INFO(L"Dropout bernoulli inverted: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          this->ptr_array_layers_dropout_array_values[tmp_option][0] =
              parse_real<real>(
                  0_r, 1_r,
                  (tmp_layer_name + L" layer, retention probability: ").c_str());

          this->ptr_array_layers_dropout_array_values[tmp_option][1] = 0_r;

          if (this->ptr_array_layers_dropout_array_values[tmp_option][0] !=
              1_r) {
            tmp_use_dropout = true;
          }
          break;
        case LAYER_DROPOUT::GAUSSIAN:
          INFO(L"");
          INFO(L"Dropout gaussian: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          this->ptr_array_layers_dropout_array_values[tmp_option][0] =
              parse_real<real>(
                  0_r, 1_r,
                  (tmp_layer_name + L" layer, dropout probability: ").c_str());

          this->ptr_array_layers_dropout_array_values[tmp_option][1] = 0_r;

          if (this->ptr_array_layers_dropout_array_values[tmp_option][0] !=
              0_r) {
            tmp_use_dropout = true;
          }
          break;
        case LAYER_DROPOUT::SHAKEDROP:
          INFO(L"");
          INFO(L"Dropout ShakeDrop: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          this->ptr_array_layers_dropout_array_values[tmp_option][0] =
              parse_real<real>(
                  0_r, 1_r,
                  (tmp_layer_name + L" layer, dropout probability: ").c_str());

          this->ptr_array_layers_dropout_array_values[tmp_option][1] = 0_r;

          if (this->ptr_array_layers_dropout_array_values[tmp_option][0] !=
              0_r) {
            tmp_use_dropout = true;
          }
          break;
        case LAYER_DROPOUT::UOUT:
          INFO(L"");
          INFO(L"Dropout Uout: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          this->ptr_array_layers_dropout_array_values[tmp_option][0] =
              parse_real<real>(
                  0_r, 1_r,
                  (tmp_layer_name + L" layer, dropout probability: ").c_str());

          this->ptr_array_layers_dropout_array_values[tmp_option][1] = 0_r;

          if (this->ptr_array_layers_dropout_array_values[tmp_option][0] !=
              0_r) {
            tmp_use_dropout = true;
          }
          break;
        case LAYER_DROPOUT::ZONEOUT:
          INFO(L"");
          INFO(L"Zoneout cell: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          this->ptr_array_layers_dropout_array_values[tmp_option][0] =
              parse_real<real>(
                  0_r, 1_r,
                  (tmp_layer_name + L" layer, zoneout cell probability: ")
                      .c_str());

          INFO(L"");
          INFO(L"Zoneout hidden: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.05.");

          this->ptr_array_layers_dropout_array_values[tmp_option][1] =
              parse_real<real>(
                  0_r, 1_r,
                  (tmp_layer_name + L" layer, zoneout hidden probability: ")
                      .c_str());

          if (this->ptr_array_layers_dropout_array_values[tmp_option][0] !=
                  0_r ||
              this->ptr_array_layers_dropout_array_values[tmp_option][1] !=
                  0_r) {
            tmp_use_dropout = true;
          }
          break;
        default:
          ERR(
              L"Type dropout layer (%u | %ls) is not managed in "
              "the switch.",
              this->ptr_array_layers_type_dropout[tmp_option],
              LAYER_DROPOUT_NAME
                  [this->ptr_array_layers_type_dropout[tmp_option]]
                      .c_str());
          return false;
      }

      if (type_network_received == MODEL::AUTOENCODER &&
          this->ptr_array_layers_type_dropout[tmp_option] !=
              LAYER_DROPOUT::NONE &&
          tmp_option != 0_UZ) {
        this->ptr_array_layers_use_coded_dropout[tmp_option] = accept(
            L"Pre-training: Use dropout inside the coded layer?");
      }
    } else if (tmp_option == tmp_option_end) {
      return true;
    } else {
      ERR(
          L"An error has been triggered from the "
          "\"parse_discrete<size_t>(%zu, %zu)\" function.", 0_UZ, tmp_option_end);
    }
  }

  if (tmp_use_dropout == false) {
    this->Deallocate__Layers_Using_Dropout();
  }

  return true;
}

bool Dropout_Initializer::Output_Initialize(
    Model *const ptr_Neural_Network_received) const {
  if (this->ptr_array_layers_dropout_array_values != nullptr &&
      this->ptr_array_layers_type_dropout != nullptr) {
    size_t const tmp_number_layers(std::min<size_t>(
        this->number_layers, ptr_Neural_Network_received->total_layers -
                                 1_UZ));  // Subtract output layer.
    size_t tmp_layer_index;

    for (tmp_layer_index = 0_UZ; tmp_layer_index != tmp_number_layers;
         ++tmp_layer_index) {
      if (ptr_Neural_Network_received->set_dropout(
              tmp_layer_index,
              this->ptr_array_layers_type_dropout[tmp_layer_index],
              this->ptr_array_layers_dropout_array_values[tmp_layer_index]) ==
          false) {
        ERR(
            L"An error has been triggered from the "
            "\"set_dropout(%zu, %u, %f, %f)\" function.", tmp_layer_index,
            this->ptr_array_layers_type_dropout[tmp_layer_index],
            this->ptr_array_layers_dropout_array_values[tmp_layer_index]
                                                              [0],
            this->ptr_array_layers_dropout_array_values[tmp_layer_index]
                                                              [1]);

        return false;
      }

      ptr_Neural_Network_received->ptr_array_layers[tmp_layer_index]
          .use_coded_dropout =
          this->ptr_array_layers_use_coded_dropout[tmp_layer_index];
    }
  }

  return true;
}

void Dropout_Initializer::Deallocate__Layers_Using_Dropout(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_layers_use_coded_dropout);

  if (this->ptr_array_layers_dropout_array_values != nullptr) {
    SAFE_DELETE_ARRAY(this->ptr_array_layers_dropout_array_values[0]);
    SAFE_DELETE_ARRAY(this->ptr_array_layers_dropout_array_values);
  }

  SAFE_DELETE_ARRAY(this->ptr_array_layers_type_dropout);
}

bool Dropout_Initializer::Allocate__Layers_Using_Dropout(
    size_t const n_layers) {
  bool *tmp_ptr_array_layers_use_coded_dropout(
      new bool[n_layers]);
  Mem::fill(
      tmp_ptr_array_layers_use_coded_dropout,
      tmp_ptr_array_layers_use_coded_dropout + n_layers, false);
  this->ptr_array_layers_use_coded_dropout =
      tmp_ptr_array_layers_use_coded_dropout;

  // Dropout value.
  real **tmp_ptr_array_layers_dropout_array_values(new real *[n_layers]);
  Mem::fill_null<real *>(
      tmp_ptr_array_layers_dropout_array_values,
      tmp_ptr_array_layers_dropout_array_values + n_layers);
  this->ptr_array_layers_dropout_array_values =
      tmp_ptr_array_layers_dropout_array_values;

  real *tmp_ptr_array_values(new real[n_layers * 2_UZ]());

  for (size_t tmp_index(0_UZ); tmp_index != n_layers;
       ++tmp_index) {
    this->ptr_array_layers_dropout_array_values[tmp_index] =
        tmp_ptr_array_values + tmp_index * 2_UZ;
  }
  // |END| Dropout value. |END|

  LAYER_DROPOUT::TYPE *tmp_ptr_array_layers_type_dropout(
      new LAYER_DROPOUT::TYPE[n_layers]);
  Mem::fill(
      tmp_ptr_array_layers_type_dropout,
      tmp_ptr_array_layers_type_dropout + n_layers,
      LAYER_DROPOUT::NONE);
  this->ptr_array_layers_type_dropout = tmp_ptr_array_layers_type_dropout;

  this->number_layers = n_layers;

  return true;
}

Dropout_Initializer::~Dropout_Initializer(void) {
  this->Deallocate__Layers_Using_Dropout();
}

bool Normalization_Initializer::Input_Initialize(
    size_t const n_layers, size_t const number_batch_received,
    MODEL::TYPE const type_network_received) {
  bool tmp_use_batch_normalization(false), tmp_use_batch_renormalization(false);

  unsigned int tmp_type_normalization_layer_index;

  size_t const tmp_option_end(type_network_received == MODEL::AUTOENCODER
                                  ? n_layers / 2_UZ + 1_UZ
                                  : n_layers - 1_UZ);
  size_t tmp_option, tmp_layer_index;

  if (this->Allocate__Layers_Using_Normalization(
          n_layers - 1_UZ) == false)  // Subtract output layer.
  {
    ERR(
        L"An error has been triggered from the "
        "\"Allocate__Layers_Using_Normalization(%zu)\" function.",
        n_layers - 1_UZ);

    return false;
  }

  this->ptr_array_layers_using_normalization[0] = LAYER_NORM::NONE;

  while (true) {
    INFO(L"");
    INFO(L"Normalization initializer:");
    for (tmp_layer_index = 1_UZ; tmp_layer_index != tmp_option_end;
         ++tmp_layer_index) {
      INFO(L"[%zu] Hidden layer[%zu]: %ls, %ls.", tmp_layer_index - 1_UZ,
             tmp_layer_index - 1_UZ,
             LAYER_NORM_NAME
                 [this->ptr_array_layers_using_normalization[tmp_layer_index]]
                     .c_str(),
             this->ptr_array_layers_normalization_before_activation
                     [tmp_layer_index]
                 ? "true"
                 : "false");
    }
    INFO(L"[%zu]: Quit.",
           tmp_option_end - 1_UZ);

    tmp_option = parse_discrete<size_t>(0_UZ, tmp_option_end - 1_UZ,
                                                  L"Option: ") +
                 1_UZ;

    if (tmp_option < tmp_option_end) {
      tmp_layer_index = tmp_option;

      INFO(L"");
      INFO(L"Layer normalization:");
      for (tmp_type_normalization_layer_index = 0u;
           tmp_type_normalization_layer_index != LAYER_NORM::LENGTH;
           ++tmp_type_normalization_layer_index) {
        INFO(L"[%d]: %ls.",
               tmp_type_normalization_layer_index,
               LAYER_NORM_NAME[static_cast<LAYER_NORM::TYPE>(
                                       tmp_type_normalization_layer_index)]
                   .c_str());
      }
      INFO(
          L"default=%ls.",
          LAYER_NORM_NAME[LAYER_NORM::BATCH_RENORMALIZATION].c_str());

      if ((this->ptr_array_layers_using_normalization[tmp_layer_index] =
               static_cast<LAYER_NORM::TYPE>(
                   parse_discrete<int>(
                       0, LAYER_NORM::LENGTH - 1,
                       (L"Hidden layer " + std::to_wstring(tmp_layer_index) +
                        L", type: ")
                           .c_str()))) >= LAYER_NORM::LENGTH) {
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 1,
            LAYER_NORM::LENGTH - 1u);

        return false;
      }

      if (this->ptr_array_layers_using_normalization[tmp_layer_index] !=
          LAYER_NORM::NONE) {
        this->ptr_array_layers_normalization_before_activation
            [tmp_layer_index] = accept(
            (L"Hidden layer " + std::to_wstring(tmp_layer_index) +
             L", use normalization before activation?")
                .c_str());
      }

      switch (this->ptr_array_layers_using_normalization[tmp_layer_index]) {
        case LAYER_NORM::BATCH_NORMALIZATION:
          tmp_use_batch_normalization = true;
          break;
        case LAYER_NORM::BATCH_RENORMALIZATION:
          tmp_use_batch_renormalization = true;
          break;
        default:
          break;
      }
    } else if (tmp_option == tmp_option_end) {
      return true;
    } else {
      ERR(
          L"An error has been triggered from the "
          "\"parse_discrete<size_t>(%zu, %zu)\" function.", 0_UZ,
          tmp_option_end - 1_UZ);
    }
  }

  if (tmp_use_batch_normalization || tmp_use_batch_renormalization) {
    // Normalization parameter.
    INFO(L"");
    INFO(L"Momentum average:");
    INFO(L"Range[0.0, 1.0].");
    INFO(L"default=%.9f.",
           number_batch_received <= 1_UZ
                      ? 0.999_r
                      : 1_r / static_cast<real>(number_batch_received));

    this->normalization_momentum_average =
        parse_real(0_r, 1_r, L"Momentum average: ");

    INFO(L"");
    INFO(L"Epsilon:");
    INFO(L"Range[0.0, inf].");
    INFO(L"default=1e-5.");

    this->normalization_epsilon =
        parse_real(0_r, L"Epsilon: ");
    // |END| Normalization parameter. |END|
  }

  if (tmp_use_batch_renormalization) {
    // Batch renormalization parameter.
    INFO(L"");
    INFO(L"r correction maximum:");
    INFO(L"Range[0.0, inf].");
    INFO(L"default=1.");

    this->batch_renormalization_r_correction_maximum =
        parse_real(0_r, L"r correction maximum: ");

    INFO(L"");
    INFO(L"d correction maximum:");
    INFO(L"Range[0.0, inf].");
    INFO(L"default=0.");

    this->batch_renormalization_d_correction_maximum =
        parse_real(0_r, L"d correction maximum: ");
    // |END| Batch renormalization parameter. |END|
  }

  if (tmp_use_batch_normalization == false &&
      tmp_use_batch_renormalization == false) {
    this->Deallocate__Layers_Using_Normalization();
  }

  return true;
}

bool Normalization_Initializer::Output_Initialize(
    Model *const ptr_Neural_Network_received) const {
  if (this->ptr_array_layers_using_normalization != nullptr) {
    bool tmp_use_normalization(false), tmp_use_renormalization(false);

    size_t const tmp_number_layers(std::min<size_t>(
        this->number_layers, ptr_Neural_Network_received->total_layers -
                                 1_UZ));  // Subtract output layer.
    size_t tmp_layer_index;

    for (tmp_layer_index = 1_UZ; tmp_layer_index != tmp_number_layers;
         ++tmp_layer_index) {
      if (ptr_Neural_Network_received->Set__Layer_Normalization(
              tmp_layer_index,
              this->ptr_array_layers_using_normalization[tmp_layer_index]) ==
          false) {
        ERR(
            L"An error has been triggered from the "
            "\"Set__Layer_Normalization(%zu, %u)\" function.", tmp_layer_index,
            this->ptr_array_layers_using_normalization[tmp_layer_index]);

        return false;
      }

      switch (this->ptr_array_layers_using_normalization[tmp_layer_index]) {
        case LAYER_NORM::BATCH_NORMALIZATION:
          tmp_use_normalization = true;
          break;
        case LAYER_NORM::BATCH_RENORMALIZATION:
          tmp_use_renormalization = true;
          break;
        default:
          break;
      }

      ptr_Neural_Network_received->ptr_array_layers[tmp_layer_index]
          .use_layer_normalization_before_activation =
          this->ptr_array_layers_normalization_before_activation
              [tmp_layer_index];
    }

    if (tmp_use_normalization || tmp_use_renormalization) {
      // Normalization parameter.
      if (ptr_Neural_Network_received->Set__Normalization_Momentum_Average(
              this->normalization_momentum_average) == false) {
        ERR(
            L"An error has been triggered from the "
            "\"Set__Normalization_Momentum_Average(%f)\" function.",
            this->normalization_momentum_average);

        return false;
      }

      if (ptr_Neural_Network_received->Set__Normalization_Epsilon(
              this->normalization_epsilon) == false) {
        ERR(
            L"An error has been triggered from the "
            "\"Set__Normalization_Epsilon(%f)\" function.",
            this->normalization_epsilon);

        return false;
      }
      // |END| Normalization parameter. |END|
    }

    if (tmp_use_renormalization) {
      // Batch renormalization parameter.
      if (ptr_Neural_Network_received
              ->Set__Batch_Renormalization_r_Correction_Maximum(
                  this->batch_renormalization_r_correction_maximum) == false) {
        ERR(
            L"An error has been triggered from the "
            "\"Set__Batch_Renormalization_r_Correction_Maximum(%f)\" function.",
            this->batch_renormalization_r_correction_maximum);

        return false;
      }

      if (ptr_Neural_Network_received
              ->Set__Batch_Renormalization_d_Correction_Maximum(
                  this->batch_renormalization_d_correction_maximum) == false) {
        ERR(
            L"An error has been triggered from the "
            "\"Set__Batch_Renormalization_d_Correction_Maximum(%f)\" function.",
            this->batch_renormalization_d_correction_maximum);

        return false;
      }
      // |END| Batch renormalization parameter. |END|
    }
  }

  return true;
}

void Normalization_Initializer::Deallocate__Layers_Using_Normalization(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_layers_normalization_before_activation);

  SAFE_DELETE_ARRAY(this->ptr_array_layers_using_normalization);
}

bool Normalization_Initializer::Allocate__Layers_Using_Normalization(
    size_t const n_layers) {
  bool *tmp_ptr_array_layers_normalization_before_activation(
      new bool[n_layers]);
  memset(tmp_ptr_array_layers_normalization_before_activation, 0,
         n_layers * sizeof(bool));
  this->ptr_array_layers_normalization_before_activation =
      tmp_ptr_array_layers_normalization_before_activation;

  LAYER_NORM::TYPE *tmp_ptr_array_layers_using_normalization(
      new LAYER_NORM::TYPE[n_layers]);
  memset(tmp_ptr_array_layers_using_normalization, 0,
         n_layers * sizeof(LAYER_NORM::TYPE));
  this->ptr_array_layers_using_normalization =
      tmp_ptr_array_layers_using_normalization;

  this->number_layers = n_layers;

  return true;
}

Normalization_Initializer::~Normalization_Initializer(void) {
  this->Deallocate__Layers_Using_Normalization();
}

bool Model::set_cu(bool const use_cuda, size_t const allowable_memory) {
#ifdef COMPILE_CUDA
  if ((this->use_cu == false && use_cuda) ||
      (this->use_cu && use_cuda && this->is_cu_initialized == false)) {
    if (this->Initialize__CUDA(allowable_memory) == false) {
      ERR(
          L"An error has been triggered from the "
          "\"Initialize__CUDA(%zu)\" function.", allowable_memory);

      return false;
    }
  } else if ((this->use_cu && use_cuda == false) ||
             (this->use_cu == false && use_cuda == false &&
              this->is_cu_initialized)) {
    if (this->Deinitialize__CUDA() == false) {
      ERR(
          L"An error has been triggered from the "
          "\"Deinitialize__CUDA()\" function.",);

      return false;
    }
  }
#else
  if(use_cuda)
    ERR(L"`CUDA` functionality was not built. Pass `-DCOMPILE_CUDA` to the "
        L"compiler.");
#endif

  this->use_cu = use_cuda;

  return true;
}

bool Model::Initialize__CUDA(size_t const allowable_memory) {
#ifdef COMPILE_CUDA
  if (this->is_cu_initialized == false) {
    CUDA__Safe_Call(cudaMalloc((void **)&this->cumodel, sizeof(cuModel)));

    if (this->cumodel->Copy__Host_To_Device(this, allowable_memory) == false) {
      ERR(
          L"An error has been triggered from the "
          "\"Copy__Host_To_Device(ptr, %zu)\" function.", allowable_memory);

      CUDA__Safe_Call(cudaFree(this->cumodel));

      return false;
    }

    this->is_cu_initialized = true;
  }
#else
  ERR(L"`CUDA` functionality was not built. Pass `-DCOMPILE_CUDA` to the "
      L"compiler.");
#endif

  return true;
}

bool Model::Initialize__CUDA__Thread(Datasets const *const datasets) {
#ifdef COMPILE_CUDA
    Datasets const *const datasets) {
  if (this->is_cu_initialized == false) {
    ERR(L"Device not initialized.",);

    return false;
  }

  size_t const tmp_number_examples_training(
      datasets->get_dataset(ENV::TRAIN)->get_n_data()),
      tmp_number_examples_validation(
          datasets->get_dataset(ENV::VALID)->get_n_data()),
      tmp_number_examples_testing(
          datasets->get_dataset(ENV::TESTG)->get_n_data());
  size_t tmp_number_examples_max(0_UZ);

  tmp_number_examples_max =
      std::max<size_t>(tmp_number_examples_max, tmp_number_examples_training);
  tmp_number_examples_max =
      std::max<size_t>(tmp_number_examples_max, tmp_number_examples_validation);
  tmp_number_examples_max =
      std::max<size_t>(tmp_number_examples_max, tmp_number_examples_testing);

  INFO(L"GPU: Neural network: Update threads size");
  if (this->cumodel->update_mem_thread_size(tmp_number_examples_max) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"update_mem_thread_size(%zu)\" function.", tmp_number_examples_max);

    if (this->Deinitialize__CUDA() == false) {
      ERR(
          L"An error has been triggered from the "
          "\"Deinitialize__CUDA()\" function.",);

      return false;
    }

    return false;
  }

  INFO(L"GPU: Neural network: Update batch size");
  if (this->cumodel->update_mem_batch_size(tmp_number_examples_max) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"update_mem_batch_size(%zu)\" function.", tmp_number_examples_max);

    if (this->Deinitialize__CUDA() == false) {
      ERR(
          L"An error has been triggered from the "
          "\"Deinitialize__CUDA()\" function.",);

      return false;
    }

    return false;
  }

  INFO(
      L"GPU: Neural network: Setting up limit device runtime pending launch "
      "count.");
  this->cumodel->Set__Limit_Device_Runtime_Pending_Launch_Count();
#else
  ERR(L"`CUDA` functionality was not built. Pass `-DCOMPILE_CUDA` to the compiler.");
#endif

  return true;
}

bool Model::Deinitialize__CUDA(void) {
#ifdef COMPILE_CUDA
  if (this->is_cu_initialized) {
    if (this->cumodel->Deallocate() == false) {
      ERR(
          L"An error has been triggered from the "
          "\"Deallocate()\" function.",);

      CUDA__Safe_Call(cudaFree(this->cumodel));

      return false;
    }

    CUDA__Safe_Call(cudaFree(this->cumodel));

    this->is_cu_initialized = false;
  }
#else
  ERR(L"`CUDA` functionality was not built. Pass `-DCOMPILE_CUDA` to the "
      L"compiler.");
#endif

  return true;
}

bool Model::Use__CUDA(void) const { return (this->use_cu); }

bool Model::Use__Clip_Gradient(void) const { return (this->use_clip_gradient); }

bool Model::Use__Regularization_Parameter(void) const {
  if (this->regularization__l1 != 0_r || this->regularization__l2 != 0_r ||
      this->regularization__srip != 0_r ||
      this->weight_decay != 0_r) {
    return true;
  }

  return false;
}

bool Model::Use__Normalization(void) const {
  return (this->total_batch_normalization_layers +
              this->total_batch_renormalization_layers +
              this->total_ghost_batch_normalization_layers +
              this->total_streaming_normalization_layers !=
          0_UZ);
}

bool Model::Use__Batch_Normalization(void) const {
  return (this->total_batch_normalization_layers != 0_UZ);
}

bool Model::Use__Batch_Renormalization(void) const {
  return (this->total_batch_renormalization_layers != 0_UZ);
}

bool Model::Use__Ghost_Batch_Normalization(void) const {
  return (this->total_ghost_batch_normalization_layers != 0_UZ);
}

bool Model::Use__Streaming_Normalization(void) const {
  return (this->total_streaming_normalization_layers != 0_UZ);
}

bool Model::Use__Dropout__Alpha(void) const {
  return (this->total_dropout_alpha_layers != 0_UZ);
}

bool Model::Use__Dropout__Bernoulli(void) const {
  return (this->total_dropout_bernoulli_layers != 0_UZ);
}

bool Model::Use__Dropout__Bernoulli__Inverted(void) const {
  return (this->total_dropout_bernoulli_inverted_layers != 0_UZ);
}

bool Model::Use__Dropout__Gaussian(void) const {
  return (this->total_dropout_gaussian_layers != 0_UZ);
}

bool Model::Use__Dropout__ShakeDrop(void) const {
  return (this->total_dropout_shakedrop_layers != 0_UZ);
}

bool Model::Use__Dropout__Uout(void) const {
  return (this->total_dropout_uout_layers != 0_UZ);
}

bool Model::Use__Dropout__Zoneout(void) const {
  return (this->total_dropout_zoneout_layers != 0_UZ);
}

bool Model::Use__K_Sparse(void) const {
  return (this->total_k_sparse_layers != 0_UZ);
}

bool Model::Use__Tied_Parameter(void) const {
  return (this->total_tied_parameter_layers != 0_UZ);
}

bool Model::Use__Regularization__Constraint_Recurrent_Weight(void) const {
  return (this->total_constraint_recurrent_weight_layers != 0_UZ);
}

bool Model::Use__Multi_Label(void) const { return (this->use_multi_label); }

bool Model::Set__Multi_Label(bool const use_multi_label_received) {
  if (this->use_multi_label == use_multi_label_received) {
    return true;
  } else if (this->n_out == 1u && use_multi_label_received) {
    ERR(
        L"Can not use multi label with only one output.",);

    return false;
  }

  this->use_multi_label = use_multi_label_received;

  return true;
}

bool Model::Set__Input_Mode(bool const use_first_layer_as_input_received) {
  if (this->use_first_layer_as_input == use_first_layer_as_input_received) {
    return true;
  }

  switch (this->type) {
    case MODEL::AUTOENCODER:
      if (use_first_layer_as_input_received == false &&
          this->use_last_layer_as_output == use_first_layer_as_input_received) {
        ERR(
            L"Can not use the decoded layer has input. The "
            "decoded layer is the output.",);

        return false;
      }

      this->use_first_layer_as_input = use_first_layer_as_input_received;
      break;
    default:
      ERR(
          L"Network type (%d | %ls) is not managed in the switch.", this->type,
          MODEL_NAME[this->type].c_str());
      return false;
  }

  return true;
}

bool Model::Set__Output_Mode(bool const use_last_layer_as_output_received) {
  if (this->use_last_layer_as_output == use_last_layer_as_output_received) {
    return true;
  }

  switch (this->type) {
    case MODEL::AUTOENCODER:
      if (use_last_layer_as_output_received == false &&
          this->use_first_layer_as_input == use_last_layer_as_output_received) {
        ERR(
            L"Can not use the decoded layer has output. The "
            "decoded layer is the input.",);

        return false;
      }

      this->use_last_layer_as_output = use_last_layer_as_output_received;
      break;
    default:
      ERR(
          L"Network type (%d | %ls) is not managed in the switch.", this->type,
          MODEL_NAME[this->type].c_str());

      return false;
  }

  return true;
}

bool Model::Usable_Warm_Restarts(void) const {
  return (this->type_optimizer_function != OPTIMIZER::IRPROP_MINUS &&
          this->type_optimizer_function != OPTIMIZER::IRPROP_PLUS);
}

var Model::Activation_Function(ACTIVATION::TYPE const type, var x) {
  AF_FIRE(type, x, x);
  return x;
}

real Model::activation_fn_derivative(ACTIVATION::TYPE const type, real const x,
                                     real const q, real const y) {
  switch (type) {
    case ACTIVATION::COSINE:
      return (AF_COS_derive(x));
    case ACTIVATION::COSINE_SYMMETRIC:
      return (AF_COS_SYMMETRIC_derive(x));
    case ACTIVATION::ELU:
      return (AF_ELU_derive(x,
                            q, 1_r));
    case ACTIVATION::ELLIOT:
      return (AF_ELLIOT_derive(x));
    case ACTIVATION::ELLIOT_SYMMETRIC:
      return (
          AF_ELLIOT_SYMMETRIC_derive(x));
    case ACTIVATION::GAUSSIAN:
      return (AF_GAUSSIAN_derive(x,
                                 q));
    case ACTIVATION::GAUSSIAN_SYMMETRIC:
      return (AF_GAUSSIAN_SYMMETRIC_derive(x,
                                           q));
    case ACTIVATION::ISRU:
      return (AF_ISRU_derive(x,
                             q, 1_r));
    case ACTIVATION::ISRLU:
      return (AF_ISRLU_derive(x,
                              q, 1_r));
    case ACTIVATION::LINEAR:
    case ACTIVATION::LINEAR_PIECE:
    case ACTIVATION::LINEAR_PIECE_SYMMETRIC:
      return AF_LINEAR_derive;
    case ACTIVATION::LEAKY_RELU:
      return (AF_LRELU_derive(x,
                              AF_LRELU_ALPHA));
    case ACTIVATION::PARAMETRIC_RELU:
      return (AF_PRELU_derive(x,
                              AF_PRELU_ALPHA));
    case ACTIVATION::RELU:
      return (AF_RELU_derive(x));
    case ACTIVATION::SELU:
      return (AF_SELU_derive(x,
                             q));
    case ACTIVATION::SIGMOID:
      return (AF_SIGMOID_derive(q));
    case ACTIVATION::SINE:
      return (AF_SIN_derive(x));
    case ACTIVATION::SIGMOID_STEPWISE:
      return (AF_SIGMOID_derive(q));
    case ACTIVATION::SINE_SYMMETRIC:
      return (AF_SIN_SYMMETRIC_derive(x));
    case ACTIVATION::TANH:
      return (AF_TANH_derive(q));
    case ACTIVATION::TANH_STEPWISE:
      return (AF_TANH_derive(q));
    case ACTIVATION::SOFTMAX:
      // w.r.t Categorical Cross Entropy
      return q * (y - q);
    default:
      ERR(
          L"Activation function type (%d | %ls) is not managed in the switch.",
          type,
          ACTIVATION_NAME[type].c_str());
      return q;
  }

  return q;
}

bool Model::set_max_batch_size(size_t const max_batch_size) {
  if (max_batch_size == 0_UZ) {
    ERR(
        L"Maximum batch size can not be equal to zero.",);

    return false;
  }

  if (this->maximum_batch_size != max_batch_size) {
    this->maximum_batch_size = max_batch_size;

    if (this->update_mem_batch_size(max_batch_size, true) == false) {
      ERR(
          L"An error has been triggered from the "
          "\"update_mem_batch_size(%zu, true)\" function.",
          max_batch_size);

      return false;
    }
  }

  return true;
}

void Model::Clear_Outputs(void) {
  Layer *tmp_ptr_output_layer;

  switch (this->type) {
    case MODEL::AUTOENCODER:
      // Decode the encoded input as output.
      if (this->use_last_layer_as_output || this->pre_training_level != 0_UZ) {
        tmp_ptr_output_layer =
            this->ptr_last_layer - (this->pre_training_level == 0_UZ
                                        ? 1_UZ
                                        : this->pre_training_level);
      }
      // Else it use the coded part as output.
      else {
        tmp_ptr_output_layer =
            this->ptr_last_layer - ((this->total_layers - 3_UZ) / 2_UZ + 2_UZ);
      }
      break;
    default:
      tmp_ptr_output_layer = this->ptr_last_layer - 1;
      break;
  }

  size_t const tmp_number_outputs(*tmp_ptr_output_layer->ptr_number_outputs);

  VARZERO(tmp_ptr_output_layer->ptr_array_outputs,
         this->batch_size * tmp_number_outputs * this->seq_w *
             sizeof(var));
}

Layer *Model::Get__Input_Layer(void) const {
  Layer *tmp_ptr_input_layer;

  switch (this->type) {
    case MODEL::AUTOENCODER:
      // Use first layer or it is in pre-training mode.
      if (this->use_first_layer_as_input || this->pre_training_level != 0_UZ) {
        tmp_ptr_input_layer = this->ptr_array_layers;
      }
      // Else it use the coded part as input.
      else {
        tmp_ptr_input_layer =
            this->ptr_last_layer - ((this->total_layers - 3_UZ) / 2_UZ + 2_UZ);
      }
      break;
    default:
      tmp_ptr_input_layer = this->ptr_array_layers;
      break;
  }

  return (tmp_ptr_input_layer);
}

Layer *Model::Get__Output_Layer(void) const {
  Layer *tmp_ptr_output_layer;

  switch (this->type) {
    case MODEL::AUTOENCODER:
      // Decode the encoded input as output.
      if (this->use_last_layer_as_output || this->pre_training_level != 0_UZ) {
        tmp_ptr_output_layer =
            this->ptr_last_layer - (this->pre_training_level == 0_UZ
                                        ? 1_UZ
                                        : this->pre_training_level);
      }
      // Else it use the coded part as output.
      else {
        tmp_ptr_output_layer =
            this->ptr_last_layer - ((this->total_layers - 3_UZ) / 2_UZ + 2_UZ);
      }
      break;
    default:
      tmp_ptr_output_layer = this->ptr_last_layer - 1;
      break;
  }

  return (tmp_ptr_output_layer);
}

size_t Model::Get__Input_Size(void) const { return (this->n_inp); }

size_t Model::get_n_out(void) const {
  Layer const *tmp_ptr_output_layer;

  switch (this->type) {
    case MODEL::AUTOENCODER:
      // Decode the encoded input as output.
      if (this->use_last_layer_as_output || this->pre_training_level != 0_UZ) {
        tmp_ptr_output_layer =
            this->ptr_last_layer - (this->pre_training_level == 0_UZ
                                        ? 1_UZ
                                        : this->pre_training_level);
      }
      // Else it use the coded part as output.
      else {
        tmp_ptr_output_layer =
            this->ptr_last_layer - ((this->total_layers - 3_UZ) / 2_UZ + 2_UZ);
      }
      break;
    default:
      tmp_ptr_output_layer = this->ptr_last_layer - 1;
      break;
  }

  return (*tmp_ptr_output_layer->ptr_number_outputs);
}

var const *Model::get_out(size_t const data_index_received,
                              size_t const time_step_index_received) const {
  Layer const *tmp_ptr_output_layer;

  switch (this->type) {
    case MODEL::AUTOENCODER:
      // Decode the encoded input as output.
      if (this->use_last_layer_as_output || this->pre_training_level != 0_UZ) {
        tmp_ptr_output_layer =
            this->ptr_last_layer - (this->pre_training_level == 0_UZ
                                        ? 1_UZ
                                        : this->pre_training_level);
      }
      // Else it use the coded part as output.
      else {
        tmp_ptr_output_layer =
            this->ptr_last_layer - ((this->total_layers - 3_UZ) / 2_UZ + 2_UZ);
      }
      break;
    default:
      tmp_ptr_output_layer = this->ptr_last_layer - 1;
      break;
  }

  return (this->get_out(tmp_ptr_output_layer, data_index_received,
                             time_step_index_received));
}

var const *Model::get_out(Layer const *const layer_it,
                              size_t const data_index_received,
                              size_t const time_step_index_received) const {
  size_t const tmp_number_outputs(*layer_it->ptr_number_outputs);

  return (layer_it->ptr_array_outputs +
          data_index_received * tmp_number_outputs +
          this->batch_size * tmp_number_outputs * time_step_index_received);
}

real Model::get_layer_variance(
    size_t const layer_index_received,
    size_t const max_batch_size) const {
  if (max_batch_size == 0_UZ) {
    ERR(
        L"Maximum batch size can not be equal to zero.",);

    return (0_r);
  }

  return (this->get_layer_variance(
      this->ptr_array_layers + layer_index_received,
      max_batch_size));
}

real Model::get_layer_variance(
    Layer const *const ptr_layer_received,
    size_t const max_batch_size) const {
  if (max_batch_size == 0_UZ) {
    ERR(
        L"Maximum batch size can not be equal to zero.",);

    return (0_r);
  }

  size_t const tmp_output_size(*ptr_layer_received->ptr_number_outputs);
  size_t tmp_time_step_index, tmp_example_index, tmp_output_index;

  var const *const tmp_ptr_layer_ptr_array_outputs(
      ptr_layer_received->ptr_array_outputs),
      *tmp_ptr_array_outputs;
  real const tmp_batch_scaled(1_r / static_cast<real>(this->seq_w *
                                             max_batch_size *
                                             tmp_output_size));
  real tmp_output, tmp_mean(0_r), tmp_variance(0_r);

  for (tmp_time_step_index = 0_UZ;
       tmp_time_step_index != this->seq_w;
       ++tmp_time_step_index) {
    for (tmp_example_index = 0_UZ;
         tmp_example_index != max_batch_size;
         ++tmp_example_index) {
      tmp_ptr_array_outputs =
          tmp_ptr_layer_ptr_array_outputs +
          tmp_example_index * tmp_output_size +
          this->batch_size * tmp_output_size * tmp_time_step_index;

      for (tmp_output_index = 0_UZ; tmp_output_index != tmp_output_size;
           ++tmp_output_index) {
        tmp_output = cast(tmp_ptr_array_outputs[tmp_output_index]);

        tmp_mean += tmp_output;
        tmp_variance += tmp_output * tmp_output;
      }
    }
  }

  // Variance = V / B - pow(M / B, 2)
  tmp_mean *= tmp_batch_scaled;
  tmp_variance = tmp_variance * tmp_batch_scaled - tmp_mean * tmp_mean;

  return (tmp_variance);
}

size_t Layer::get_n_out(void) const {
  return (*this->ptr_number_outputs);
}

size_t Layer::Get__First_Connection_Index(void) const {
  return (*this->ptr_first_connection_index);
}

size_t Layer::Get__Last_Connection_Index(void) const {
  return (*this->ptr_last_connection_index);
}

std::wstring Model::Get__Parameters(bool const full_description_received) {
  std::wstring tmp_string(L"|===| GENERAL PARAMETERS |===|");
  tmp_string += CRLF
  tmp_string += L"Network type: " + MODEL_NAME[this->type] + L", " +
                std::to_wstring(this->type) + CRLF;
  tmp_string += L"Number time prediction(s): " +
                std::to_wstring(this->seq_w) + CRLF;
  tmp_string +=
      L"Number time delay(s): " + std::to_wstring(this->n_time_delay) +
      CRLF;
  tmp_string += L"Use the first layer as input: " +
                std::wstring(this->use_first_layer_as_input ? L"true" : L"false") +
                CRLF;
  tmp_string += L"Use the last layer as output: " +
                std::wstring(this->use_last_layer_as_output ? L"true" : L"false") +
                CRLF;
  tmp_string += L"|END| GENERAL PARAMETERS |END|" CRLF;
  tmp_string += CRLF;

  if (this->type_optimizer_function == OPTIMIZER::GD ||
      full_description_received) {
    tmp_string += L"|===| GRADIENT DESCENT PARAMETERS |===|" CRLF;
    tmp_string += L"Learning rate: " +
                  to_wstring(
                      this->learning_rate) +
                  CRLF;
    tmp_string += L"Learning momentum: " +
                  to_wstring(
                      this->learning_momentum) +
                  CRLF;
    tmp_string +=
        L"Use Nesterov: " + std::wstring(this->use_nesterov ? L"true" : L"false") +
        CRLF;
    tmp_string += L"|END| GRADIENT DESCENT PARAMETERS |END|" CRLF;
    tmp_string += CRLF;
  }

  if (this->type_optimizer_function == OPTIMIZER::QUICKPROP ||
      full_description_received) {
    tmp_string += L"|===| QUICKPROP PARAMETERS |===|" CRLF;
    tmp_string += L"Decay: " +
                  to_wstring(
                      this->quickprop_decay) +
                  CRLF;
    tmp_string += L"Mu: " +
                  to_wstring(
                      this->quickprop_mu) +
                  CRLF;
    tmp_string += L"|END| QUICKPROP PARAMETERS |END|" CRLF;
    tmp_string += CRLF;
  }

  if (this->type_optimizer_function == OPTIMIZER::IRPROP_PLUS ||
      this->type_optimizer_function == OPTIMIZER::IRPROP_MINUS ||
      full_description_received) {
    tmp_string += L"|===| RESILLENT PROPAGATION PARAMETERS |===|" CRLF;
    tmp_string += L"Increase factor: " +
                  to_wstring(
                      this->rprop_increase_factor) +
                  CRLF;
    tmp_string += L"Decrease factor: " +
                  to_wstring(
                      this->rprop_decrease_factor) +
                  CRLF;
    tmp_string += L"Delta minimum: " +
                  to_wstring(
                      this->rprop_delta_min) +
                  CRLF;
    tmp_string += L"Delta maximum: " +
                  to_wstring(
                      this->rprop_delta_max) +
                  CRLF;
    tmp_string += L"Delta zero: " +
                  to_wstring(
                      this->rprop_delta_zero) +
                  CRLF;
    tmp_string += L"|END| RESILLENT PROPAGATION PARAMETERS |END|" CRLF;
    tmp_string += CRLF;
  }

  if (this->type_optimizer_function == OPTIMIZER::SARPROP ||
      full_description_received) {
    tmp_string += L"|===| SARPROP PARAMETERS |===|" CRLF;
    tmp_string += L"Weight decay shift: " +
                  to_wstring(
                      this->sarprop_weight_decay_shift) +
                  CRLF;
    tmp_string += L"Step error threshold factor: " +
                  to_wstring(
                      this->sarprop_step_error_threshold_factor) +
                  CRLF;
    tmp_string += L"Step error shift: " +
                  to_wstring(
                      this->sarprop_step_error_shift) +
                  CRLF;
    tmp_string += L"Temperature: " +
                  to_wstring(
                      this->sarprop_temperature) +
                  CRLF;
    tmp_string += L"Epoch(s): " + std::to_wstring(this->sarprop_epoch) + CRLF;
    tmp_string += L"|END| SARPROP PARAMETERS |END|" CRLF;
    tmp_string += CRLF;
  }

  if (this->type_optimizer_function == OPTIMIZER::NOSADAM ||
      full_description_received) {
    tmp_string +=
        L"|===| " +
        to_upper(OPTIMIZER_NAME[this->type_optimizer_function]) +
        L" PARAMETERS |===|" CRLF;
    tmp_string += L"Learning rate: " +
                  to_wstring(
                      this->adam_learning_rate) +
                  CRLF;
    tmp_string +=
        L"Beta1: " +
        to_wstring(this->adam_beta1) +
        CRLF;
    tmp_string +=
        L"Beta2: " +
        to_wstring(this->adam_beta2) +
        CRLF;
    tmp_string += L"Epsilon: " +
                  to_wstring(
                      this->adam_epsilon) +
                  CRLF;
    tmp_string +=
        L"Bias correction: " +
        std::wstring(this->use_adam_bias_correction ? L"true" : L"false") + CRLF;
    tmp_string +=
        L"Gamma: " +
        to_wstring(this->adam_gamma) +
        CRLF;
    tmp_string +=
        L"|END| " +
        to_upper(OPTIMIZER_NAME[this->type_optimizer_function]) +
        L" PARAMETERS |END|" CRLF;
    tmp_string += CRLF;
  } else if (this->type_optimizer_function == OPTIMIZER::ADAM ||
             this->type_optimizer_function == OPTIMIZER::ADAMAX ||
             this->type_optimizer_function == OPTIMIZER::AMSGRAD ||
             full_description_received) {
    tmp_string +=
        L"|===| " +
        to_upper(OPTIMIZER_NAME[this->type_optimizer_function]) +
        L" PARAMETERS |===|" CRLF;
    tmp_string += L"Learning rate: " +
                  to_wstring(
                      this->adam_learning_rate) +
                  CRLF;
    tmp_string +=
        L"Beta1: " +
        to_wstring(this->adam_beta1) +
        CRLF;
    tmp_string +=
        L"Beta2: " +
        to_wstring(this->adam_beta2) +
        CRLF;
    tmp_string += L"Epsilon: " +
                  to_wstring(
                      this->adam_epsilon) +
                  CRLF;
    tmp_string +=
        L"Bias correction: " +
        std::wstring(this->use_adam_bias_correction ? L"true" : L"false") + CRLF;
    tmp_string +=
        L"|END| " +
        to_upper(OPTIMIZER_NAME[this->type_optimizer_function]) +
        L" PARAMETERS |END|" CRLF;
    tmp_string += CRLF;
  } else if (this->type_optimizer_function == OPTIMIZER::ADABOUND ||
             this->type_optimizer_function == OPTIMIZER::AMSBOUND ||
             full_description_received) {
    tmp_string +=
        L"|===| " +
        to_upper(OPTIMIZER_NAME[this->type_optimizer_function]) +
        L" PARAMETERS |===|" CRLF;
    tmp_string += L"Learning rate: " +
                  to_wstring(
                      this->adam_learning_rate) +
                  CRLF;
    tmp_string += L"Learning rate, final: " +
                  to_wstring(
                      this->learning_rate_final) +
                  CRLF;
    tmp_string +=
        L"Beta1: " +
        to_wstring(this->adam_beta1) +
        CRLF;
    tmp_string +=
        L"Beta2: " +
        to_wstring(this->adam_beta2) +
        CRLF;
    tmp_string += L"Epsilon: " +
                  to_wstring(
                      this->adam_epsilon) +
                  CRLF;
    tmp_string +=
        L"Bias correction: " +
        std::wstring(this->use_adam_bias_correction ? L"true" : L"false") + CRLF;
    tmp_string += L"Gamma: " +
                  to_wstring(
                      this->learning_gamma) +
                  CRLF;
    tmp_string +=
        L"|END| " +
        to_upper(OPTIMIZER_NAME[this->type_optimizer_function]) +
        L" PARAMETERS |END|" CRLF;
    tmp_string += CRLF;
  }

  tmp_string += L"|===| WARM RESTARTS PARAMETERS |===|" CRLF;
  tmp_string += L"Use warm restarts: " +
                std::wstring(this->use_warm_restarts ? L"true" : L"false") + CRLF;
  if (this->use_warm_restarts) {
    tmp_string += L"Learning rate, decay: " +
                  to_wstring(
                      this->warm_restarts_decay_learning_rate) +
                  CRLF;
    tmp_string += L"Maximum learning rate: " +
                  to_wstring(
                      this->warm_restarts_maximum_learning_rate) +
                  L" / " +
                  to_wstring(
                      this->warm_restarts_initial_maximum_learning_rate) +
                  CRLF;
    tmp_string += L"Minimum learning rate: " +
                  to_wstring(
                      this->warm_restarts_minimum_learning_rate) +
                  CRLF;
    tmp_string += L"Ti: " +
                  to_wstring(
                      this->warm_restarts_T_i) +
                  CRLF;
    tmp_string += L"Initial, Ti: " +
                  to_wstring(
                      this->warm_restarts_initial_T_i) +
                  CRLF;
    tmp_string += L"Warm restart multiplier: " +
                  to_wstring(
                      this->warm_restarts_multiplier) +
                  CRLF;
  }
  tmp_string += L"|END| WARM RESTARTS PARAMETERS |END|" CRLF;
  tmp_string += CRLF;

  tmp_string += L"|===| TRAINING PARAMETERS |===|" CRLF;
  tmp_string += L"Training algorithm: " +
                OPTIMIZER_NAME[this->type_optimizer_function] + L", " +
                std::to_wstring(this->type_optimizer_function) + CRLF;
  tmp_string += L"Loss function: " + LOSS_FN_NAME[this->type_loss_function] +
                L", " + std::to_wstring(this->type_loss_function) + CRLF;
  tmp_string +=
      L"Accuracy function: " + ACC_FN_NAME[this->type_accuracy_function] +
      L", " + std::to_wstring(this->type_accuracy_function) + CRLF;
  if (this->type_loss_function == LOSS_FN::BIT ||
      full_description_received) {
    tmp_string += L"Fail-limit: " +
                  to_wstring(
                      this->bit_fail_limit) +
                  CRLF;
  }
  tmp_string += L"Optimizer time step: " +
                std::to_wstring(static_cast<size_t>(this->optimizer_time_step)) +
                CRLF;
  tmp_string += L"Epoch time step: " +
                std::to_wstring(static_cast<size_t>(this->epoch_time_step)) +
                CRLF;
  tmp_string +=
      L"Pre-training level: " + std::to_wstring(this->pre_training_level) + CRLF;
  tmp_string += L"Use clip gradient: " +
                std::wstring(this->Use__Clip_Gradient() ? L"true" : L"false") +
                CRLF;
  if (this->Use__Clip_Gradient() || full_description_received) {
    tmp_string += L"clip gradient: " +
                  to_wstring(
                      this->clip_gradient) +
                  CRLF;
  }
  tmp_string += L"|END| TRAINING PARAMETERS |END|" CRLF;
  tmp_string += CRLF;

  tmp_string += L"|===| REGULARIZATION PARAMETERS |===|" CRLF;
  tmp_string +=
      L"Use dropout, bernoulli: " +
      std::wstring(this->Use__Dropout__Bernoulli() ? L"true" : L"false") + CRLF;
  tmp_string +=
      L"Use dropout, bernoulli inverted: " +
      std::wstring(this->Use__Dropout__Bernoulli__Inverted() ? L"true"
                                                            : L"false") +
      CRLF;
  tmp_string += L"Use dropout, gaussian: " +
                std::wstring(this->Use__Dropout__Gaussian() ? L"true" : L"false") +
                CRLF;
  tmp_string +=
      L"Use dropout, shakedrop: " +
      std::wstring(this->Use__Dropout__ShakeDrop() ? L"true" : L"false") + CRLF;
  tmp_string += L"Use dropout, uout: " +
                std::wstring(this->Use__Dropout__Uout() ? L"true" : L"false") +
                CRLF;
  tmp_string += L"Use dropout, zoneout: " +
                std::wstring(this->Use__Dropout__Zoneout() ? L"true" : L"false") +
                CRLF;
  tmp_string += L"Max-norm contraints: " +
                to_wstring(
                    this->regularization__max_norm_constraints) +
                CRLF;
  tmp_string += L"L1 regularization: " +
                to_wstring(
                    this->regularization__l1) +
                CRLF;
  tmp_string += L"L2 regularization: " +
                to_wstring(
                    this->regularization__l2) +
                CRLF;
  tmp_string += L"SRIP regularization: " +
                to_wstring(
                    this->regularization__srip) +
                CRLF;
  tmp_string += L"Weight decay: " +
                to_wstring(
                    this->weight_decay) +
                CRLF;
  tmp_string +=
      L"Use normalized weight decay: " +
      std::wstring(this->use_normalized_weight_decay ? L"true" : L"false") + CRLF;
  tmp_string += L"Use tied parameter: " +
                std::wstring(this->Use__Tied_Parameter() ? L"true" : L"false") +
                CRLF;
  tmp_string +=
      L"Use k-Sparse: " + std::wstring(this->Use__K_Sparse() ? L"true" : L"false") +
      CRLF;
  tmp_string +=
      L"Use constraint recurrent weight: " +
      std::wstring(this->Use__Regularization__Constraint_Recurrent_Weight()
                      ? L"true"
                      : L"false") +
      CRLF;
  tmp_string += L"|END| REGULARIZATION PARAMETERS |END|" CRLF;
  tmp_string += CRLF;

  tmp_string += L"|===| NORMALIZATION PARAMETERS |===|" CRLF;
  tmp_string +=
      L"Use batch normalization: " +
      std::wstring(this->Use__Batch_Normalization() ? L"true" : L"false") + CRLF;
  tmp_string +=
      L"Use batch renormalization: " +
      std::wstring(this->Use__Batch_Renormalization() ? L"true" : L"false") + CRLF;
  tmp_string += L"momentum average: " +
                to_wstring(
                    this->normalization_momentum_average) +
                CRLF;
  tmp_string += L"normalization epsilon: " +
                to_wstring(
                    this->normalization_epsilon) +
                CRLF;
  tmp_string += L"r correction maximum: " +
                to_wstring(
                    this->batch_renormalization_r_correction_maximum) +
                CRLF;
  tmp_string += L"d correction maximum: " +
                to_wstring(
                    this->batch_renormalization_d_correction_maximum) +
                CRLF;
  tmp_string += L"|===| NORMALIZATION PARAMETERS |===|" CRLF;
  tmp_string += CRLF;

  tmp_string += L"|===| LOSS PARAMETERS |===|" CRLF;
  tmp_string += L"Train: " +
                to_wstring(
                    this->loss_train) +
                CRLF;
  tmp_string += L"Valid: " +
                to_wstring(
                    this->loss_valid) +
                CRLF;
  tmp_string +=
      L"Testg: " +
      to_wstring(this->loss_testg) +
      CRLF;
  tmp_string += L"|END| LOSS PARAMETERS |END|" CRLF;
  tmp_string += CRLF;

  tmp_string += L"|===| ACCURANCY PARAMETERS |===|" CRLF;
  tmp_string += L"Variance: " +
                to_wstring(
                    this->acc_var) +
                CRLF;
  tmp_string += L"Train: " +
                to_wstring(
                    this->acc_train) +
                CRLF;
  tmp_string += L"Valid: " +
                to_wstring(
                    this->acc_valid) +
                CRLF;
  tmp_string += L"Testg: " +
                to_wstring(
                    this->acc_testg) +
                CRLF;
  tmp_string += L"|END| ACCURANCY PARAMETERS |END|" CRLF;
  tmp_string += CRLF;

  tmp_string += L"|===| COMPUTATION PARAMETERS |===|" CRLF;
  tmp_string +=
      L"Use CUDA: " + std::wstring(this->use_cu ? L"true" : L"false") + CRLF;
  tmp_string +=
      L"Use OpenMP: " + std::wstring(this->use_mp ? L"true" : L"false") + CRLF;
  tmp_string +=
      L"Maximum threads (percent): " + std::to_wstring(this->pct_threads) + L"%" +
      CRLF;
  tmp_string +=
      L"Number of threads: " + std::to_wstring(this->number_threads) + CRLF;
  tmp_string += L"Batch size: " + std::to_wstring(this->batch_size) + L" / " +
                std::to_wstring(this->maximum_batch_size) + CRLF;
  tmp_string += L"Maximum allowable memory: " +
                std::to_wstring(this->maximum_allowable_memory_bytes) +
                L" bytes | " +
                to_wstring(
                    static_cast<double>(this->maximum_allowable_memory_bytes) /
                        1024.0 / 1024.0,
                    4, std::ios_base::fixed) +
                L" MBs" + CRLF;
  tmp_string +=
      L"Size for one thread: " + std::to_wstring(this->Get__Threads_Sizeof(1u)) +
      L" bytes | " +
      to_wstring(
          static_cast<double>(this->Get__Threads_Sizeof(1u)) / 1024.0 / 1024.0,
                           4, std::ios_base::fixed) +
      L" MBs" + CRLF;
  tmp_string +=
      L"Size for a batch of size one: " +
      std::to_wstring(this->Get__Batch_Sizeof(1u)) + L" bytes | " +
      to_wstring(
          static_cast<double>(this->Get__Batch_Sizeof(1u)) / 1024.0 / 1024.0, 4,
          std::ios_base::fixed) +
      L" MBs" + CRLF;
  tmp_string +=
      L"Size neural network: " + std::to_wstring(this->Get__Sizeof()) +
      L" bytes | " +
      to_wstring(static_cast<double>(this->Get__Sizeof()) / 1024.0 / 1024.0, 4,
                 std::ios_base::fixed) +
      L" MBs" + CRLF;
  tmp_string += L"|END| COMPUTATION PARAMETERS |END|" CRLF;
  tmp_string += CRLF;

  tmp_string += L"|===| DIMENSION |===|" CRLF;
  tmp_string += L"Total layer(s): " + std::to_wstring(this->total_layers) + CRLF;
  tmp_string +=
      L"Total basic unit(s): " + std::to_wstring(this->total_basic_units) + L"/" +
      std::to_wstring(this->total_basic_units_allocated) + CRLF;
  tmp_string += L"Total basic indice unit(s): " +
                std::to_wstring(this->total_basic_indice_units) + L"/" +
                std::to_wstring(this->total_basic_indice_units_allocated) + CRLF;
  tmp_string +=
      L"Total neuron unit(s): " + std::to_wstring(this->total_neuron_units) +
      L"/" + std::to_wstring(this->total_neuron_units_allocated) + CRLF;
  tmp_string += L"Total AF unit(s): " + std::to_wstring(this->total_AF_units) +
                L"/" + std::to_wstring(this->total_AF_units_allocated) + CRLF;
  tmp_string += L"Total AF Ind unit(s): " +
                std::to_wstring(this->total_AF_Ind_recurrent_units) + L"/" +
                std::to_wstring(this->total_AF_Ind_recurrent_units_allocated) +
                CRLF;
  tmp_string += L"Total normalized unit(s): " +
                std::to_wstring(this->total_normalized_units) + L"/" +
                std::to_wstring(this->total_normalized_units_allocated) + CRLF;
  tmp_string +=
      L"Total block unit(s): " + std::to_wstring(this->total_block_units) + L"/" +
      std::to_wstring(this->total_block_units_allocated) + CRLF;
  tmp_string +=
      L"Total cell unit(s): " + std::to_wstring(this->total_cell_units) + L"/" +
      std::to_wstring(this->total_cell_units_allocated) + CRLF;
  tmp_string +=
      L"Total parameter(s): " + std::to_wstring(this->total_parameters) + L"/" +
      std::to_wstring(this->total_parameters_allocated) + CRLF;
  tmp_string += L"Total weight(s): " + std::to_wstring(this->total_weights) +
                L"/" + std::to_wstring(this->total_weights_allocated) + CRLF;
  tmp_string += L"Total bias(s): " + std::to_wstring(this->total_bias) + L"/" +
                std::to_wstring(this->total_bias_allocated) + CRLF;

  Layer const *const last_layer(this->ptr_last_layer - 1),
      *tmp_ptr_previous_layer, *layer_it(this->ptr_array_layers);

  // Input layer.
  tmp_string += L"  Input layer:" CRLF;
  tmp_string += L"    Type: " + LAYER_NAME[layer_it->type_layer] +
                L", " + std::to_wstring(layer_it->type_layer) + CRLF;
  tmp_string += L"    Type activation: " +
                LAYER_ACTIVATION_NAME[layer_it->type_activation] +
                L", " + std::to_wstring(layer_it->type_activation) + CRLF;
  tmp_string += L"    Type dropout: " +
                LAYER_DROPOUT_NAME[layer_it->type_dropout] + L", " +
                std::to_wstring(layer_it->type_dropout) + CRLF;
  tmp_string += L"      Use coded dropout: " +
                std::to_wstring(layer_it->use_coded_dropout) + CRLF;
  tmp_string += L"      Dropout value[0]: " +
                to_wstring(
                    layer_it->dropout_values[0]) +
                CRLF;
  tmp_string += L"      Dropout value[1]: " +
                to_wstring(
                    layer_it->dropout_values[1]) +
                CRLF;
  tmp_string += L"      Dropout value[2]: " +
                to_wstring(
                    layer_it->dropout_values[2]) +
                CRLF;
  tmp_string += L"    First connection index: " +
                std::to_wstring(*layer_it->ptr_first_connection_index) +
                CRLF;
  tmp_string += L"    Last connection index: " +
                std::to_wstring(*layer_it->ptr_last_connection_index) +
                CRLF;
  tmp_string +=
      L"    Number input(s): " + std::to_wstring(this->n_inp) + CRLF;
  // |END| Input layer. |END|

  auto information_layer_norm_fn(
      [self = this](
          Layer const *const layer_it) -> std::wstring {
        std::wstring tmp_string(L"");

        if (static_cast<size_t>(
                layer_it->ptr_last_normalized_unit -
                layer_it->ptr_array_normalized_units) <= 12_UZ) {
          // Normalization.
          if (self->Information__Layer__Normalization(
                  tmp_string, layer_it) == false) {
            ERR(
                L"An error has been triggered from the "
                L"\"Information__Layer__Normalization()\" function.",);

            return (L"");
          }
          // |END| Normalization. |END|
        } else {
          tmp_string +=
              L"    Type normalization: " +
              LAYER_NORM_NAME[layer_it->type_normalization] +
              L", " + std::to_wstring(layer_it->type_normalization) +
              CRLF;
          tmp_string +=
              L"      Use layer normalization before activation: " +
              std::wstring(layer_it
                                  ->use_layer_normalization_before_activation
                              ? L"true"
                              : L"false") +
              CRLF;
          tmp_string +=
              L"      Number normalized unit(s): " +
              std::to_wstring(static_cast<size_t>(
                  layer_it->ptr_last_normalized_unit -
                  layer_it->ptr_array_normalized_units)) +
              CRLF;
        }

        return (tmp_string);
      });

  auto information_layer_fc_fn(
      [self = this](
          Layer const *const layer_it,
          Layer const *const ptr_previous_layer_connected_received)
          -> std::wstring {
        std::wstring tmp_string(L"");

        if (*ptr_previous_layer_connected_received->ptr_number_outputs <=
                12_UZ &&
            *layer_it->ptr_number_outputs <= 12_UZ) {
          // Neuron(s).
          if (self->Information__Layer__FC(
                  tmp_string, layer_it,
                  ptr_previous_layer_connected_received) == false) {
            ERR(
                L"An error has been triggered from the "
                L"\"Information__Layer__FC()\" function.",);

            return (L"");
          }
          // |END| Neuron(s). |END|

          // Bias parameter(s).
          if (self->Information__Layer__Bias(tmp_string,
                                             layer_it) == false) {
            ERR(
                L"An error has been triggered from the "
                L"\"Information__Layer__Bias()\" function.",);

            return (L"");
          }
          // |END| Bias parameter(s). |END|
        } else {
          tmp_string += L"    Number neuron(s): " +
                        std::to_wstring(static_cast<size_t>(
                            layer_it->ptr_last_neuron_unit -
                            layer_it->ptr_array_neuron_units)) +
                        CRLF;
          tmp_string +=
              L"    Number bias: " +
              std::to_wstring(
                  layer_it->last_bias_connection_index -
                  layer_it->first_bias_connection_index) +
              CRLF;
        }

        return (tmp_string);
      });

  auto information_layer_af_fn(
      [self = this](
          Layer const *const layer_it) -> std::wstring {
        std::wstring tmp_string(L"");

        if (*layer_it->ptr_number_outputs <= 12_UZ) {
          // AF(s).
          if (self->Information__Layer__AF(tmp_string, layer_it) ==
              false) {
            ERR(
                L"An error has been triggered from the "
                L"\"Information__Layer__AF()\" function.",);

            return (L"");
          }
          // |END| AF(s). |END|
        } else {
          tmp_string += L"    Number AF(s): " +
                        std::to_wstring(static_cast<size_t>(
                            layer_it->ptr_last_AF_unit -
                            layer_it->ptr_array_AF_units)) +
                        CRLF;
        }

        return (tmp_string);
      });

  auto information_layer_af_ind_recurrent_fn(
      [self = this](
          Layer const *const layer_it) -> std::wstring {
    std::wstring tmp_string(L"");

        if (*layer_it->ptr_number_outputs <= 12_UZ) {
          // AF(s) Ind recurrent.
          if (self->Information__Layer__AF_Ind_Recurrent(
                  tmp_string, layer_it) == false) {
            ERR(
                L"An error has been triggered from the "
                L"\"Information__Layer__AF()\" function.",);

            return (L"");
          }
          // |END| AF(s) Ind recurrent. |END|
        } else {
          tmp_string +=
              L"    Number AF(s) Ind recurrent: " +
              std::to_wstring(static_cast<size_t>(
                  layer_it->ptr_last_AF_Ind_recurrent_unit -
                  layer_it->ptr_array_AF_Ind_recurrent_units)) +
              CRLF;
        }

        return (tmp_string);
      });

  auto information_layer_lstm_fn(
      [self = this](
          Layer const *const layer_it,
          Layer const *const ptr_previous_layer_connected_received)
          -> std::wstring {
        std::wstring tmp_string(L"");

        if (*ptr_previous_layer_connected_received->ptr_number_outputs <=
                12_UZ &&
            *layer_it->ptr_number_outputs <= 12_UZ) {
          // Blocks.
          if (self->Information__Layer__LSTM(
                  tmp_string, layer_it,
                  ptr_previous_layer_connected_received) == false) {
            ERR(
                L"An error has been triggered from the "
                L"\"Information__Layer__LSTM()\" function.",);

            return (L"");
          }
          // |END| Blocks. |END|

          // Bias parameter(s).
          if (self->Information__Layer__Bias(tmp_string,
                                             layer_it) == false) {
            ERR(
                L"An error has been triggered from the "
                L"\"Information__Layer__Bias()\" function.",);

            return (L"");
          }
          // |END| Bias parameter(s). |END|
        } else {
          tmp_string += L"    Number block unit(s): " +
                        std::to_wstring(static_cast<size_t>(
                            layer_it->ptr_last_block_unit -
                            layer_it->ptr_array_block_units)) +
                        CRLF;
          tmp_string += L"    Number cell unit(s): " +
                        std::to_wstring(static_cast<size_t>(
                            layer_it->ptr_last_cell_unit -
                            layer_it->ptr_array_cell_units)) +
                        CRLF;
        }

        return (tmp_string);
      });

  // Hidden layer(s).
  for (++layer_it; layer_it != last_layer;
       ++layer_it) {
    tmp_ptr_previous_layer = layer_it->previous_connected_layers[0];

    tmp_string += L"  Hidden layer [" +
                  std::to_wstring(layer_it - this->ptr_array_layers) +
                  L"]" CRLF;
    tmp_string += L"    Type: " + LAYER_NAME[layer_it->type_layer] +
                  L", " + std::to_wstring(layer_it->type_layer) + CRLF;
    tmp_string += L"    Use bidirectional: " +
                  std::to_wstring(layer_it->use_bidirectional) + CRLF;

    switch (layer_it->type_layer) {
      case LAYER::AVERAGE_POOLING:
      case LAYER::MAX_POOLING:
        tmp_string += L"    Kernel size: " +
                      std::to_wstring(layer_it->pooling_values[0]) +
                      CRLF;
        tmp_string += L"    Stride: " +
                      std::to_wstring(layer_it->pooling_values[1]) +
                      CRLF;
        tmp_string += L"    Padding: " +
                      std::to_wstring(layer_it->pooling_values[2]) +
                      CRLF;
        tmp_string += L"    Dilation: " +
                      std::to_wstring(layer_it->pooling_values[3]) +
                      CRLF;
        tmp_string +=
            L"    Ceil mode: " +
            std::wstring(layer_it->pooling_values[4] > 0_UZ ? L"true"
                                                                    : L"false") +
            CRLF;
        tmp_string += L"    Number feature(s): " +
                      std::to_wstring(*layer_it->ptr_number_outputs) +
                      CRLF;
        break;
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_RECURRENT:
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        tmp_string +=
            L"    Type activation: " +
            LAYER_ACTIVATION_NAME[layer_it->type_activation] +
            L", " + std::to_wstring(layer_it->type_activation) + CRLF;
        tmp_string += L"    Type dropout: " +
                      LAYER_DROPOUT_NAME[layer_it->type_dropout] +
                      L", " + std::to_wstring(layer_it->type_dropout) +
                      CRLF;
        tmp_string += L"      Use coded dropout: " +
                      std::to_wstring(layer_it->use_coded_dropout) +
                      CRLF;
        tmp_string += L"      Dropout value[0]: " +
                      to_wstring(
                          layer_it->dropout_values[0]) +
                      CRLF;
        tmp_string += L"      Dropout value[1]: " +
                      to_wstring(
                          layer_it->dropout_values[1]) +
                      CRLF;
        tmp_string += L"      Dropout value[2]: " +
                      to_wstring(
                          layer_it->dropout_values[2]) +
                      CRLF;
        tmp_string += information_layer_norm_fn(layer_it);
        tmp_string +=
            L"    Use tied parameter: " +
            std::wstring(layer_it->use_tied_parameter ? L"true"
                                                             : L"false") +
            CRLF;
        tmp_string +=
            L"    k-Sparsity: " + std::to_wstring(layer_it->k_sparsity) +
            CRLF;
        tmp_string += L"    Alpha sparsity: " +
                      to_wstring(
                          layer_it->alpha_sparsity) +
                      CRLF;
        tmp_string +=
            L"    Constraint recurrent weight lower bound: " +
            to_wstring(
                layer_it->constraint_recurrent_weight_lower_bound) +
            CRLF;
        tmp_string +=
            L"    Constraint recurrent weight upper bound: " +
            to_wstring(
                layer_it->constraint_recurrent_weight_upper_bound) +
            CRLF;
        tmp_string +=
            L"    First connection index: " +
            std::to_wstring(*layer_it->ptr_first_connection_index) +
            CRLF;
        tmp_string +=
            L"    Last connection index: " +
            std::to_wstring(*layer_it->ptr_last_connection_index) + CRLF;
        tmp_string +=
            L"    First bias connection index: " +
            std::to_wstring(layer_it->first_bias_connection_index) +
            CRLF;
        tmp_string +=
            L"    Last bias connection index: " +
            std::to_wstring(layer_it->last_bias_connection_index) + CRLF;
        tmp_string += L"    Number feature(s): " +
                      std::to_wstring(*layer_it->ptr_number_outputs) +
                      CRLF;
        tmp_string += information_layer_fc_fn(layer_it,
                                                 tmp_ptr_previous_layer);
        tmp_string += information_layer_af_fn(layer_it);
        tmp_string +=
            information_layer_af_ind_recurrent_fn(layer_it);
        break;
      case LAYER::LSTM:
        tmp_string +=
            L"    Type activation: " +
            LAYER_ACTIVATION_NAME[layer_it->type_activation] +
            L", " + std::to_wstring(layer_it->type_activation) + CRLF;
        tmp_string += L"    Type dropout: " +
                      LAYER_DROPOUT_NAME[layer_it->type_dropout] +
                      L", " + std::to_wstring(layer_it->type_dropout) +
                      CRLF;
        tmp_string += L"        Dropout value[0]: " +
                      to_wstring(
                          layer_it->dropout_values[0]) +
                      CRLF;
        tmp_string += L"        Dropout value[1]: " +
                      to_wstring(
                          layer_it->dropout_values[1]) +
                      CRLF;
        tmp_string += L"        Dropout value[2]: " +
                      to_wstring(
                          layer_it->dropout_values[2]) +
                      CRLF;
        tmp_string += information_layer_norm_fn(layer_it);
        tmp_string +=
            L"    Use tied parameter: " +
            std::wstring(layer_it->use_tied_parameter ? L"true"
                                                             : L"false") +
            CRLF;
        tmp_string +=
            L"    k-Sparsity: " + std::to_wstring(layer_it->k_sparsity) +
            CRLF;
        tmp_string += L"    Alpha sparsity: " +
                      to_wstring(
                          layer_it->alpha_sparsity) +
                      CRLF;
        tmp_string +=
            L"    Constraint recurrent weight lower bound: " +
            to_wstring(
                layer_it->constraint_recurrent_weight_lower_bound) +
            CRLF;
        tmp_string +=
            L"    Constraint recurrent weight upper bound: " +
            to_wstring(
                layer_it->constraint_recurrent_weight_upper_bound) +
            CRLF;
        tmp_string +=
            L"    First connection index: " +
            std::to_wstring(*layer_it->ptr_first_connection_index) +
            CRLF;
        tmp_string +=
            L"    Last connection index: " +
            std::to_wstring(*layer_it->ptr_last_connection_index) + CRLF;
        tmp_string +=
            L"    First bias connection index: " +
            std::to_wstring(layer_it->first_bias_connection_index) +
            CRLF;
        tmp_string +=
            L"    Last bias connection index: " +
            std::to_wstring(layer_it->last_bias_connection_index) + CRLF;
        tmp_string += L"    Number feature(s): " +
                      std::to_wstring(*layer_it->ptr_number_outputs) +
                      CRLF;
        tmp_string += information_layer_lstm_fn(layer_it,
                                                   tmp_ptr_previous_layer);
        break;
      case LAYER::RESIDUAL:
        tmp_string += L"    Block depth: " +
                      std::to_wstring(layer_it->block_depth) + CRLF;
        tmp_string += L"    Padding: " +
                      std::to_wstring(layer_it->pooling_values[2]) +
                      CRLF;
        tmp_string += information_layer_norm_fn(layer_it);
        tmp_string += L"    Number feature(s): " +
                      std::to_wstring(*layer_it->ptr_number_outputs) +
                      CRLF;
        break;
      default:
        ERR(
            L"Layer type (%d | %ls) is not managed in the "
            "switch.",
            layer_it->type_layer,
            LAYER_NAME[layer_it->type_layer].c_str());
        return (L"");
    }
  }
  // |END| Hidden layer(s). |END|

  // Output layer.
  tmp_ptr_previous_layer = layer_it->previous_connected_layers[0];

  tmp_string += L"  Output layer:" CRLF;
  tmp_string += L"    Type: " + LAYER_NAME[layer_it->type_layer] +
                L", " + std::to_wstring(layer_it->type_layer) + CRLF;
  tmp_string += L"    Type activation: " +
                LAYER_ACTIVATION_NAME[layer_it->type_activation] +
                L", " + std::to_wstring(layer_it->type_activation) + CRLF;
  tmp_string += L"    First connection index: " +
                std::to_wstring(*layer_it->ptr_first_connection_index) +
                CRLF;
  tmp_string += L"    Last connection index: " +
                std::to_wstring(*layer_it->ptr_last_connection_index) +
                CRLF;
  tmp_string += L"    First bias connection index: " +
                std::to_wstring(layer_it->first_bias_connection_index) +
                CRLF;
  tmp_string += L"    Last bias connection index: " +
                std::to_wstring(layer_it->last_bias_connection_index) +
                CRLF;

  if (*tmp_ptr_previous_layer->ptr_number_outputs <= 12_UZ &&
      *layer_it->ptr_number_outputs <= 12_UZ) {
    // Neuron(s).
    if (this->Information__Output_Layer(tmp_string, layer_it,
                                        tmp_ptr_previous_layer) == false) {
      ERR(
          L"An error has been triggered from the "
          L"\"Information__Output_Layer()\" function.",);

      return (L"");
    }
    // |END| Neuron(s). |END|

    // Bias parameter(s).
    if (this->Information__Layer__Bias(tmp_string, layer_it) == false) {
      ERR(
          L"An error has been triggered from the "
          L"\"Information__Layer__Bias()\" function.",);

      return (L"");
    }
    // |END| Bias parameter(s). |END|
  } else {
    tmp_string +=
        L"    Number output(s): " + std::to_wstring(this->n_out) + CRLF;
  }
  // |END| Output layer. |END|
  tmp_string += L"|END| DIMENSION |END|" CRLF;

  return (tmp_string);
}

bool Model::Multi_Class_Classification(void) const
//{ return((this->ptr_last_layer - 1)->type_activation ==
// LAYER_ACTIVATION::SOFTMAX); }
{
  return (this->n_out > 1_UZ);
}

size_t Model::Get__Total_Layers(void) const { return (this->total_layers); }

void Model::Reset__Parameter__Mask_Dropout(
    bool *ptr_array_units_mask_dropout_bernoulli_received) {
  Layer const *const last_layer(this->ptr_last_layer);
  Layer *layer_it(this->ptr_array_layers);

  this->ptr_array_units_mask_dropout_bernoulli =
      ptr_array_units_mask_dropout_bernoulli_received;

  for (layer_it = this->ptr_array_layers;
       layer_it != last_layer; ++layer_it) {
    layer_it->ptr_array__mask__dropout__bernoulli =
        ptr_array_units_mask_dropout_bernoulli_received;

    ptr_array_units_mask_dropout_bernoulli_received +=
        static_cast<size_t>(layer_it->ptr_last_AF_unit -
                            layer_it->ptr_array_AF_units) *
        this->seq_w;
    ptr_array_units_mask_dropout_bernoulli_received +=
        static_cast<size_t>(
            layer_it->ptr_last_AF_Ind_recurrent_unit -
            layer_it->ptr_array_AF_Ind_recurrent_units) *
        this->seq_w;
  }
}

void Model::Reset__Parameters__Cell_Unit__Mask_Dropout(
    bool *ptr_array_cell_units_mask_dropout_received) {
  Layer const *const last_layer(this->ptr_last_layer);
  Layer *layer_it(this->ptr_array_layers);

  BlockUnit const *tmp_ptr_last_block_unit;
  BlockUnit *tmp_ptr_block_unit_it;

  CellUnit const *tmp_ptr_last_cell_unit;
  CellUnit *tmp_ptr_cell_unit_it;

  this->ptr_array_cell_units_mask_dropout_zoneout =
      ptr_array_cell_units_mask_dropout_received;

  for (layer_it = this->ptr_array_layers;
       layer_it != last_layer; ++layer_it) {
    for (tmp_ptr_last_block_unit = layer_it->ptr_last_block_unit,
        tmp_ptr_block_unit_it = layer_it->ptr_array_block_units;
         tmp_ptr_block_unit_it != tmp_ptr_last_block_unit;
         ++tmp_ptr_block_unit_it) {
      tmp_ptr_block_unit_it->ptr_array_mask_dropout_zoneout =
          ptr_array_cell_units_mask_dropout_received;

      for (tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
          tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units;
           tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit;
           ++tmp_ptr_cell_unit_it,
          ++ptr_array_cell_units_mask_dropout_received) {
        tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_state =
            ptr_array_cell_units_mask_dropout_received;
        tmp_ptr_cell_unit_it->ptr_mask_dropout_zoneout_output =
            ptr_array_cell_units_mask_dropout_received +
            this->seq_w * this->total_cell_units_allocated;
      }
    }

    ptr_array_cell_units_mask_dropout_received +=
        static_cast<size_t>(layer_it->ptr_last_cell_unit -
                            layer_it->ptr_array_cell_units) *
        (this->seq_w - 1_UZ);
  }
}

void Model::Reset__Parameter__Normalized_Unit(void) {
  size_t tmp_number_units, tmp_index;

  void **tmp_ptr_array_ptr_connections(this->ptr_array_ptr_connections +
                                       this->total_weights_allocated +
                                       this->total_bias_allocated);

  var *tmp_ptr_array_parameters_scale_it(this->ptr_array_parameters +
                                        this->total_weights_allocated +
                                        this->total_bias_allocated),
      *tmp_ptr_array_parameters_shift_it(
          this->ptr_array_parameters + this->total_weights_allocated +
          this->total_bias_allocated + this->total_normalized_units);

  Layer const *const last_layer(this->ptr_last_layer);
  Layer *layer_it(this->ptr_array_layers);

  BlockUnit const *tmp_ptr_last_block_unit;
  BlockUnit *tmp_ptr_block_unit_it;

  CellUnit const *tmp_ptr_last_cell_unit;
  CellUnit *tmp_ptr_cell_unit_it;

  union Normalized_unit const *tmp_ptr_last_normalized_unit;
  union Normalized_unit *tmp_ptr_normalized_unit_it;

  for (layer_it = this->ptr_array_layers;
       layer_it != last_layer; ++layer_it) {
    if ((tmp_number_units = static_cast<size_t>(
             layer_it->ptr_last_normalized_unit -
             layer_it->ptr_array_normalized_units)) != 0_UZ) {
      switch (layer_it->type_layer) {
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::RESIDUAL:
          for (tmp_ptr_last_normalized_unit =
                   layer_it->ptr_last_normalized_unit,
              tmp_ptr_normalized_unit_it =
                   layer_it->ptr_array_normalized_units;
               tmp_ptr_normalized_unit_it != tmp_ptr_last_normalized_unit;
               ++tmp_ptr_normalized_unit_it,
              ++tmp_ptr_array_parameters_scale_it,
              ++tmp_ptr_array_parameters_shift_it,
              ++tmp_ptr_array_ptr_connections) {
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_scale =
                tmp_ptr_array_parameters_scale_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units.ptr_shift =
                tmp_ptr_array_parameters_shift_it;

            tmp_ptr_array_ptr_connections[0] = tmp_ptr_normalized_unit_it;
            tmp_ptr_array_ptr_connections[this->total_normalized_units] =
                tmp_ptr_normalized_unit_it;
          }
          break;
        case LAYER::LSTM:
          if (static_cast<size_t>(layer_it->ptr_last_block_unit -
                                  layer_it->ptr_array_block_units) !=
              0_UZ) {
            // [0]: Block input, input.
            // [1]: Block input, recurrent.
            // [2]: Cell state activate.

            tmp_ptr_last_cell_unit = layer_it->ptr_last_cell_unit;

            tmp_number_units =
                static_cast<size_t>(layer_it->ptr_last_cell_unit -
                                    layer_it->ptr_array_cell_units);

            for (tmp_index = 0_UZ; tmp_index != 3_UZ; ++tmp_index) {
              for (tmp_ptr_cell_unit_it =
                       layer_it->ptr_array_cell_units;
                   tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit;
                   ++tmp_ptr_cell_unit_it) {
                tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index]
                    .normalized_batch_units.ptr_scale =
                    tmp_ptr_array_parameters_scale_it++;
                tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index]
                    .normalized_batch_units.ptr_shift =
                    tmp_ptr_array_parameters_shift_it++;

                tmp_ptr_array_ptr_connections[0] = tmp_ptr_cell_unit_it;
                tmp_ptr_array_ptr_connections[this->total_normalized_units] =
                    tmp_ptr_cell_unit_it;
                ++tmp_ptr_array_ptr_connections;
              }
            }

            // [3]: Input gate, input.
            // [4]: Input gate, recurrent.
            // [5]: Forget gate, input.
            // [6]: Forget gate, recurrent.
            // [7]: Output gate, input.
            // [8]: Output gate, recurrent.

            tmp_ptr_last_block_unit = layer_it->ptr_last_block_unit;

            tmp_number_units =
                static_cast<size_t>(layer_it->ptr_last_block_unit -
                                    layer_it->ptr_array_block_units);

            for (tmp_index = 0_UZ; tmp_index != 6_UZ; ++tmp_index) {
              for (tmp_ptr_block_unit_it =
                       layer_it->ptr_array_block_units;
                   tmp_ptr_block_unit_it != tmp_ptr_last_block_unit;
                   ++tmp_ptr_block_unit_it) {
                tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index]
                    .normalized_batch_units.ptr_scale =
                    tmp_ptr_array_parameters_scale_it++;
                tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index]
                    .normalized_batch_units.ptr_shift =
                    tmp_ptr_array_parameters_shift_it++;

                tmp_ptr_array_ptr_connections[0] = tmp_ptr_block_unit_it;
                tmp_ptr_array_ptr_connections[this->total_normalized_units] =
                    tmp_ptr_block_unit_it;
                ++tmp_ptr_array_ptr_connections;
              }
            }
          }
          break;
        default:
          ERR(
              L"Type layer (%u | %ls) is not managed in the "
              L"switch.",
              layer_it->type_layer,
              LAYER_NAME[layer_it->type_layer].c_str());
          return;
      }
    }
  }
}

void Model::Reset__Derivative_Parameter__Normalized_Unit(void) {
  size_t tmp_index;

  real *tmp_ptr_array_derivatives_parameters_scale_it(
      this->ptr_array_derivatives_parameters + this->total_weights_allocated +
      this->total_bias_allocated),
      *tmp_ptr_array_derivatives_parameters_shift_it(
          this->ptr_array_derivatives_parameters +
          this->total_weights_allocated + this->total_bias_allocated +
          this->total_normalized_units);

  Layer const *const last_layer(this->ptr_last_layer);
  Layer *layer_it(this->ptr_array_layers);

  BlockUnit const *tmp_ptr_last_block_unit;
  BlockUnit *tmp_ptr_block_unit_it;

  CellUnit const *tmp_ptr_last_cell_unit;
  CellUnit *tmp_ptr_cell_unit_it;

  union Normalized_unit const *tmp_ptr_last_normalized_unit;
  union Normalized_unit *tmp_ptr_normalized_unit_it;

  for (layer_it = this->ptr_array_layers;
       layer_it != last_layer; ++layer_it) {
    if (static_cast<size_t>(layer_it->ptr_last_normalized_unit -
                            layer_it->ptr_array_normalized_units) !=
        0_UZ) {
      switch (layer_it->type_layer) {
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::RESIDUAL:
          for (tmp_ptr_last_normalized_unit =
                   layer_it->ptr_last_normalized_unit,
              tmp_ptr_normalized_unit_it =
                   layer_it->ptr_array_normalized_units;
               tmp_ptr_normalized_unit_it != tmp_ptr_last_normalized_unit;
               ++tmp_ptr_normalized_unit_it,
              ++tmp_ptr_array_derivatives_parameters_scale_it,
              ++tmp_ptr_array_derivatives_parameters_shift_it) {
            tmp_ptr_normalized_unit_it->normalized_batch_units
                .ptr_array_derivatives_scales =
                tmp_ptr_array_derivatives_parameters_scale_it;
            tmp_ptr_normalized_unit_it->normalized_batch_units
                .ptr_array_derivatives_shifts =
                tmp_ptr_array_derivatives_parameters_shift_it;
          }
          break;
        case LAYER::LSTM:
          if (static_cast<size_t>(layer_it->ptr_last_block_unit -
                                  layer_it->ptr_array_block_units) !=
              0_UZ) {
            // [0]: Block input, input.
            // [1]: Block input, recurrent.
            // [2]: Cell state activate.

            tmp_ptr_last_cell_unit = layer_it->ptr_last_cell_unit;

            for (tmp_index = 0_UZ; tmp_index != 3_UZ; ++tmp_index) {
              for (tmp_ptr_cell_unit_it =
                       layer_it->ptr_array_cell_units;
                   tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit;
                   ++tmp_ptr_cell_unit_it) {
                tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index]
                    .normalized_batch_units.ptr_array_derivatives_scales =
                    tmp_ptr_array_derivatives_parameters_scale_it++;
                tmp_ptr_cell_unit_it->ptr_array_normalized_units[tmp_index]
                    .normalized_batch_units.ptr_array_derivatives_shifts =
                    tmp_ptr_array_derivatives_parameters_shift_it++;
              }
            }

            // [3]: Input gate, input.
            // [4]: Input gate, recurrent.
            // [5]: Forget gate, input.
            // [6]: Forget gate, recurrent.
            // [7]: Output gate, input.
            // [8]: Output gate, recurrent.

            tmp_ptr_last_block_unit = layer_it->ptr_last_block_unit;

            for (tmp_index = 0_UZ; tmp_index != 6_UZ; ++tmp_index) {
              for (tmp_ptr_block_unit_it =
                       layer_it->ptr_array_block_units;
                   tmp_ptr_block_unit_it != tmp_ptr_last_block_unit;
                   ++tmp_ptr_block_unit_it) {
                tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index]
                    .normalized_batch_units.ptr_array_derivatives_scales =
                    tmp_ptr_array_derivatives_parameters_scale_it++;
                tmp_ptr_block_unit_it->ptr_array_normalized_units[tmp_index]
                    .normalized_batch_units.ptr_array_derivatives_shifts =
                    tmp_ptr_array_derivatives_parameters_shift_it++;
              }
            }
          }
          break;
        default:
          ERR(
              L"Type layer (%u | %ls) is not managed in the "
              L"switch.",
              layer_it->type_layer,
              LAYER_NAME[layer_it->type_layer].c_str());
          return;
      }
    }
  }
}

void Model::Clear__Parameter__Normalized_Unit(void) {
  size_t tmp_number_units;

  Layer const *const last_layer(this->ptr_last_layer);
  Layer *layer_it(this->ptr_array_layers);

  for (layer_it = this->ptr_array_layers;
       layer_it != last_layer; ++layer_it) {
    if ((tmp_number_units = static_cast<size_t>(
             layer_it->ptr_last_normalized_unit -
             layer_it->ptr_array_normalized_units)) != 0_UZ) {
      // clear shift.
      VARZERO(layer_it->ptr_array_normalized_units
                 ->normalized_batch_units.ptr_shift,
             tmp_number_units * sizeof(var));

      // clear scale.
      switch (layer_it->type_layer) {
        case LAYER::FULLY_CONNECTED:
          Mem::fill<var>(layer_it->ptr_array_normalized_units
                                ->normalized_batch_units.ptr_scale,
                            layer_it->ptr_array_normalized_units
                                    ->normalized_batch_units.ptr_scale +
                                tmp_number_units,
                            1_r);
          break;
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::LSTM:
          Mem::fill<var>(layer_it->ptr_array_normalized_units
                                ->normalized_batch_units.ptr_scale,
                            layer_it->ptr_array_normalized_units
                                    ->normalized_batch_units.ptr_scale +
                                tmp_number_units,
                            0.1_r);
          break;
        case LAYER::RESIDUAL:
          Mem::fill<var>(layer_it->ptr_array_normalized_units
                                ->normalized_batch_units.ptr_scale,
                            layer_it->ptr_array_normalized_units
                                    ->normalized_batch_units.ptr_scale +
                                tmp_number_units,
                            this->seq_w == 1_UZ ? 1_r : 0.1_r);
          break;
        default:
          ERR(
              L"Type layer (%u | %ls) is not managed in the "
              L"switch.",
              layer_it->type_layer,
              LAYER_NAME[layer_it->type_layer].c_str());
          return;
      }

      // clear average mean.
      VARZERO(layer_it->ptr_array_normalized_units
                 ->normalized_batch_units.ptr_mean_average,
             tmp_number_units * this->seq_w * sizeof(var));

      // clear average variance.
      Mem::fill<var>(layer_it->ptr_array_normalized_units
                            ->normalized_batch_units.ptr_variance_average,
                        layer_it->ptr_array_normalized_units
                                ->normalized_batch_units.ptr_variance_average +
                            tmp_number_units * this->seq_w,
                        1_r);
    }
  }
}

bool Model::weights_initialized(void) const {
  return (this->_initialized__weight);
}

bool Model::initialize_weights(
    DatasetV1 const *const ptr_Dataset_received) {
  switch (this->_type_weights_initializer) {
    case INITIALIZER::LSUV:
      if (this->Initialization__LSUV(ptr_Dataset_received) == false) {
        ERR(
            L"An error has been triggered from the "
            L"\"Initialization__LSUV(ptr)\" function.",);

        return false;
      }
      break;
    default:
      ERR(
          L"Type weights initializer (%u | %ls) is not managed in "
          L"the switch.",
          this->_type_weights_initializer,
          INITIALIZER_NAME[this->_type_weights_initializer].c_str());
      return false;
  }

  return true;
}

size_t Model::Get__Batch_Sizeof(size_t batch_size) const {
  size_t tmp_total_size_t(0_UZ);

  if (batch_size == 0_UZ) {
    batch_size = this->batch_size;
  }

  // Basic unit(s).
  if (this->ptr_array_basic_units_values != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_basic_units_allocated * sizeof(var);
  }
  if (this->ptr_array_basic_units_errors != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_basic_units_allocated * sizeof(real);
  }
  // |END| Basic unit(s). |END|

  // Basic indice unit(s).
  if (this->ptr_array_basic_indice_units_indices != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_basic_indice_units_allocated * sizeof(size_t);
  }
  if (this->ptr_array_basic_indice_units_values != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_basic_indice_units_allocated * sizeof(var);
  }
  if (this->ptr_array_basic_indice_units_errors != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_basic_indice_units_allocated * sizeof(real);
  }
  // |END| Basic indice unit(s). |END|

  // Neuron unit(s).
  if (this->ptr_array_neuron_units_summations != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_neuron_units_allocated * sizeof(var);
  }
  if (this->ptr_array_neuron_units_errors != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_neuron_units_allocated * sizeof(real);
  }
  // |END| Neuron unit(s). |END|

  // AF unit(s).
  if (this->ptr_array_AF_units_values != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_AF_units_allocated * sizeof(var);
  }
  if (this->ptr_array_AF_units_errors != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_AF_units_allocated * sizeof(real);
  }
  // |END| AF unit(s). |END|

  // AF Ind recurrent unit(s).
  if (this->ptr_array_AF_Ind_recurrent_units_pre_AFs != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_AF_Ind_recurrent_units_allocated *
                        sizeof(var);
  }
  if (this->ptr_array_AF_Ind_recurrent_units_errors != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_AF_Ind_recurrent_units_allocated *
                        sizeof(real);
  }
  if (this->ptr_array_AF_Ind_recurrent_units_dAFs != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_AF_Ind_recurrent_units_allocated *
                        sizeof(real);
  }
  // |END| AF Ind recurrent unit(s). |END|

  // Normalized unit(s).
  if (this->ptr_array_normalized_batch_units_values_hats != nullptr) {
    tmp_total_size_t += batch_size *
                        this->total_normalized_units_allocated *
                        this->seq_w * sizeof(var);
  }
  if (this->ptr_array_normalized_batch_units_values_normalizes != nullptr) {
    tmp_total_size_t += batch_size *
                        this->total_normalized_units_allocated *
                        this->seq_w * sizeof(var);
  }
  if (this->ptr_array_normalized_batch_units_errors != nullptr) {
    tmp_total_size_t += batch_size *
                        this->total_normalized_units_allocated *
                        this->seq_w * sizeof(real);
  }
  // |END| Normalized unit(s). |END|

  // LSTM.
  if (this->ptr_array_cells_summations_cells_inputs != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(var);
  }
  if (this->ptr_array_cells_summations_input_cells_inputs != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(var);
  }
  if (this->ptr_array_cells_summations_recurrent_cells_inputs != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(var);
  }
  if (this->ptr_array_blocks_summations_inputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_blocks_summations_input_inputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_blocks_summations_recurrent_inputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_blocks_summations_forgets_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_blocks_summations_input_forgets_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_blocks_summations_recurrent_forgets_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_blocks_summations_outputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_blocks_summations_input_outputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_blocks_summations_recurrent_outputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_cells_inputs != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(var);
  }
  if (this->ptr_array_cells_states != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(var);
  }
  if (this->ptr_array_cells_states_activates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(var);
  }
  if (this->ptr_array_cells_outputs != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(var);
  }
  if (this->ptr_array_blocks_inputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_blocks_forgets_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_blocks_outputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(var);
  }
  if (this->ptr_array_cells_delta_inputs != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(real);
  }
  if (this->ptr_array_cells_delta_input_inputs != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(real);
  }
  if (this->ptr_array_cells_delta_recurrent_inputs != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(real);
  }
  if (this->ptr_array_cells_delta_states != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(real);
  }
  if (this->ptr_array_cells_delta_outputs != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_cell_units * sizeof(real);
  }
  if (this->ptr_array_blocks_delta_inputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(real);
  }
  if (this->ptr_array_blocks_delta_input_inputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(real);
  }
  if (this->ptr_array_blocks_delta_recurrent_inputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(real);
  }
  if (this->ptr_array_blocks_delta_forgets_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(real);
  }
  if (this->ptr_array_blocks_delta_input_forgets_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(real);
  }
  if (this->ptr_array_blocks_delta_recurrent_forgets_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(real);
  }
  if (this->ptr_array_blocks_delta_outputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(real);
  }
  if (this->ptr_array_blocks_delta_input_outputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(real);
  }
  if (this->ptr_array_blocks_delta_recurrent_outputs_gates != nullptr) {
    tmp_total_size_t += batch_size * this->seq_w *
                        this->total_block_units * sizeof(real);
  }
  // |END| LSTM. |END|

  return (tmp_total_size_t);
}

size_t Model::Get__Threads_Sizeof(size_t number_threads_received) const {
  size_t tmp_total_size_t(0_UZ);

  if (number_threads_received == 0_UZ) {
    number_threads_received = this->number_threads;
  }

  if (this->ptr_array_k_sparse_activities != nullptr) {
    tmp_total_size_t += number_threads_received *
                        (this->total_AF_units_allocated +
                         this->total_AF_Ind_recurrent_units_allocated +
                         this->total_block_units_allocated) *
                        sizeof(std::pair<size_t, var>);
  }

  // Normalized unit(s).
  if (this->ptr_array_normalized_batch_units_means != nullptr) {
    tmp_total_size_t += number_threads_received *
                        this->total_normalized_units_allocated *
                        this->seq_w * sizeof(var);
  }
  if (this->ptr_array_normalized_batch_units_variances != nullptr) {
    tmp_total_size_t += number_threads_received *
                        this->total_normalized_units_allocated *
                        this->seq_w * sizeof(var);
  }
  if (this->ptr_array_normalized_batch_units_derivatives_means != nullptr) {
    tmp_total_size_t += number_threads_received *
                        this->total_normalized_units_allocated *
                        this->seq_w * sizeof(real);
  }
  if (this->ptr_array_normalized_batch_units_derivatives_variances != nullptr) {
    tmp_total_size_t += number_threads_received *
                        this->total_normalized_units_allocated *
                        this->seq_w * sizeof(real);
  }
  // |END| Normalized unit(s). |END|

  // Cost.
  if (this->ptr_array_number_loss != nullptr) {
    tmp_total_size_t += number_threads_received * sizeof(size_t);
  }
  if (this->ptr_array_number_bit_fail != nullptr) {
    tmp_total_size_t += number_threads_received * sizeof(size_t);
  }
  if (this->ptr_array_loss_values != nullptr) {
    tmp_total_size_t += number_threads_received * sizeof(double);
  }
  if (this->ptr_array_accuracy_values[0] != nullptr) {
    tmp_total_size_t += number_threads_received * sizeof(double);
  }
  if (this->ptr_array_accuracy_values[1] != nullptr) {
    tmp_total_size_t += number_threads_received * sizeof(double);
  }
  if (this->ptr_array_accuracy_values[2] != nullptr) {
    tmp_total_size_t += number_threads_received * sizeof(double);
  }
  if (this->ptr_array_accuracy_values[3] != nullptr) {
    tmp_total_size_t += number_threads_received * sizeof(double);
  }
  if (this->ptr_array_accuracy_values[4] != nullptr) {
    tmp_total_size_t += number_threads_received * sizeof(double);
  }
  // |END| Cost. |END|

  // Parameters.
  if (this->ptr_array_derivatives_parameters != nullptr) {
    tmp_total_size_t +=
        number_threads_received * this->total_parameters_allocated * sizeof(real);
  }
  // |END| Parameters. |END|

  // Generator.
  if (this->bernoulli != nullptr) {
    tmp_total_size_t +=
        number_threads_received * sizeof(Dist::Bernoulli);
  }
  if (this->ptr_array_Class_Generator_Bernoulli_Zoneout_State != nullptr) {
    tmp_total_size_t +=
        number_threads_received * sizeof(Dist::Bernoulli);
  }
  if (this->ptr_array_Class_Generator_Bernoulli_Zoneout_Hidden != nullptr) {
    tmp_total_size_t +=
        number_threads_received * sizeof(Dist::Bernoulli);
  }
  if (this->ptr_array_Class_Generator_Real_Uout != nullptr) {
    tmp_total_size_t +=
        number_threads_received * sizeof(Dist::Real);
  }
  if (this->ptr_array_Class_Generator_Real_Gaussian != nullptr) {
    tmp_total_size_t +=
        number_threads_received * sizeof(Dist::Gaussian);
  }
  // |END| Generator. |END|

  return (tmp_total_size_t);
}

size_t Model::Get__Sizeof(size_t number_threads_received,
                          size_t batch_size) const {
  size_t tmp_total_size_t(0_UZ);

  tmp_total_size_t += sizeof(Model);  // this

  tmp_total_size_t += this->Get__Threads_Sizeof(number_threads_received == 0_UZ
                                                    ? this->number_threads
                                                    : number_threads_received);

  tmp_total_size_t += this->Get__Batch_Sizeof(
      batch_size == 0_UZ ? this->batch_size : batch_size);

  // Parameters.
  if (this->ptr_array_ptr_connections != nullptr) {
    tmp_total_size_t += this->total_parameters_allocated * sizeof(void *);
  }

  if (this->ptr_array_parameters != nullptr) {
    tmp_total_size_t += this->total_parameters_allocated * sizeof(var);
  }
  if (this->ptr_array_mask_regularized_parameters != nullptr) {
    tmp_total_size_t += this->total_parameters_allocated * sizeof(real);
  }

  //    Optimizer iRPROP.
  if (this->ptr_array_previous_steps != nullptr) {
    tmp_total_size_t += this->total_parameters_allocated * sizeof(real);
  }
  if (this->ptr_array_previous_delta_parameters != nullptr) {
    tmp_total_size_t += this->total_parameters_allocated * sizeof(real);
  }
  if (this->ptr_array_previous_derivatives_parameters != nullptr) {
    tmp_total_size_t += this->total_parameters_allocated * sizeof(real);
  }
  //    |END| Optimizer iRPROP. |END|

  //    Optimizer Adam.
  if (this->ptr_array_previous_biased_first_moment != nullptr) {
    tmp_total_size_t += this->total_parameters_allocated * sizeof(real);
  }
  if (this->ptr_array_previous_biased_second_moment != nullptr) {
    tmp_total_size_t += this->total_parameters_allocated * sizeof(real);
  }
  //    |END| Optimizer Adam. |END|

  //    Optimizer AMSGrad.
  if (this->ptr_array_previous_biased_second_moment_hat != nullptr) {
    tmp_total_size_t += this->total_parameters_allocated * sizeof(real);
  }
  //    |END| Optimizer AMSGrad. |END|
  // |END| Parameters. |END|

  // Dropout variable.
  if (this->ptr_array_units_mask_dropout_bernoulli != nullptr) {
    tmp_total_size_t += (this->total_AF_units_allocated +
                         this->total_AF_Ind_recurrent_units_allocated) *
                        this->seq_w * sizeof(bool);
  }
  if (this->ptr_array_layers_mask_dropout_shakedrop != nullptr) {
    tmp_total_size_t += this->total_layers * this->batch_size *
                        this->seq_w * sizeof(bool);
  }
  if (this->ptr_array_cell_units_mask_dropout_zoneout != nullptr) {
    tmp_total_size_t += 2_UZ * this->total_cell_units_allocated *
                        this->seq_w * sizeof(bool);
  }
  // |END| Dropout variable. |END|

  // Layer(s).
  if (this->ptr_array_layers != nullptr) {
    tmp_total_size_t += this->total_layers * sizeof(Layer);
  }
  if (this->ptr_array_layers_number_outputs != nullptr) {
    tmp_total_size_t += this->total_layers * sizeof(size_t);
  }
  if (this->ptr_array_layers_first_connection_index != nullptr) {
    tmp_total_size_t += this->total_layers * sizeof(size_t);
  }
  if (this->ptr_array_layers_last_connection_index != nullptr) {
    tmp_total_size_t += this->total_layers * sizeof(size_t);
  }

  if (this->ptr_array_number_neurons_by_layer != nullptr) {
    tmp_total_size_t += this->total_layers * sizeof(size_t);
  }
  if (this->ptr_array_number_connections_by_layer != nullptr) {
    tmp_total_size_t += this->total_layers * sizeof(size_t);
  }
  // |END| Layer(s). |END|

  // Neuron unit(s).
  if (this->ptr_array_neuron_units != nullptr) {
    tmp_total_size_t +=
        this->total_neuron_units_allocated * sizeof(Neuron_unit);
  }

  if (this->ptr_array_neuron_units_first_forward_connection_index != nullptr) {
    tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t);
  }
  if (this->ptr_array_neuron_units_last_forward_connection_index != nullptr) {
    tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t);
  }
  if (this->ptr_array_neuron_units_number_forward_connections != nullptr) {
    tmp_total_size_t += this->total_neuron_units_allocated * sizeof(size_t);
  }
  // |END| Neuron unit(s). |END|

  // AF unit(s).
  if (this->ptr_array_AF_units != nullptr) {
    tmp_total_size_t += this->total_AF_units_allocated * sizeof(AF_unit);
  }

  if (this->ptr_array_AF_units_type_activation_function != nullptr) {
    tmp_total_size_t +=
        this->total_AF_units_allocated * sizeof(ACTIVATION::TYPE);
  }
  // |END| AF unit(s). |END|

  // AF Ind recurrent unit(s).
  if (this->ptr_array_AF_Ind_recurrent_units != nullptr) {
    tmp_total_size_t += this->total_AF_Ind_recurrent_units_allocated *
                        sizeof(AF_Ind_recurrent_unit);
  }

  if (this->ptr_array_AF_Ind_recurrent_units_recurrent_connection_index !=
      nullptr) {
    tmp_total_size_t +=
        this->total_AF_Ind_recurrent_units_allocated * sizeof(size_t);
  }

  if (this->ptr_array_AF_Ind_recurrent_units_type_activation_function !=
      nullptr) {
    tmp_total_size_t += this->total_AF_Ind_recurrent_units_allocated *
                        sizeof(ACTIVATION::TYPE);
  }
  // |END| AF Ind recurrent unit(s). |END|

  // Cell unit(s).
  if (this->ptr_array_cell_units != nullptr) {
    tmp_total_size_t +=
        this->total_cell_units_allocated * sizeof(CellUnit);
  }

  // Block unit(s).
  if (this->ptr_array_block_units != nullptr) {
    tmp_total_size_t +=
        this->total_block_units_allocated * sizeof(BlockUnit);
  }

  // Normalized unit(s).
  if (this->ptr_array_normalized_units != nullptr) {
    tmp_total_size_t +=
        this->total_normalized_units_allocated * sizeof(union Normalized_unit);
  }

  if (this->ptr_array_normalized_batch_units_r_corrections != nullptr) {
    tmp_total_size_t += this->total_normalized_units_allocated *
                        this->seq_w * sizeof(var);
  }
  if (this->ptr_array_normalized_batch_units_d_corrections != nullptr) {
    tmp_total_size_t += this->total_normalized_units_allocated *
                        this->seq_w * sizeof(var);
  }
  if (this->ptr_array_normalized_batch_units_means_averages != nullptr) {
    tmp_total_size_t += this->total_normalized_units_allocated *
                        this->seq_w * sizeof(var);
  }
  if (this->ptr_array_normalized_batch_units_variances_averages != nullptr) {
    tmp_total_size_t += this->total_normalized_units_allocated *
                        this->seq_w * sizeof(var);
  }
  // |END| Normalized unit(s). |END|

  // CUDA
#ifdef COMPILE_CUDA
  if (this->cumodel != NULL) {
    tmp_total_size_t += this->cumodel->Get__Sizeof();
  }
#endif
  // |END| CUDA |END|

  return (tmp_total_size_t);
}

var const *Layer::Get__Array_Summations__Cell__Block_Input__Input__Activation(
    void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_normalized_units[0]
                    .normalized_batch_units.ptr_array_values_normalizes);
      } else {
        return (this->ptr_array_cell_units->ptr_summation_input_cell_input);
      }
    default:
      return nullptr;
  }
}

var const *
Layer::Get__Array_Summations__Cell__Block_Input__Recurrent__Activation(
    void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_normalized_units[1]
                    .normalized_batch_units.ptr_array_values_normalizes);
      } else {
        return (this->ptr_array_cell_units->ptr_summation_recurrent_cell_input);
      }
    default:
      return nullptr;
  }
}

var const *Layer::Get__Array_Summations__Cell__Cell_State__Activation(
    void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_normalized_units[2]
                    .normalized_batch_units.ptr_array_values_normalizes);
      } else {
        return (this->ptr_array_cell_units->ptr_cell_state);
      }
    default:
      return nullptr;
  }
}

var const *Layer::Get__Array_Summations__Block__Input_Gate__Input__Activation(
    void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_normalized_units[3]
                    .normalized_batch_units.ptr_array_values_normalizes);
      } else {
        return (this->ptr_array_block_units->ptr_summation_input_inputs_gates);
      }
    default:
      return nullptr;
  }
}

var const *
Layer::Get__Array_Summations__Block__Input_Gate__Recurrent__Activation(
    void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_normalized_units[4]
                    .normalized_batch_units.ptr_array_values_normalizes);
      } else {
        return (
            this->ptr_array_block_units->ptr_summation_recurrent_inputs_gates);
      }
    default:
      return nullptr;
  }
}

var const *Layer::Get__Array_Summations__Block__Forget_Gate__Input__Activation(
    void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_normalized_units[5]
                    .normalized_batch_units.ptr_array_values_normalizes);
      } else {
        return (this->ptr_array_block_units->ptr_summation_input_forgets_gates);
      }
    default:
      return nullptr;
  }
}

var const *
Layer::Get__Array_Summations__Block__Forget_Gate__Recurrent__Activation(
    void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_normalized_units[6]
                    .normalized_batch_units.ptr_array_values_normalizes);
      } else {
        return (
            this->ptr_array_block_units->ptr_summation_recurrent_forgets_gates);
      }
    default:
      return nullptr;
  }
}

var const *Layer::Get__Array_Summations__Block__Output_Gate__Input__Activation(
    void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_normalized_units[7]
                    .normalized_batch_units.ptr_array_values_normalizes);
      } else {
        return (this->ptr_array_block_units->ptr_summation_input_outputs_gates);
      }
    default:
      return nullptr;
  }
}

var const *
Layer::Get__Array_Summations__Block__Output_Gate__Recurrent__Activation(
    void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_normalized_units[8]
                    .normalized_batch_units.ptr_array_values_normalizes);
      } else {
        return (
            this->ptr_array_block_units->ptr_summation_recurrent_outputs_gates);
      }
    default:
      return nullptr;
  }
}

real const *Layer::Get__Array_Deltas__Cell__Block_Input__Input(void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_cell_units->ptr_delta_cell_input_input);
      } else {
        return (this->ptr_array_cell_units->ptr_delta_cell_input);
      }
    default:
      return nullptr;
  }
}

real const *Layer::Get__Array_Deltas__Cell__Block_Input__Recurrent(void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_cell_units->ptr_delta_cell_recurrent_input);
      } else {
        return (this->ptr_array_cell_units->ptr_delta_cell_input);
      }
    default:
      return nullptr;
  }
}

real const *Layer::Get__Array_Deltas__Block__Input_Gate__Input(void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_block_units->ptr_delta_input_inputs_gates);
      } else {
        return (this->ptr_array_block_units->ptr_delta_inputs_gates);
      }
    default:
      return nullptr;
  }
}

real const *Layer::Get__Array_Deltas__Block__Input_Gate__Recurrent(void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_block_units->ptr_delta_recurrent_inputs_gates);
      } else {
        return (this->ptr_array_block_units->ptr_delta_inputs_gates);
      }
    default:
      return nullptr;
  }
}

real const *Layer::Get__Array_Deltas__Block__Forget_Gate__Input(void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_block_units->ptr_delta_input_forgets_gates);
      } else {
        return (this->ptr_array_block_units->ptr_delta_forgets_gates);
      }
    default:
      return nullptr;
  }
}

real const *Layer::Get__Array_Deltas__Block__Forget_Gate__Recurrent(void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_block_units->ptr_delta_recurrent_forgets_gates);
      } else {
        return (this->ptr_array_block_units->ptr_delta_forgets_gates);
      }
    default:
      return nullptr;
  }
}

real const *Layer::Get__Array_Deltas__Block__Output_Gate__Input(void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_block_units->ptr_delta_input_outputs_gates);
      } else {
        return (this->ptr_array_block_units->ptr_delta_outputs_gates);
      }
    default:
      return nullptr;
  }
}

real const *Layer::Get__Array_Deltas__Block__Output_Gate__Recurrent(void) const {
  switch (this->type_layer) {
    case LAYER::LSTM:
      if (this->Use__Normalization()) {
        return (this->ptr_array_block_units->ptr_delta_recurrent_outputs_gates);
      } else {
        return (this->ptr_array_block_units->ptr_delta_outputs_gates);
      }
    default:
      return nullptr;
  }
}

void Model::Update_Parameter(size_t const batch_size,
                             size_t const training_size) {
  if (this->use_mp && this->is_mp_initialized) {
    this->update_weights_mp(batch_size, training_size);
  } else {
    this->update_weights_st(batch_size, training_size);
  }
}

void Model::update_weights_mp(size_t const batch_size,
                                     size_t const training_size) {
  if (this->Get__Regularization__L1() != 0_r) {
    this->Update_Derivative_Weight__Regularization__L1__OpenMP(
        batch_size, 0_UZ, this->total_weights);
  }

  if (this->Get__Regularization__L2() != 0_r) {
    this->Update_Derivative_Weight__Regularization__L2__OpenMP(
        batch_size, 0_UZ, this->total_weights);
  }

  switch (this->type_optimizer_function) {
    case OPTIMIZER::GD:
    case OPTIMIZER::IRPROP_MINUS:
    case OPTIMIZER::IRPROP_PLUS:
    case OPTIMIZER::QUICKPROP:
    case OPTIMIZER::SARPROP:
    case OPTIMIZER::ADABOUND:
    case OPTIMIZER::ADAM:
    case OPTIMIZER::ADAMAX:
    case OPTIMIZER::AMSBOUND:
    case OPTIMIZER::AMSGRAD:
    case OPTIMIZER::NOSADAM:
      this->merge_mp_derivatives(0_UZ, this->total_parameters);
      break;
    default:
      ERR(
          L"Optimizer type (%d | %ls) is not managed in the "
          L"switch.",
          this->type_optimizer_function,
          OPTIMIZER_NAME[this->type_optimizer_function].c_str());
      break;
  }

  if (this->Use__Clip_Gradient()) {
    this->Clip_Gradient__OpenMP(0_UZ, this->total_parameters);
  }

  switch (this->type_optimizer_function) {
    case OPTIMIZER::GD:
      this->Update_Parameter__Gradient_Descent(batch_size,
                                               training_size, 0_UZ,
                                               this->total_parameters);
      break;
    case OPTIMIZER::IRPROP_PLUS:
      this->Update_Parameter__iRPROP_plus(0_UZ, this->total_parameters);
      break;
    // case OPTIMIZER::QUICKPROP: Update_Weight_QuickProp_Parallel(this,
    // this->get_n_data(), 0_UZ, this->total_parameters); break; case
    // OPTIMIZER::SARPROP: Update_Weight_SARProp_Parallel(this,
    // this->sarprop_epoch, 0_UZ, this->total_parameters); break;
    case OPTIMIZER::ADABOUND:
      this->Update_Parameters__AdaBound(batch_size,
                                        training_size, 0_UZ,
                                        this->total_parameters);
      break;
    case OPTIMIZER::ADAM:
      this->Update_Parameters__Adam(batch_size, training_size,
                                    0_UZ, this->total_parameters);
      break;
    // case OPTIMIZER::ADAMAX:
    // case OPTIMIZER::TYPE_OPTIMIZER_SADAMAX:
    // this->Update_Weight_AdaMax(0_UZ, this->total_parameters); break;
    case OPTIMIZER::AMSBOUND:
      this->Update_Parameters__AMSBound(batch_size,
                                        training_size, 0_UZ,
                                        this->total_parameters);
      break;
    case OPTIMIZER::AMSGRAD:
      this->Update_Parameters__AMSGrad(batch_size,
                                       training_size, 0_UZ,
                                       this->total_parameters);
      break;
    case OPTIMIZER::NOSADAM:
      this->Update_Parameters__NosAdam(batch_size,
                                       training_size, 0_UZ,
                                       this->total_parameters);
      break;
    default:
      ERR(
          L"Optimizer type (%d | %ls) is not managed in the "
          L"switch.",
          this->type_optimizer_function,
          OPTIMIZER_NAME[this->type_optimizer_function].c_str());
      break;
  }

  if (this->Get__Regularization__Max_Norm_Constraints() != 0_r) {
    this->Update_Weight_Regularization__Max_Norm_Constraints__OpenMP(
        0_UZ, this->total_weights);
  }

  if (this->Use__Regularization__Constraint_Recurrent_Weight()) {
    this->Update_Weight_Regularization__Constraint_Recurrent_Weight(
        0_UZ, this->total_weights);
  }

  if (this->Use__Tied_Parameter()) {
    this->Tied__Transpose();
  }
}

void Model::update_weights_st(size_t const batch_size,
                                   size_t const training_size) {
  if (this->Get__Regularization__L1() != 0_r) {
    this->Update_Derivative_Weight__Regularization__L1__Loop(
        batch_size, 0_UZ, this->total_weights);
  }

  if (this->Get__Regularization__L2() != 0_r) {
    this->Update_Derivative_Weight__Regularization__L2__Loop(
        batch_size, 0_UZ, this->total_weights);
  }

  if (this->Use__Clip_Gradient()) {
    this->Clip_Gradient__Loop(0_UZ, this->total_parameters);
  }

  switch (this->type_optimizer_function) {
    case OPTIMIZER::GD:
      this->Update_Parameter__Gradient_Descent(batch_size,
                                               training_size, 0_UZ,
                                               this->total_parameters);
      break;
    case OPTIMIZER::IRPROP_PLUS:
      this->Update_Parameter__iRPROP_plus(0_UZ, this->total_parameters);
      break;
    // case OPTIMIZER::QUICKPROP: update_model_quickprop(this,
    // this->get_n_data(), 0_UZ, this->total_parameters); break; case
    // OPTIMIZER::SARPROP: update_model_sarprop(this, this->sarprop_epoch,
    // 0_UZ, this->total_parameters); break;
    case OPTIMIZER::ADABOUND:
      this->Update_Parameters__AdaBound(batch_size,
                                        training_size, 0_UZ,
                                        this->total_parameters);
      break;
    case OPTIMIZER::ADAM:
      this->Update_Parameters__Adam(batch_size, training_size,
                                    0_UZ, this->total_parameters);
      break;
    // case OPTIMIZER::ADAMAX:
    // case OPTIMIZER::TYPE_OPTIMIZER_SADAMAX:
    // this->Update_Weight_AdaMax(0_UZ, this->total_parameters); break;
    case OPTIMIZER::AMSBOUND:
      this->Update_Parameters__AMSBound(batch_size,
                                        training_size, 0_UZ,
                                        this->total_parameters);
      break;
    case OPTIMIZER::AMSGRAD:
      this->Update_Parameters__AMSGrad(batch_size,
                                       training_size, 0_UZ,
                                       this->total_parameters);
      break;
    case OPTIMIZER::NOSADAM:
      this->Update_Parameters__NosAdam(batch_size,
                                       training_size, 0_UZ,
                                       this->total_parameters);
      break;
    default:
      ERR(
          L"Optimizer type (%d | %ls) is not managed in the "
          L"switch.",
          this->type_optimizer_function,
          OPTIMIZER_NAME[this->type_optimizer_function].c_str());
      break;
  }

  if (this->Get__Regularization__Max_Norm_Constraints() != 0_r) {
    this->Update_Weight_Regularization__Max_Norm_Constraints__Loop(
        0_UZ, this->total_weights);
  }

  if (this->Use__Regularization__Constraint_Recurrent_Weight()) {
    this->Update_Weight_Regularization__Constraint_Recurrent_Weight(
        0_UZ, this->total_weights);
  }

  if (this->Use__Tied_Parameter()) {
    this->Tied__Transpose();
  }
}

Layer const *Model::Get__Layer(size_t const index_received) const {
  return (this->ptr_array_layers + index_received);
}

Layer const *Model::Get__End_Layer__Active(void) const {
  Layer *tmp_ptr_last_layer_active;

  switch (this->type) {
    case MODEL::AUTOENCODER:
      tmp_ptr_last_layer_active =
          this->ptr_last_layer - ((this->total_layers - 3_UZ) / 2_UZ + 1_UZ);
      break;
    default:
      tmp_ptr_last_layer_active = this->ptr_last_layer;
      break;
  }

  return (tmp_ptr_last_layer_active);
}

/* Strategy comparison index:
    [0], default: tr_validating < td_validating && tr_testing <= td_testing ||
   tr_validating <= td_validating && tr_testing < td_testing. [1]: tr_testing <=
   global_testing. [2]: tr_validating <= td_testing && tr_testing < td_testing.
 */
bool Model::Strategy_Comparison__Loss(
    unsigned int const strategy_index_received,
    ENV::TYPE const type_dataset_in_received,
    ENV::TYPE const type_dataset_out_received,
    Model const *const ptr_source_Neural_Network_received) const {
  switch (strategy_index_received) {
    case 0:
    default:
      if ((ptr_source_Neural_Network_received->get_loss(
               type_dataset_in_received) <
               this->get_loss(type_dataset_in_received) &&
           ptr_source_Neural_Network_received->get_loss(
               type_dataset_out_received) <=
               this->get_loss(type_dataset_out_received)) ||
          (ptr_source_Neural_Network_received->get_loss(
               type_dataset_in_received) <=
               this->get_loss(type_dataset_in_received) &&
           ptr_source_Neural_Network_received->get_loss(
               type_dataset_out_received) <
               this->get_loss(type_dataset_out_received))) {
        return true;
      }
      break;
    case 1:
      if (ptr_source_Neural_Network_received->get_loss(
              type_dataset_in_received) <=
              this->get_loss(type_dataset_out_received) &&
          ptr_source_Neural_Network_received->get_loss(
              type_dataset_out_received) <
              this->get_loss(type_dataset_out_received)) {
        return true;
      }
      break;
  }

  return false;
}

bool Model::Strategy_Comparison__Accuracy(
    unsigned int const strategy_index_received,
    ENV::TYPE const type_dataset_in_received,
    ENV::TYPE const type_dataset_out_received,
    Model const *const ptr_source_Neural_Network_received) const {
  switch (strategy_index_received) {
    case 0:
    default:
      if ((ptr_source_Neural_Network_received->get_accu(
               type_dataset_in_received) >
               this->get_accu(type_dataset_in_received) &&
           ptr_source_Neural_Network_received->get_accu(
               type_dataset_out_received) >=
               this->get_accu(type_dataset_out_received)) ||
          (ptr_source_Neural_Network_received->get_accu(
               type_dataset_in_received) >=
               this->get_accu(type_dataset_in_received) &&
           ptr_source_Neural_Network_received->get_accu(
               type_dataset_out_received) >
               this->get_accu(type_dataset_out_received))) {
        return true;
      }
      break;
    case 1:
      if (ptr_source_Neural_Network_received->get_accu(
              type_dataset_in_received) >=
              this->get_accu(type_dataset_out_received) &&
          ptr_source_Neural_Network_received->get_accu(
              type_dataset_out_received) >
              this->get_accu(type_dataset_out_received)) {
        return true;
      }
      break;
  }

  return false;
}

bool Model::Compare(
    bool const use_metric_loss_received,
    bool const dataset_in_equal_less_dataset_out_accepted_received,
    ENV::TYPE const type_holdout_dataset_received,
    double const minimum_loss_holdout_dataset_accepted_received,
    Model const *const ptr_source_Neural_Network_received) const {
  ENV::TYPE tmp_type_dataset_in, tmp_type_dataset_out;

  switch (type_holdout_dataset_received) {
    case ENV::TRAIN:
      tmp_type_dataset_in = ENV::TRAIN;
      tmp_type_dataset_out = ENV::TRAIN;
      break;
    case ENV::VALID:
      tmp_type_dataset_in = ENV::TRAIN;
      tmp_type_dataset_out = ENV::VALID;
      break;
    case ENV::TESTG:
      tmp_type_dataset_in = ENV::VALID;
      tmp_type_dataset_out = ENV::TESTG;
      break;
    default:
      ERR(
          L"Evaluation type (%d | %ls) is not managed in the "
          L"switch.",
          type_holdout_dataset_received,
          ENV_NAME[type_holdout_dataset_received].c_str());
      return false;
  }

  if (use_metric_loss_received) {
    if (ptr_source_Neural_Network_received->get_loss(tmp_type_dataset_out) <=
            minimum_loss_holdout_dataset_accepted_received &&
        ((dataset_in_equal_less_dataset_out_accepted_received == false &&
          this->Strategy_Comparison__Loss(
              0, tmp_type_dataset_in, tmp_type_dataset_out,
              ptr_source_Neural_Network_received)) ||
         (dataset_in_equal_less_dataset_out_accepted_received &&
          (this->Strategy_Comparison__Loss(
               1, tmp_type_dataset_in, tmp_type_dataset_out,
               ptr_source_Neural_Network_received) ||
           (this->get_loss(tmp_type_dataset_in) >
                this->get_loss(tmp_type_dataset_out) &&
            this->Strategy_Comparison__Loss(
                0, tmp_type_dataset_in, tmp_type_dataset_out,
                ptr_source_Neural_Network_received)))))) {
      return true;
    } else {
      return false;
    }
  } else {
    if (ptr_source_Neural_Network_received->get_accu(
            tmp_type_dataset_out) >=
            minimum_loss_holdout_dataset_accepted_received &&
        ((dataset_in_equal_less_dataset_out_accepted_received == false &&
          this->Strategy_Comparison__Accuracy(
              0, tmp_type_dataset_in, tmp_type_dataset_out,
              ptr_source_Neural_Network_received)) ||
         (dataset_in_equal_less_dataset_out_accepted_received &&
          (this->Strategy_Comparison__Accuracy(
               1, tmp_type_dataset_in, tmp_type_dataset_out,
               ptr_source_Neural_Network_received) ||
           (this->get_accu(tmp_type_dataset_in) <
                this->get_accu(tmp_type_dataset_out) &&
            this->Strategy_Comparison__Accuracy(
                0, tmp_type_dataset_in, tmp_type_dataset_out,
                ptr_source_Neural_Network_received)))))) {
      return true;
    } else {
      return false;
    }
  }
}

void Model::layer_initialize_const_bias(real const bias,
                                       Layer const *const layer_it) {
  var const *const tmp_ptr_parameter_end(
      this->ptr_array_parameters +
      layer_it->last_bias_connection_index);
  var *tmp_ptr_parameter_it(this->ptr_array_parameters +
                           layer_it->first_bias_connection_index);

  for (; tmp_ptr_parameter_it != tmp_ptr_parameter_end;
       ++tmp_ptr_parameter_it) {
    *tmp_ptr_parameter_it = bias;
  }
}

void Model::lstm_initialize_const_bias(real const bias,
                                             Layer const *const layer_it) {
  size_t const tmp_number_cell_units(
      static_cast<size_t>(layer_it->ptr_last_cell_unit -
                          layer_it->ptr_array_cell_units)),
      tmp_number_block_units(
          static_cast<size_t>(layer_it->ptr_last_block_unit -
                              layer_it->ptr_array_block_units));

  if (tmp_number_cell_units * tmp_number_block_units != 0_UZ) {
    var const *tmp_ptr_parameter_end;
    var *tmp_ptr_parameter_it(
        this->ptr_array_parameters +
        layer_it->first_bias_connection_index);

    // Cell input && Input gate.
    for (tmp_ptr_parameter_end = tmp_ptr_parameter_it + tmp_number_cell_units +
                                 tmp_number_block_units;
         tmp_ptr_parameter_it != tmp_ptr_parameter_end;
         ++tmp_ptr_parameter_it) {
      *tmp_ptr_parameter_it = bias;
    }
    // |END| Cell input && Input gate. |END|

    // Forget gate.
    for (tmp_ptr_parameter_end = tmp_ptr_parameter_it + tmp_number_block_units;
         tmp_ptr_parameter_it != tmp_ptr_parameter_end;
         ++tmp_ptr_parameter_it) {
      *tmp_ptr_parameter_it = 1_r;
    }
    // |END| Forget gate. |END|

    // Output gate.
    for (tmp_ptr_parameter_end = tmp_ptr_parameter_it + tmp_number_block_units +
                                 tmp_number_block_units;
         tmp_ptr_parameter_it != tmp_ptr_parameter_end;
         ++tmp_ptr_parameter_it) {
      *tmp_ptr_parameter_it = bias;
    }
    // |END| Output gate. |END|
  }
}

void Model::weights_initialize_uniform(var *ptr_array_weights_received,
                                var const *const ptr_last_weight_received,
    real const lower_bound, real const upper_bound) {
  this->real_gen.range(lower_bound, upper_bound);

  for (; ptr_array_weights_received != ptr_last_weight_received;
       ++ptr_array_weights_received) {
    *ptr_array_weights_received = this->real_gen();
  }
}

void Model::lstm_initialize_uniform(real const lower_bound[5],
                                      real const upper_bound[5],
    Layer const *const layer_it) {
  BlockUnit const *tmp_ptr_last_block_unit(
      layer_it->ptr_last_block_unit);
  BlockUnit *tmp_ptr_block_unit_it(
      layer_it->ptr_array_block_units);

  CellUnit const *tmp_ptr_last_cell_unit;
  CellUnit *tmp_ptr_cell_unit_it;

  size_t const tmp_number_peephole_connections(
      tmp_ptr_block_unit_it->last_index_peephole_input_gate -
      tmp_ptr_block_unit_it->first_index_peephole_input_gate),
      tmp_number_feedforward_connections(
          tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate -
          tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate),
      tmp_number_recurrent_connections(
          tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate -
          tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate);
  size_t tmp_connection_index;

  var *tmp_ptr_array_parameters;

  // Loop through each blocks.
  for (; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit;
       ++tmp_ptr_block_unit_it) {
    // Loop through each cells.
    for (tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
        tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units;
         tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit;
         ++tmp_ptr_cell_unit_it) {
      // Input, cell.
      this->real_gen.range(lower_bound[0],
                                       upper_bound[0]);

      tmp_ptr_array_parameters =
          this->ptr_array_parameters +
          tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

      for (tmp_connection_index = 0_UZ;
           tmp_connection_index != tmp_number_feedforward_connections;
           ++tmp_connection_index) {
        tmp_ptr_array_parameters[tmp_connection_index] =
            this->real_gen();
      }
      // |END| Input, cell. |END|

      // Recurrent, cell.
      this->real_gen.range(lower_bound[2],
                                       upper_bound[2]);

      tmp_ptr_array_parameters =
          this->ptr_array_parameters +
          tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;

      for (tmp_connection_index = 0_UZ;
           tmp_connection_index != tmp_number_recurrent_connections;
           ++tmp_connection_index) {
        tmp_ptr_array_parameters[tmp_connection_index] =
            this->real_gen();
      }
      // |END| Recurrent, cell. |END|
    }

    // Input, gates.
    this->real_gen.range(lower_bound[1],
                                     upper_bound[1]);

    //  Input gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_feedforward_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->real_gen();
    }
    //  |END| Input gate. |END|

    //  Forget gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_feedforward_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->real_gen();
    }
    //  |END| Forget gate. |END|

    //  Output gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_feedforward_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->real_gen();
    }
    //  |END| Output gate. |END|
    // |END| Input, gates. |END|

    // Recurrent, gates.
    this->real_gen.range(lower_bound[3],
                                     upper_bound[3]);

    //  Input gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_recurrent_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->real_gen();
    }
    //  |END| Input gate. |END|

    //  Forget gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_recurrent_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->real_gen();
    }
    //  |END| Forget gate. |END|

    //  Output gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_recurrent_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->real_gen();
    }
    //  |END| Output gate. |END|
    // |END| Recurrent, gates. |END|

#ifndef NO_PEEPHOLE
    this->real_gen.range(lower_bound[4],
                                     upper_bound[4]);

    // Peepholes.
    //  Input gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_peephole_input_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_peephole_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->real_gen();
    }
    //  |END| Input gate. |END|

    //  Forget gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_peephole_forget_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_peephole_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->real_gen();
    }
    //  |END| Forget gate. |END|

    //  Output gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_peephole_output_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_peephole_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->real_gen();
    }
    //  |END| Output gate. |END|
    // |END| Peepholes. |END|
#endif
  }
}

void Model::indrec_initialize_uniform(
    Layer const *const layer_it) {
  AF_Ind_recurrent_unit const
      *const tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit(
          layer_it->ptr_array_AF_Ind_recurrent_units);

  size_t const tmp_number_units(static_cast<size_t>(
      layer_it->ptr_last_AF_Ind_recurrent_unit -
      tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit));

  var *tmp_ptr_weight_it(this->ptr_array_parameters +
                         *tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit
                              ->ptr_recurrent_connection_index);
  real tmp_lower_bound, tmp_upper_bound;
  var const *const tmp_ptr_weight_end(tmp_ptr_weight_it + tmp_number_units);
      real tmp_MAG(Math::clip(this->clip_gradient, 2_r, 10_r));

  switch (layer_it->type_activation) {
    case LAYER_ACTIVATION::SYMMETRIC:
      tmp_lower_bound = -pow(
          tmp_MAG / pow(0.9_r, static_cast<real>(this->seq_w) / 10_r),
               1_r / static_cast<real>(this->seq_w));
      tmp_upper_bound = pow(
          tmp_MAG / pow(0.9_r, static_cast<real>(this->seq_w) / 10_r),
              1_r / static_cast<real>(this->seq_w));
      break;
    case LAYER_ACTIVATION::ASYMMETRIC:
    case LAYER_ACTIVATION::SOFTMAX:
    case LAYER_ACTIVATION::RECTIFIER:
    case LAYER_ACTIVATION::SELF_NORMALIZATION:
      tmp_lower_bound = 0_r;
      tmp_upper_bound = pow(tmp_MAG, 1_r / static_cast<real>(this->seq_w));
      break;
    default:
      ERR(
          L"Layer activation type (%d | %ls) is not managed in "
          L"the switch.",
          layer_it->type_activation,
          LAYER_ACTIVATION_NAME[layer_it->type_activation]
              .c_str());
      return;
  }

  this->real_gen.range(-tmp_lower_bound, tmp_upper_bound);

  for (; tmp_ptr_weight_it != tmp_ptr_weight_end; ++tmp_ptr_weight_it)
    *tmp_ptr_weight_it = this->real_gen();
}

void Model::indrec_initialize_uniform_ltm(void) {
  real const tmp_MAG(Math::clip(this->clip_gradient, 2_r, 10_r));
  real tmp_lower_bound, tmp_upper_bound;

  Layer *layer_it;

  // Loop though each layer (backward).
  for (layer_it = this->ptr_last_layer - 2;
       layer_it > this->ptr_array_layers; --layer_it) {
    if (layer_it->type_layer ==
        LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT) {
      switch (layer_it->type_activation) {
        case LAYER_ACTIVATION::SYMMETRIC:
          tmp_lower_bound =
              pow(1_r / tmp_MAG /
                      pow(0.9_r, static_cast<real>(this->seq_w) / 10_r),
              1_r / static_cast<real>(this->seq_w));
          tmp_upper_bound =
              pow(tmp_MAG /
                      pow(0.9_r, static_cast<real>(this->seq_w) / 10_r),
                  1_r / static_cast<real>(this->seq_w));
          break;
        case LAYER_ACTIVATION::ASYMMETRIC:
        case LAYER_ACTIVATION::SOFTMAX:
        case LAYER_ACTIVATION::RECTIFIER:
        case LAYER_ACTIVATION::SELF_NORMALIZATION:
          tmp_lower_bound =
              pow(1_r / tmp_MAG, 1_r / static_cast<real>(this->seq_w));
          tmp_upper_bound = pow(tmp_MAG, 1_r / static_cast<real>(this->seq_w));
          break;
        default:
          ERR(
              L"Layer activation type (%d | %ls) is not managed "
              L"in the switch.",
              layer_it->type_activation,
              LAYER_ACTIVATION_NAME[layer_it->type_activation]
                  .c_str());
          return;
      }

      AF_Ind_recurrent_unit const
          *const tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit(
              layer_it->ptr_array_AF_Ind_recurrent_units);

      size_t const tmp_number_units(static_cast<size_t>(
          layer_it->ptr_last_AF_Ind_recurrent_unit -
          tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit));

      var *tmp_ptr_array_weights(
          this->ptr_array_parameters +
          *tmp_ptr_layer_it_ptr_first_AF_Ind_recurrent_unit
               ->ptr_recurrent_connection_index);
      var const *const tmp_ptr_last_weight(tmp_ptr_array_weights +
                                          tmp_number_units);

      this->real_gen.range(tmp_lower_bound, tmp_upper_bound);

      // Recurrent connection(s).
      do {
        *tmp_ptr_array_weights = this->real_gen();
      } while (++tmp_ptr_array_weights != tmp_ptr_last_weight);

      break;
    }
  }
}

void Model::weights_initialize_gaussian(var *ptr_array_weights_received,
                                 var const *const ptr_last_weight_received,
                                 real const variance) {
  this->gaussian.range(0_r, variance);

  for (; ptr_array_weights_received != ptr_last_weight_received;
       ++ptr_array_weights_received)
    *ptr_array_weights_received = this->gaussian();
}

void Model::lstm_initialize_gaussian(
    real const fwp_cell_var,
    real const fwp_gate_var,
    real const rec_cell_var,
    real const rec_gate_var,
    real const phl_gate_var,
    Layer *const layer_it) {
  BlockUnit const *tmp_ptr_last_block_unit(
      layer_it->ptr_last_block_unit);
  BlockUnit *tmp_ptr_block_unit_it(
      layer_it->ptr_array_block_units);

  CellUnit const *tmp_ptr_last_cell_unit;
  CellUnit *tmp_ptr_cell_unit_it;

  size_t const tmp_number_peephole_connections(
      tmp_ptr_block_unit_it->last_index_peephole_input_gate -
      tmp_ptr_block_unit_it->first_index_peephole_input_gate),
      tmp_number_feedforward_connections(
          tmp_ptr_block_unit_it->last_index_feedforward_connection_input_gate -
          tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate),
      tmp_number_recurrent_connections(
          tmp_ptr_block_unit_it->last_index_recurrent_connection_input_gate -
          tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate);
  size_t tmp_connection_index;

  var *tmp_ptr_array_parameters;

  // Loop through each blocks.
  for (; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit;
       ++tmp_ptr_block_unit_it) {
    // Loop through each cells.
    for (tmp_ptr_last_cell_unit = tmp_ptr_block_unit_it->ptr_last_cell_unit,
        tmp_ptr_cell_unit_it = tmp_ptr_block_unit_it->ptr_array_cell_units;
         tmp_ptr_cell_unit_it != tmp_ptr_last_cell_unit;
         ++tmp_ptr_cell_unit_it) {
      // Input, cell.
      tmp_ptr_array_parameters =
          this->ptr_array_parameters +
          tmp_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

      this->gaussian.range(0_r,
                                           fwp_cell_var);

      for (tmp_connection_index = 0_UZ;
           tmp_connection_index != tmp_number_feedforward_connections;
           ++tmp_connection_index) {
        tmp_ptr_array_parameters[tmp_connection_index] =
            this->gaussian();
      }
      // |END| Input, cell. |END|

      // Recurrent, cell.
      tmp_ptr_array_parameters =
          this->ptr_array_parameters +
          tmp_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;

      this->gaussian.range(0_r,
                                           rec_cell_var);

      for (tmp_connection_index = 0_UZ;
           tmp_connection_index != tmp_number_recurrent_connections;
           ++tmp_connection_index) {
        tmp_ptr_array_parameters[tmp_connection_index] =
            this->gaussian();
      }
      // |END| Recurrent, cell. |END|
    }

    // Input, gates.
    this->gaussian.range(0_r,
                                         fwp_gate_var);

    //  Input gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_feedforward_connection_input_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_feedforward_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->gaussian();
    }
    //  |END| Input gate. |END|

    //  Forget gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_feedforward_connection_forget_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_feedforward_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->gaussian();
    }
    //  |END| Forget gate. |END|

    //  Output gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_feedforward_connection_output_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_feedforward_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->gaussian();
    }
    //  |END| Output gate. |END|
    // |END| Input, gates. |END|

    // Recurrent, gates.
    this->gaussian.range(0_r,
                                         rec_gate_var);

    //  Input gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_recurrent_connection_input_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_recurrent_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->gaussian();
    }
    //  |END| Input gate. |END|

    //  Forget gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_recurrent_connection_forget_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_recurrent_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->gaussian();
    }
    //  |END| Forget gate. |END|

    //  Output gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_recurrent_connection_output_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_recurrent_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->gaussian();
    }
    //  |END| Output gate. |END|
    // |END| Recurrent, gates. |END|

#ifndef NO_PEEPHOLE
    // Peepholes.
    this->gaussian.range(0_r, phl_gate_var);

    //  Input gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_peephole_input_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_peephole_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->gaussian();
    }
    //  |END| Input gate. |END|

    //  Forget gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_peephole_forget_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_peephole_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->gaussian();
    }
    //  |END| Forget gate. |END|

    //  Output gate.
    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_unit_it->first_index_peephole_output_gate;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_peephole_connections;
         ++tmp_connection_index) {
      tmp_ptr_array_parameters[tmp_connection_index] =
          this->gaussian();
    }
    //  |END| Output gate. |END|
    // |END| Peepholes. |END|
#endif
  }
}
}  // namespace DL
