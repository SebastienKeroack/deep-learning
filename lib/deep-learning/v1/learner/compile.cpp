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
#include "deep-learning/io/logger.hpp"

namespace DL::v1 {
bool Model::Assign__Layers(
    Layer_Parameters const *const ptr_array_layers_received) {
  Layer const *const last_layer(this->ptr_last_layer);
  Layer *layer_it(this->ptr_array_layers);

  for (; layer_it != last_layer; ++layer_it)
    if (this->Assign__Layer(layer_it, ptr_array_layers_received) == false) {
      ERR(L"An error has been triggered from the "
          L"`Assign__Layer()` function.");
      return false;
    }

  return true;
}

bool Model::Assign__Layer(
    Layer *&layer_it, Layer_Parameters const *const ptr_array_layers_received) {
  size_t const tmp_layer_index(
      static_cast<size_t>(layer_it - this->ptr_array_layers));

  layer_it->type_layer = ptr_array_layers_received[tmp_layer_index].type_layer;

  layer_it->use_bidirectional =
      ptr_array_layers_received[tmp_layer_index].use_bidirectional;

  switch (layer_it->type_layer) {
    case LAYER::AVERAGE_POOLING:
    case LAYER::MAX_POOLING:
      layer_it->pooling_values[0] = ptr_array_layers_received[tmp_layer_index]
                                        .unit_parameters[0];  // Kernel size.
      layer_it->pooling_values[1] = ptr_array_layers_received[tmp_layer_index]
                                        .unit_parameters[1];  // Stride.
      layer_it->pooling_values[2] = ptr_array_layers_received[tmp_layer_index]
                                        .unit_parameters[2];  // Padding.
      layer_it->pooling_values[3] = ptr_array_layers_received[tmp_layer_index]
                                        .unit_parameters[3];  // Dilation.
      layer_it->pooling_values[4] = ptr_array_layers_received[tmp_layer_index]
                                        .unit_parameters[4];  // Ceil mode.
      break;
    case LAYER::FULLY_CONNECTED:
    case LAYER::FULLY_CONNECTED_RECURRENT:
      *layer_it->ptr_number_outputs =
          ptr_array_layers_received[tmp_layer_index].unit_parameters[0];

      layer_it->ptr_last_neuron_unit =
          layer_it->ptr_array_neuron_units + *layer_it->ptr_number_outputs;
      layer_it->ptr_last_AF_unit =
          layer_it->ptr_array_AF_units + *layer_it->ptr_number_outputs;

      this->total_neuron_units += static_cast<size_t>(
          layer_it->ptr_last_neuron_unit - layer_it->ptr_array_neuron_units);
      this->total_AF_units += static_cast<size_t>(layer_it->ptr_last_AF_unit -
                                                  layer_it->ptr_array_AF_units);
      break;
    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
      *layer_it->ptr_number_outputs =
          ptr_array_layers_received[tmp_layer_index].unit_parameters[0];

      layer_it->ptr_last_neuron_unit =
          layer_it->ptr_array_neuron_units + *layer_it->ptr_number_outputs;
      layer_it->ptr_last_AF_Ind_recurrent_unit =
          layer_it->ptr_array_AF_Ind_recurrent_units +
          *layer_it->ptr_number_outputs;

      this->total_neuron_units += static_cast<size_t>(
          layer_it->ptr_last_neuron_unit - layer_it->ptr_array_neuron_units);
      this->total_AF_Ind_recurrent_units +=
          static_cast<size_t>(layer_it->ptr_last_AF_Ind_recurrent_unit -
                              layer_it->ptr_array_AF_Ind_recurrent_units);
      break;
    case LAYER::LSTM:
      *layer_it->ptr_number_outputs =
          ptr_array_layers_received[tmp_layer_index].unit_parameters[0] *
          ptr_array_layers_received[tmp_layer_index].unit_parameters[1];

      if (ptr_array_layers_received[tmp_layer_index].use_bidirectional) {
        layer_it->ptr_last_block_unit =
            layer_it->ptr_array_block_units +
            ptr_array_layers_received[tmp_layer_index].unit_parameters[0] * 2u;
        layer_it->ptr_last_cell_unit =
            layer_it->ptr_array_cell_units +
            ptr_array_layers_received[tmp_layer_index].unit_parameters[0] *
                ptr_array_layers_received[tmp_layer_index].unit_parameters[1] *
                2u;
      } else {
        layer_it->ptr_last_block_unit =
            layer_it->ptr_array_block_units +
            ptr_array_layers_received[tmp_layer_index].unit_parameters[0];
        layer_it->ptr_last_cell_unit =
            layer_it->ptr_array_cell_units +
            ptr_array_layers_received[tmp_layer_index].unit_parameters[0] *
                ptr_array_layers_received[tmp_layer_index].unit_parameters[1];
      }

      this->total_block_units += static_cast<size_t>(
          layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units);
      this->total_cell_units += static_cast<size_t>(
          layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units);
      break;
    case LAYER::RESIDUAL:
      layer_it->block_depth =
          ptr_array_layers_received[tmp_layer_index].unit_parameters[0];

      if (this->Assign__Residual_Block(layer_it, ptr_array_layers_received) ==
          false) {
        ERR(L"An error has been triggered from the "
            L"`Assign__Residual_Block()` function.");
        return false;
      }
      break;
    default:
      ERR(L"Layer type (%d | %ls) is not managed in", layer_it->type_layer,
          LAYER_NAME[layer_it->type_layer].c_str());
      return false;
  }

  return true;
}

bool Model::Assign__Residual_Block(
    Layer *&layer_it, Layer_Parameters const *const ptr_array_layers_received) {
  if (layer_it->type_layer != LAYER::RESIDUAL) {
    ERR(L"Layer received as argument is not a residual unit.");
    return false;
  }

  Layer const *const tmp_ptr_residual_block_end(layer_it +
                                                layer_it->block_depth + 1);

  for (++layer_it; layer_it != tmp_ptr_residual_block_end; ++layer_it) {
    if (this->Assign__Residual_Layer(layer_it, ptr_array_layers_received) ==
        false) {
      ERR(L"An error has been triggered from the "
          L"`Assign__Residual_Layer()` function.");
      return false;
    }
  }

  // Assign layer iterator to the last layer inside the block.
  --layer_it;

  return true;
}

bool Model::Assign__Residual_Layer(
    Layer *&layer_it, Layer_Parameters const *const ptr_array_layers_received) {
  size_t const tmp_layer_index(
      static_cast<size_t>(layer_it - this->ptr_array_layers));

  layer_it->type_layer = ptr_array_layers_received[tmp_layer_index].type_layer;

  layer_it->use_bidirectional =
      ptr_array_layers_received[tmp_layer_index].use_bidirectional;

  switch (layer_it->type_layer) {
    case LAYER::AVERAGE_POOLING:
    case LAYER::MAX_POOLING:
      layer_it->pooling_values[0] = ptr_array_layers_received[tmp_layer_index]
                                        .unit_parameters[0];  // Kernel size.
      layer_it->pooling_values[1] = ptr_array_layers_received[tmp_layer_index]
                                        .unit_parameters[1];  // Stride.
      layer_it->pooling_values[2] = ptr_array_layers_received[tmp_layer_index]
                                        .unit_parameters[2];  // Padding.
      layer_it->pooling_values[3] = ptr_array_layers_received[tmp_layer_index]
                                        .unit_parameters[3];  // Dilation.
      layer_it->pooling_values[4] = ptr_array_layers_received[tmp_layer_index]
                                        .unit_parameters[4];  // Ceil mode.
      break;
    case LAYER::FULLY_CONNECTED:
    case LAYER::FULLY_CONNECTED_RECURRENT:
    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
      *layer_it->ptr_number_outputs =
          ptr_array_layers_received[tmp_layer_index].unit_parameters[0];

      layer_it->ptr_last_neuron_unit =
          layer_it->ptr_array_neuron_units + *layer_it->ptr_number_outputs;

      this->total_neuron_units += static_cast<size_t>(
          layer_it->ptr_last_neuron_unit - layer_it->ptr_array_neuron_units);
      break;
    case LAYER::LSTM:
      *layer_it->ptr_number_outputs =
          ptr_array_layers_received[tmp_layer_index].unit_parameters[0] *
          ptr_array_layers_received[tmp_layer_index].unit_parameters[1];

      if (ptr_array_layers_received[tmp_layer_index].use_bidirectional) {
        layer_it->ptr_last_block_unit =
            layer_it->ptr_array_block_units +
            ptr_array_layers_received[tmp_layer_index].unit_parameters[0] * 2u;
        layer_it->ptr_last_cell_unit =
            layer_it->ptr_array_cell_units +
            ptr_array_layers_received[tmp_layer_index].unit_parameters[0] *
                ptr_array_layers_received[tmp_layer_index].unit_parameters[1] *
                2u;
      } else {
        layer_it->ptr_last_block_unit =
            layer_it->ptr_array_block_units +
            ptr_array_layers_received[tmp_layer_index].unit_parameters[0];
        layer_it->ptr_last_cell_unit =
            layer_it->ptr_array_cell_units +
            ptr_array_layers_received[tmp_layer_index].unit_parameters[0] *
                ptr_array_layers_received[tmp_layer_index].unit_parameters[1];
      }

      this->total_block_units += static_cast<size_t>(
          layer_it->ptr_last_block_unit - layer_it->ptr_array_block_units);
      this->total_cell_units += static_cast<size_t>(
          layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units);
      break;
    case LAYER::RESIDUAL:
      layer_it->block_depth =
          ptr_array_layers_received[tmp_layer_index].unit_parameters[0];

      if (this->Assign__Residual_Block(layer_it, ptr_array_layers_received) ==
          false) {
        ERR(L"An error has been triggered from the "
            L"`Assign__Residual_Block()` function.");
        return false;
      }
      break;
    default:
      ERR(L"Layer type (%d | %ls) is not managed in", layer_it->type_layer,
          LAYER_NAME[layer_it->type_layer].c_str());
      return false;
  }

  return true;
}

bool Model::Assign__Post__Layers(void) {
  Layer const *const last_layer(this->ptr_last_layer -
                                1);             // Subtract output layer.
  Layer *layer_it(this->ptr_array_layers + 1);  // Skip input layer.

  for (; layer_it != last_layer; ++layer_it)
    if (this->Assign__Post__Layer(layer_it) == false) {
      ERR(L"An error has been triggered from the "
          L"`Assign__Post__Layer()` function.");
      return false;
    }

  return true;
}

bool Model::Assign__Post__Layer(Layer *&layer_it) {
  Layer const *const tmp_ptr_previous_layer_connected(
      layer_it->previous_connected_layers[0]),
      *tmp_ptr_residual_block_last_layer;

  switch (layer_it->type_layer) {
    case LAYER::AVERAGE_POOLING:
      // Output_Size = floor((Input_Size + 2 * Padding - Kernel) / Stride + 1)
      if (layer_it->pooling_values[4] == 0_UZ) {
        *layer_it->ptr_number_outputs = static_cast<size_t>(
            floor(static_cast<double>(
                      *tmp_ptr_previous_layer_connected->ptr_number_outputs +
                      2_UZ * layer_it->pooling_values[2] -
                      layer_it->pooling_values[0]) /
                      static_cast<double>(layer_it->pooling_values[1]) +
                  1.0));
      } else {
        *layer_it->ptr_number_outputs = static_cast<size_t>(
            ceil(static_cast<double>(
                     *tmp_ptr_previous_layer_connected->ptr_number_outputs +
                     2_UZ * layer_it->pooling_values[2] -
                     layer_it->pooling_values[0]) /
                     static_cast<double>(layer_it->pooling_values[1]) +
                 1.0));
      }

      layer_it->ptr_last_basic_unit =
          layer_it->ptr_array_basic_units + *layer_it->ptr_number_outputs;

      this->total_basic_units += static_cast<size_t>(
          layer_it->ptr_last_basic_unit - layer_it->ptr_array_basic_units);
      break;
    case LAYER::MAX_POOLING:
      // Output_Size = floor((Input_Size + 2 * Padding - Dilation * (Kernel - 1)
      // - 1) / Stride + 1)
      if (layer_it->pooling_values[4] == 0_UZ) {
        *layer_it->ptr_number_outputs = static_cast<size_t>(
            floor(static_cast<double>(
                      *tmp_ptr_previous_layer_connected->ptr_number_outputs +
                      2_UZ * layer_it->pooling_values[2] -
                      layer_it->pooling_values[3] *
                          (layer_it->pooling_values[0] - 1_UZ) -
                      1_UZ) /
                      static_cast<double>(layer_it->pooling_values[1]) +
                  1.0));
      } else {
        *layer_it->ptr_number_outputs = static_cast<size_t>(
            ceil(static_cast<double>(
                     *tmp_ptr_previous_layer_connected->ptr_number_outputs +
                     2_UZ * layer_it->pooling_values[2] -
                     layer_it->pooling_values[3] *
                         (layer_it->pooling_values[0] - 1_UZ) -
                     1_UZ) /
                     static_cast<double>(layer_it->pooling_values[1]) +
                 1.0));
      }

      layer_it->ptr_last_basic_indice_unit =
          layer_it->ptr_array_basic_indice_units +
          *layer_it->ptr_number_outputs;

      this->total_basic_indice_units +=
          static_cast<size_t>(layer_it->ptr_last_basic_indice_unit -
                              layer_it->ptr_array_basic_indice_units);
      break;
    case LAYER::RESIDUAL:
      tmp_ptr_residual_block_last_layer = layer_it + layer_it->block_depth;

      *layer_it->ptr_number_outputs = std::max<size_t>(
          *tmp_ptr_previous_layer_connected->ptr_number_outputs,
          *tmp_ptr_residual_block_last_layer->ptr_number_outputs);

      layer_it->pooling_values[2] =
          (static_cast<size_t>(std::abs(
               static_cast<long long int>(
                   *tmp_ptr_previous_layer_connected->ptr_number_outputs) -
               static_cast<long long int>(
                   *tmp_ptr_residual_block_last_layer->ptr_number_outputs))) +
           1_UZ) /
          2_UZ;  // Padding.

      layer_it->ptr_last_basic_unit =
          layer_it->ptr_array_basic_units + *layer_it->ptr_number_outputs;

      this->total_basic_units += static_cast<size_t>(
          layer_it->ptr_last_basic_unit - layer_it->ptr_array_basic_units);

      if (this->Assign__Post__Residual_Block(layer_it) == false) {
        ERR(L"An error has been triggered from the "
            L"`Assign__Post__Residual_Block()` function.");
        return false;
      }
      break;
    default:
      break;
  }

  return true;
}

bool Model::Assign__Post__Residual_Block(Layer *&layer_it) {
  if (layer_it->type_layer != LAYER::RESIDUAL) {
    ERR(L"Layer received as argument is not a residual unit.");
    return false;
  }

  Layer const *const tmp_ptr_residual_block_end(layer_it +
                                                layer_it->block_depth + 1);

  // First block layer.
  if (this->Assign__Post__Residual_Layer(true, ++layer_it) == false) {
    ERR(L"An error has been triggered from the "
        L"`Assign__Post__Residual_Layer(true)` function.");
    return false;
  }
  // |END| First block layer. |END|

  // Remaining layer(s).
  for (++layer_it; layer_it != tmp_ptr_residual_block_end; ++layer_it) {
    if (this->Assign__Post__Residual_Layer(false, layer_it) == false) {
      ERR(L"An error has been triggered from the "
          L"`Assign__Post__Residual_Layer(false)` function.");
      return false;
    }
  }
  // |END| Remaining layer(s). |END|

  // Assign layer iterator to the last layer inside the block.
  --layer_it;

  return true;
}

bool Model::Assign__Post__Residual_Layer(
    bool const is_block_input_layer_received, Layer *&layer_it) {
  Layer const *const tmp_ptr_previous_layer_connected(
      layer_it->previous_connected_layers[0]),
      *tmp_ptr_residual_block_last_layer;

  switch (layer_it->type_layer) {
    case LAYER::AVERAGE_POOLING:
      // Output_Size = floor((Input_Size + 2 * Padding - Kernel) / Stride + 1)
      if (layer_it->pooling_values[4] == 0_UZ) {
        *layer_it->ptr_number_outputs = static_cast<size_t>(
            floor(static_cast<double>(
                      *tmp_ptr_previous_layer_connected->ptr_number_outputs +
                      2_UZ * layer_it->pooling_values[2] -
                      layer_it->pooling_values[0]) /
                      static_cast<double>(layer_it->pooling_values[1]) +
                  1.0));
      } else {
        *layer_it->ptr_number_outputs = static_cast<size_t>(
            ceil(static_cast<double>(
                     *tmp_ptr_previous_layer_connected->ptr_number_outputs +
                     2_UZ * layer_it->pooling_values[2] -
                     layer_it->pooling_values[0]) /
                     static_cast<double>(layer_it->pooling_values[1]) +
                 1.0));
      }

      layer_it->ptr_last_basic_unit =
          layer_it->ptr_array_basic_units + *layer_it->ptr_number_outputs;

      this->total_basic_units += static_cast<size_t>(
          layer_it->ptr_last_basic_unit - layer_it->ptr_array_basic_units);
      break;
    case LAYER::MAX_POOLING:
      // Output_Size = floor((Input_Size + 2 * Padding - Dilation * (Kernel - 1)
      // - 1) / Stride + 1)
      if (layer_it->pooling_values[4] == 0_UZ) {
        *layer_it->ptr_number_outputs = static_cast<size_t>(
            floor(static_cast<double>(
                      *tmp_ptr_previous_layer_connected->ptr_number_outputs +
                      2_UZ * layer_it->pooling_values[2] -
                      layer_it->pooling_values[3] *
                          (layer_it->pooling_values[0] - 1_UZ) -
                      1_UZ) /
                      static_cast<double>(layer_it->pooling_values[1]) +
                  1.0));
      } else {
        *layer_it->ptr_number_outputs = static_cast<size_t>(
            ceil(static_cast<double>(
                     *tmp_ptr_previous_layer_connected->ptr_number_outputs +
                     2_UZ * layer_it->pooling_values[2] -
                     layer_it->pooling_values[3] *
                         (layer_it->pooling_values[0] - 1_UZ) -
                     1_UZ) /
                     static_cast<double>(layer_it->pooling_values[1]) +
                 1.0));
      }

      layer_it->ptr_last_basic_indice_unit =
          layer_it->ptr_array_basic_indice_units +
          *layer_it->ptr_number_outputs;

      this->total_basic_indice_units +=
          static_cast<size_t>(layer_it->ptr_last_basic_indice_unit -
                              layer_it->ptr_array_basic_indice_units);
      break;
    case LAYER::FULLY_CONNECTED:
    case LAYER::FULLY_CONNECTED_RECURRENT:
      if (is_block_input_layer_received == false) {
        layer_it->ptr_last_AF_unit =
            layer_it->ptr_array_AF_units +
            *tmp_ptr_previous_layer_connected->ptr_number_outputs;

        this->total_AF_units += static_cast<size_t>(
            layer_it->ptr_last_AF_unit - layer_it->ptr_array_AF_units);
      }
      break;
    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
      if (is_block_input_layer_received == false) {
        layer_it->ptr_last_AF_Ind_recurrent_unit =
            layer_it->ptr_array_AF_Ind_recurrent_units +
            *tmp_ptr_previous_layer_connected->ptr_number_outputs;

        this->total_AF_Ind_recurrent_units +=
            static_cast<size_t>(layer_it->ptr_last_AF_Ind_recurrent_unit -
                                layer_it->ptr_array_AF_Ind_recurrent_units);
      }
      break;
    case LAYER::LSTM:
      break;
    case LAYER::RESIDUAL:
      tmp_ptr_residual_block_last_layer = layer_it + layer_it->block_depth;

      *layer_it->ptr_number_outputs = std::max<size_t>(
          *tmp_ptr_previous_layer_connected->ptr_number_outputs,
          *tmp_ptr_residual_block_last_layer->ptr_number_outputs);

      layer_it->pooling_values[2] =
          (static_cast<size_t>(std::abs(
               static_cast<long long int>(
                   *tmp_ptr_previous_layer_connected->ptr_number_outputs) -
               static_cast<long long int>(
                   *tmp_ptr_residual_block_last_layer->ptr_number_outputs))) +
           1_UZ) /
          2_UZ;  // Padding.

      layer_it->ptr_last_basic_unit =
          layer_it->ptr_array_basic_units + *layer_it->ptr_number_outputs;

      this->total_basic_units += static_cast<size_t>(
          layer_it->ptr_last_basic_unit - layer_it->ptr_array_basic_units);

      if (this->Assign__Post__Residual_Block(layer_it) == false) {
        ERR(L"An error has been triggered from the "
            L"`Assign__Post__Residual_Block()` function.");
        return false;
      }
      break;
    default:
      ERR(L"Layer type (%d | %ls) is not managed in", layer_it->type_layer,
          LAYER_NAME[layer_it->type_layer].c_str());
      return false;
  }

  return true;
}

bool Model::compile(size_t const n_layers,
                    size_t const number_recurrent_depth_received,
                    MODEL::TYPE const type_network_received,
                    Layer_Parameters const *const ptr_array_layers_received,
                    size_t const allowable_memory) {
  if (number_recurrent_depth_received == 0_UZ) {
    ERR(L"Recurrent depth can not be zero.");
    return false;
  } else {
    this->seq_w = number_recurrent_depth_received;
  }

  if (this->Allocate__Structure(n_layers, allowable_memory) == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Structure(%zu, %zu)` function.",
        n_layers, allowable_memory);
    return false;
  } else {
    this->type = type_network_received;
  }

  size_t tmp_fan_in;

  Layer const *const last_layer(this->ptr_last_layer),
      *tmp_ptr_previous_layer_connected;
  Layer *layer_it;

  if (this->Assign__Layers(ptr_array_layers_received) == false) {
    ERR(L"An error has been triggered from the "
        L"`Assign__Layers()` function.");
    return false;
  }

  // Layers, connections.
  this->Order__Layers__Connection();

  if (this->Assign__Post__Layers() == false) {
    ERR(L"An error has been triggered from the "
        L"`Assign__Post__Layers()` function.");
    return false;
  }

  this->n_inp = ptr_array_layers_received[0].unit_parameters[0];
  this->n_out = ptr_array_layers_received[n_layers - 1_UZ].unit_parameters[0];

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
  }

  // Layers, outputs pointers.
  this->Order__Layers__Output();

  // Initialize weight dimension.
  for (layer_it = this->ptr_array_layers + 1; layer_it != last_layer;
       ++layer_it) {
    // If the current layer is a pooling/residual layer, continue.
    if (layer_it->type_layer == LAYER::AVERAGE_POOLING ||
        layer_it->type_layer == LAYER::MAX_POOLING ||
        layer_it->type_layer == LAYER::RESIDUAL) {
      continue;
    }

    tmp_fan_in = *layer_it->previous_connected_layers[0]->ptr_number_outputs;

    switch (layer_it->type_layer) {
      case LAYER::FULLY_CONNECTED:
        this->total_weights +=
            this->Prepare__Connections__FC(tmp_fan_in, layer_it);

        if (this->set_layer_activation_function(layer_it,
                                                ACTIVATION::SIGMOID) == false) {
          ERR(L"An error has been triggered from the "
              L"`set_layer_activation_function(%u)` function.",
              ACTIVATION::SIGMOID);
          return false;
        }
        break;
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        this->total_weights +=
            this->Prepare__Connections__FC_Ind_RNN(tmp_fan_in, layer_it);

        if (this->set_layer_activation_function(layer_it, ACTIVATION::RELU) ==
            false) {
          ERR(L"An error has been triggered from the "
              L"`set_layer_activation_function(%u)` function.",
              ACTIVATION::RELU);
          return false;
        }

        // Regularization on recurrent connection(s) (Independently RNN).
        if (this->Set__Regularization__Constraint_Recurrent_Weight__Default(
                layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Regularization__Constraint_Recurrent_Weight__Default"
              L"(ptr)` function.");
          return false;
        }
        break;
      case LAYER::LSTM:
        this->total_weights +=
            this->Prepare__Connections__LSTM(tmp_fan_in, layer_it);
        break;
      default:
        ERR(L"Layer type (%d | %ls) is not managed in", layer_it->type_layer,
            LAYER_NAME[layer_it->type_layer].c_str());
        return false;
    }
  }

  // Initialize bias dimension.
  for (layer_it = this->ptr_array_layers + 1; layer_it != last_layer;
       ++layer_it) {
    // If the current layer is a pooling/residual layer, continue.
    if (layer_it->type_layer == LAYER::AVERAGE_POOLING ||
        layer_it->type_layer == LAYER::MAX_POOLING ||
        layer_it->type_layer == LAYER::RESIDUAL) {
      continue;
    }

    switch (layer_it->type_layer) {
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
      case LAYER::FULLY_CONNECTED_RECURRENT:
        this->total_bias +=
            this->Prepare__Bias__FC(this->total_weights, layer_it);
        break;
      case LAYER::LSTM:
        this->total_bias +=
            this->Prepare__Bias__LSTM(this->total_weights, layer_it);
        break;
      default:
        ERR(L"Layer type (%d | %ls) is not managed in", layer_it->type_layer,
            LAYER_NAME[layer_it->type_layer].c_str());
        return false;
    }
  }

  this->total_parameters = this->total_weights + this->total_bias;

  if (this->Allocate__Parameter() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Parameter()` function.");
    return false;
  }

  // Initialize connection(s).
  for (layer_it = this->ptr_array_layers + 1; layer_it != last_layer;
       ++layer_it) {
    // If the current layer is a pooling/residual layer, continue.
    if (layer_it->type_layer == LAYER::AVERAGE_POOLING ||
        layer_it->type_layer == LAYER::MAX_POOLING ||
        layer_it->type_layer == LAYER::RESIDUAL) {
      continue;
    }

    tmp_ptr_previous_layer_connected = layer_it->previous_connected_layers[0];

    switch (tmp_ptr_previous_layer_connected->type_layer) {
      case LAYER::AVERAGE_POOLING:
      case LAYER::RESIDUAL:
        switch (layer_it->type_layer) {
          case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          case LAYER::FULLY_CONNECTED:
            this->Initialize_Connections__Basic_unit_to_FC(
                layer_it, tmp_ptr_previous_layer_connected);
            break;
          case LAYER::LSTM:
            this->Initialize_Connections__Basic_unit_to_LSTM(
                layer_it, tmp_ptr_previous_layer_connected);
            break;
          default:
            ERR(L"Layer type (%d | %ls) is not managed in the switch.",
                layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
            return false;
        }
        break;
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_RECURRENT:
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        switch (layer_it->type_layer) {
          case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          case LAYER::FULLY_CONNECTED:
            this->Initialize_Connections__FC_to_FC(
                layer_it, tmp_ptr_previous_layer_connected);
            break;
          case LAYER::LSTM:
            this->Initialize_Connections__FC_to_LSTM(
                layer_it, tmp_ptr_previous_layer_connected);
            break;
          default:
            ERR(L"Layer type (%d | %ls) is not managed in the switch.",
                layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
            return false;
        }
        break;
      case LAYER::LSTM:
        switch (layer_it->type_layer) {
          case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          case LAYER::FULLY_CONNECTED:
            this->Initialize_Connections__LSTM_to_FC(
                layer_it, tmp_ptr_previous_layer_connected);
            break;
          case LAYER::LSTM:
            this->Initialize_Connections__LSTM_to_LSTM(
                layer_it, tmp_ptr_previous_layer_connected);
            break;
          default:
            ERR(L"Layer type (%d | %ls) is not managed in the switch.",
                layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
            return false;
        }
        break;
      case LAYER::MAX_POOLING:
        switch (layer_it->type_layer) {
          case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          case LAYER::FULLY_CONNECTED:
            this->Initialize_Connections__Basic_indice_unit_to_FC(
                layer_it, tmp_ptr_previous_layer_connected);
            break;
          case LAYER::LSTM:
            this->Initialize_Connections__Basic_indice_unit_to_LSTM(
                layer_it, tmp_ptr_previous_layer_connected);
            break;
          default:
            ERR(L"Layer type (%d | %ls) is not managed in the switch.",
                layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
            return false;
        }
        break;
      default:
        ERR(L"Layer type (%d | %ls) is not managed in the switch.",
            tmp_ptr_previous_layer_connected->type_layer,
            LAYER_NAME[tmp_ptr_previous_layer_connected->type_layer].c_str());
        return false;
    }

    switch (layer_it->type_layer) {
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        this->Initialize_Connections__AF_Ind_Recurrent(layer_it);
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_RECURRENT:
        this->Initialize_Connections__Bias(layer_it);
        break;
      case LAYER::LSTM:
        this->Initialize_Connections__LSTM__Bias(layer_it);
        break;
      default:
        ERR(L"Layer type (%d | %ls) is not managed in the switch.",
            layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
        return false;
    }
  }

  return true;
}
}  // namespace DL::v1