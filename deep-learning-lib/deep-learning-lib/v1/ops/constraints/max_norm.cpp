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

#include "deep-learning-lib/io/logger.hpp"
#include "deep-learning-lib/v1/learner/model.hpp"

#include <omp.h>

namespace DL::v1 {
real Model::Get__Regularization__Max_Norm_Constraints(void) const {
  return (this->regularization__max_norm_constraints);
}

bool Model::Set__Regularization__Max_Norm_Constraints(
    real const regularization__max_norm_constraints_received) {
  if (this->regularization__max_norm_constraints ==
      regularization__max_norm_constraints_received) {
    return true;
  } else if (regularization__max_norm_constraints_received < 0_r) {
    ERR(L"Max-norm constraints (%f) less than zero.",
        cast(regularization__max_norm_constraints_received));

    return false;
  }

  this->regularization__max_norm_constraints =
      regularization__max_norm_constraints_received;

#ifdef COMPILE_CUDA
  if (this->is_cu_initialized) {
    this->cumodel->Set__Regularization__Max_Norm_Constraints(
        regularization__max_norm_constraints_received);
  }
#endif

  return true;
}

void Model::Update_Weight_Regularization__Max_Norm_Constraints(
    size_t const str, size_t const end) {
  if (this->use_mp && this->is_mp_initialized)
    this->Update_Weight_Regularization__Max_Norm_Constraints__OpenMP(str, end);
  else
    this->Update_Weight_Regularization__Max_Norm_Constraints__Loop(str, end);
}

void Model::Update_Weight_Regularization__Max_Norm_Constraints__Neurons__Loop(
    size_t const str, size_t const end, Layer const *const layer,
    Layer const *const last_layer) {
  Neuron_unit const *const unit_end(last_layer->ptr_last_neuron_unit),
      *unit_it(layer->ptr_array_neuron_units);

  for (; unit_it != unit_end; ++unit_it) {
    if (*unit_it->ptr_first_connection_index < str)
      continue;
    else if (*unit_it->ptr_last_connection_index > end)
      break;

    euclidean_norm_st(
        0_UZ, *unit_it->ptr_number_connections,
        this->regularization__max_norm_constraints,
        this->ptr_array_parameters + *unit_it->ptr_first_connection_index);
  }
}

void Model::Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__Loop(
    size_t const str, size_t const end, Layer const *const layer,
    Layer const *const last_layer) {
  BlockUnit const *const block_end(last_layer->ptr_last_block_unit),
      *block_it(layer->ptr_array_block_units);

  CellUnit const *cell_end, *cell_it;

  for (; block_it != block_end; ++block_it) {
    // [0] Cell input.
    for (cell_end = block_it->ptr_last_cell_unit,
        cell_it = block_it->ptr_array_cell_units;
         cell_it != cell_end; ++cell_it) {
      //    [1] Input.
      if (cell_it->first_index_feedforward_connection_cell_input > str) {
        if (cell_it->last_index_feedforward_connection_cell_input > end) {
          break;
        }

        euclidean_norm_st(
            0_UZ,
            cell_it->last_index_feedforward_connection_cell_input -
                cell_it->first_index_feedforward_connection_cell_input,
            this->regularization__max_norm_constraints,
            this->ptr_array_parameters +
                cell_it->first_index_feedforward_connection_cell_input);
      }
      //    [1] |END| Input. |END|

      //    [1] Recurrent.
      if (cell_it->first_index_recurrent_connection_cell_input > str) {
        if (cell_it->last_index_recurrent_connection_cell_input > end) {
          break;
        }

        euclidean_norm_st(
            0_UZ,
            cell_it->last_index_recurrent_connection_cell_input -
                cell_it->first_index_recurrent_connection_cell_input,
            this->regularization__max_norm_constraints,
            this->ptr_array_parameters +
                cell_it->first_index_recurrent_connection_cell_input);
      }
      //    [1] |END| Recurrent. |END|
    }
    // [0] |END| Cell input. |END|

    // [0] Gates.
    //    [1] Input.
    if (block_it->first_index_feedforward_connection_input_gate > str) {
      if (block_it->last_index_feedforward_connection_input_gate > end) {
        break;
      }

      euclidean_norm_st(
          0_UZ,
          block_it->last_index_feedforward_connection_input_gate -
              block_it->first_index_feedforward_connection_input_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              block_it->first_index_feedforward_connection_input_gate);
    }

    if (block_it->first_index_feedforward_connection_forget_gate > str) {
      if (block_it->last_index_feedforward_connection_forget_gate > end) {
        break;
      }

      euclidean_norm_st(
          0_UZ,
          block_it->last_index_feedforward_connection_forget_gate -
              block_it->first_index_feedforward_connection_forget_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              block_it->first_index_feedforward_connection_forget_gate);
    }

    if (block_it->first_index_feedforward_connection_output_gate > str) {
      if (block_it->last_index_feedforward_connection_output_gate > end) {
        break;
      }

      euclidean_norm_st(
          0_UZ,
          block_it->last_index_feedforward_connection_output_gate -
              block_it->first_index_feedforward_connection_output_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              block_it->first_index_feedforward_connection_output_gate);
    }
    //    [1] |END| Input. |END|

    //    [1] Recurrent.
    if (block_it->first_index_recurrent_connection_input_gate > str) {
      if (block_it->last_index_recurrent_connection_input_gate > end) {
        break;
      }

      euclidean_norm_st(
          0_UZ,
          block_it->last_index_recurrent_connection_input_gate -
              block_it->first_index_recurrent_connection_input_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              block_it->first_index_recurrent_connection_input_gate);
    }

    if (block_it->first_index_recurrent_connection_forget_gate > str) {
      if (block_it->last_index_recurrent_connection_forget_gate > end) {
        break;
      }

      euclidean_norm_st(
          0_UZ,
          block_it->last_index_recurrent_connection_forget_gate -
              block_it->first_index_recurrent_connection_forget_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              block_it->first_index_recurrent_connection_forget_gate);
    }

    if (block_it->first_index_recurrent_connection_output_gate > str) {
      if (block_it->last_index_recurrent_connection_output_gate > end) {
        break;
      }

      euclidean_norm_st(
          0_UZ,
          block_it->last_index_recurrent_connection_output_gate -
              block_it->first_index_recurrent_connection_output_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              block_it->first_index_recurrent_connection_output_gate);
    }
    //    [1] |END| Recurrent. |END|

    //    [1] Peepholes.
#ifndef NO_PEEPHOLE
    if (block_it->first_index_peephole_input_gate > str) {
      if (block_it->last_index_peephole_input_gate > end) {
        break;
      }

      euclidean_norm_st(0_UZ,
                        block_it->last_index_peephole_input_gate -
                            block_it->first_index_peephole_input_gate,
                        this->regularization__max_norm_constraints,
                        this->ptr_array_parameters +
                            block_it->first_index_peephole_input_gate);
    }

    if (block_it->first_index_peephole_forget_gate > str) {
      if (block_it->last_index_peephole_forget_gate > end) {
        break;
      }

      euclidean_norm_st(0_UZ,
                        block_it->last_index_peephole_forget_gate -
                            block_it->first_index_peephole_forget_gate,
                        this->regularization__max_norm_constraints,
                        this->ptr_array_parameters +
                            block_it->first_index_peephole_forget_gate);
    }

    if (block_it->first_index_peephole_output_gate > str) {
      if (block_it->last_index_peephole_output_gate > end) {
        break;
      }

      euclidean_norm_st(0_UZ,
                        block_it->last_index_peephole_output_gate -
                            block_it->first_index_peephole_output_gate,
                        this->regularization__max_norm_constraints,
                        this->ptr_array_parameters +
                            block_it->first_index_peephole_output_gate);
    }
#endif
    //    [1] |END| Peepholes. |END|
    // [0] |END| Gates. |END|
  }
}

void Model::Update_Weight_Regularization__Max_Norm_Constraints__Loop(
    size_t const str, size_t const end) {
  Layer const *const last_layer(this->ptr_last_layer - 1);
  Layer const *layer_it(this->ptr_array_layers + 1);

  if (this->total_neuron_units != 0_UZ) {
    this->Update_Weight_Regularization__Max_Norm_Constraints__Neurons__Loop(
        str, end, layer_it, last_layer);
  }

  if (this->total_block_units != 0_UZ) {
    this->Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__Loop(
        str, end, layer_it, last_layer);
  }
}

void Model::Update_Weight_Regularization__Max_Norm_Constraints__Neurons__OpenMP(
    size_t const str, size_t const end, Layer const *const layer_it,
    Layer const *const last_layer) {
  int const n_units(static_cast<int>(last_layer->ptr_last_neuron_unit -
                                     layer_it->ptr_array_neuron_units));
  int i;

  Neuron_unit const *const units(layer_it->ptr_array_neuron_units);

#pragma omp for schedule(static)
  for (i = 0; i < n_units; ++i) {
    if (*units[i].ptr_first_connection_index < str ||
        *units[i].ptr_last_connection_index > end)
      continue;

    euclidean_norm_mp(
        0_UZ, *units[i].ptr_number_connections,
        this->regularization__max_norm_constraints,
        this->ptr_array_parameters + *units[i].ptr_first_connection_index);
  }
}

void Model::Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__OpenMP(
    size_t const str, size_t const end, Layer const *const layer_it,
    Layer const *const last_layer) {
  int const n_units(static_cast<int>(last_layer->ptr_last_block_unit -
                                     layer_it->ptr_array_block_units));
  int i;

  BlockUnit const *const blocks(layer_it->ptr_array_block_units);

  CellUnit const *cell_end, *cell_it;

#pragma omp for schedule(static)
  for (i = 0; i < n_units; ++i) {
    // [0] Cell input.
    for (cell_end = blocks[i].ptr_last_cell_unit,
        cell_it = blocks[i].ptr_array_cell_units;
         cell_it != cell_end; ++cell_it) {
      //    [1] Input.
      if (cell_it->first_index_feedforward_connection_cell_input > str) {
        if (cell_it->last_index_feedforward_connection_cell_input > end) {
          continue;
        }

        euclidean_norm_mp(
            0_UZ,
            cell_it->last_index_feedforward_connection_cell_input -
                cell_it->first_index_feedforward_connection_cell_input,
            this->regularization__max_norm_constraints,
            this->ptr_array_parameters +
                cell_it->first_index_feedforward_connection_cell_input);
      }
      //    [1] |END| Input. |END|

      //    [1] Recurrent.
      if (cell_it->first_index_recurrent_connection_cell_input > str) {
        if (cell_it->last_index_recurrent_connection_cell_input > end) {
          continue;
        }

        euclidean_norm_mp(
            0_UZ,
            cell_it->last_index_recurrent_connection_cell_input -
                cell_it->first_index_recurrent_connection_cell_input,
            this->regularization__max_norm_constraints,
            this->ptr_array_parameters +
                cell_it->first_index_recurrent_connection_cell_input);
      }
      //    [1] |END| Recurrent. |END|
    }
    // [0] |END| Cell input. |END|

    // [0] Gates.
    //    [1] Input.
    if (blocks[i].first_index_feedforward_connection_input_gate > str) {
      if (blocks[i].last_index_feedforward_connection_input_gate > end) {
        continue;
      }

      euclidean_norm_mp(
          0_UZ,
          blocks[i].last_index_feedforward_connection_input_gate -
              blocks[i].first_index_feedforward_connection_input_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              blocks[i].first_index_feedforward_connection_input_gate);
    }

    if (blocks[i].first_index_feedforward_connection_forget_gate > str) {
      if (blocks[i].last_index_feedforward_connection_forget_gate > end) {
        continue;
      }

      euclidean_norm_mp(
          0_UZ,
          blocks[i].last_index_feedforward_connection_forget_gate -
              blocks[i].first_index_feedforward_connection_forget_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              blocks[i].first_index_feedforward_connection_forget_gate);
    }

    if (blocks[i].first_index_feedforward_connection_output_gate > str) {
      if (blocks[i].last_index_feedforward_connection_output_gate > end) {
        continue;
      }

      euclidean_norm_mp(
          0_UZ,
          blocks[i].last_index_feedforward_connection_output_gate -
              blocks[i].first_index_feedforward_connection_output_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              blocks[i].first_index_feedforward_connection_output_gate);
    }
    //    [1] |END| Input. |END|

    //    [1] Recurrent.
    if (blocks[i].first_index_recurrent_connection_input_gate > str) {
      if (blocks[i].last_index_recurrent_connection_input_gate > end) {
        continue;
      }

      euclidean_norm_mp(
          0_UZ,
          blocks[i].last_index_recurrent_connection_input_gate -
              blocks[i].first_index_recurrent_connection_input_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              blocks[i].first_index_recurrent_connection_input_gate);
    }

    if (blocks[i].first_index_recurrent_connection_forget_gate > str) {
      if (blocks[i].last_index_recurrent_connection_forget_gate > end) {
        continue;
      }

      euclidean_norm_mp(
          0_UZ,
          blocks[i].last_index_recurrent_connection_forget_gate -
              blocks[i].first_index_recurrent_connection_forget_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              blocks[i].first_index_recurrent_connection_forget_gate);
    }

    if (blocks[i].first_index_recurrent_connection_output_gate > str) {
      if (blocks[i].last_index_recurrent_connection_output_gate > end) {
        continue;
      }

      euclidean_norm_mp(
          0_UZ,
          blocks[i].last_index_recurrent_connection_output_gate -
              blocks[i].first_index_recurrent_connection_output_gate,
          this->regularization__max_norm_constraints,
          this->ptr_array_parameters +
              blocks[i].first_index_recurrent_connection_output_gate);
    }
    //    [1] |END| Recurrent. |END|

#ifndef NO_PEEPHOLE
    //    [1] Peepholes.
    if (blocks[i].first_index_peephole_input_gate > str) {
      if (blocks[i].last_index_peephole_input_gate > end) {
        continue;
      }

      euclidean_norm_mp(0_UZ,
                        blocks[i].last_index_peephole_input_gate -
                            blocks[i].first_index_peephole_input_gate,
                        this->regularization__max_norm_constraints,
                        this->ptr_array_parameters +
                            blocks[i].first_index_peephole_input_gate);
    }

    if (blocks[i].first_index_peephole_forget_gate > str) {
      if (blocks[i].last_index_peephole_forget_gate > end) {
        continue;
      }

      euclidean_norm_mp(0_UZ,
                        blocks[i].last_index_peephole_forget_gate -
                            blocks[i].first_index_peephole_forget_gate,
                        this->regularization__max_norm_constraints,
                        this->ptr_array_parameters +
                            blocks[i].first_index_peephole_forget_gate);
    }

    if (blocks[i].first_index_peephole_output_gate > str) {
      if (blocks[i].last_index_peephole_output_gate > end) {
        continue;
      }

      euclidean_norm_mp(0_UZ,
                        blocks[i].last_index_peephole_output_gate -
                            blocks[i].first_index_peephole_output_gate,
                        this->regularization__max_norm_constraints,
                        this->ptr_array_parameters +
                            blocks[i].first_index_peephole_output_gate);
    }
    //    [1] |END| Peepholes. |END|
#endif
    // [0] |END| Gates. |END|
  }
}

void Model::Update_Weight_Regularization__Max_Norm_Constraints__OpenMP(
    size_t const str, size_t const end) {
  Layer const *const last_layer(this->ptr_last_layer - 1);
  Layer const *layer_it(this->ptr_array_layers + 1);

#pragma omp parallel
  {
    if (this->total_neuron_units != 0_UZ) {
      this->Update_Weight_Regularization__Max_Norm_Constraints__Neurons__OpenMP(
          str, end, layer_it, last_layer);
    }

    if (this->total_block_units != 0_UZ) {
      this->Update_Weight_Regularization__Max_Norm_Constraints__LSTMs__OpenMP(
          str, end, layer_it, last_layer);
    }
  }
}
}  // namespace DL
