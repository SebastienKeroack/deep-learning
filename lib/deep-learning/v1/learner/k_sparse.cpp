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
#include "deep-learning/v1/mem/reallocate.hpp"

// Standard:
#include <omp.h>

namespace DL::v1 {
bool Layer::Use__K_Sparsity(void) const { return this->k_sparsity != 0_UZ; }

size_t Layer::Get__K_Sparsity(void) const { return this->k_sparsity; }

bool Model::Set__K_Sparsity(size_t const index_layer_received,
                            size_t const k_sparsity_received) {
  if (index_layer_received >= this->total_layers) {
    ERR(L"Layer received (%zu) overflow the "
        L"number of layers (%zu) in the neural network.",
        index_layer_received, this->total_layers);
    return false;
  } else if (this->ptr_array_layers == nullptr) {
    ERR(L"`ptr_array_layers` is a nullptr.");
    return false;
  }

  return this->Set__K_Sparsity(this->ptr_array_layers + index_layer_received,
                               k_sparsity_received);
}

bool Model::Set__K_Sparsity(Layer *const ptr_layer_received,
                            size_t const k_sparsity_received) {
  if (ptr_layer_received == nullptr) {
    ERR(L"`ptr_layer_received` is a nullptr.");
    return false;
  } else if (ptr_layer_received == this->ptr_array_layers) {
    ERR(L"Layer received as argument is the input layer.");
    return false;
  } else if (ptr_layer_received == this->ptr_last_layer - 1) {
    ERR(L"Layer received as argument is the output layer.");
    return false;
  } else if (ptr_layer_received->k_sparsity == k_sparsity_received) {
    return true;
  } else if (static_cast<size_t>(static_cast<real>(k_sparsity_received) *
                                 ptr_layer_received->alpha_sparsity) >
             *ptr_layer_received->ptr_number_outputs) {
    ERR(L"k-sparse cannot be %zu because an "
        L"overflow (%zu * %f = %zu) occur (limit=%zu).",
        k_sparsity_received, k_sparsity_received,
        ptr_layer_received->alpha_sparsity,
        static_cast<size_t>(static_cast<real>(k_sparsity_received) *
                            ptr_layer_received->alpha_sparsity),
        *ptr_layer_received->ptr_number_outputs);
    return false;
  }

  if (ptr_layer_received->k_sparsity == 0_UZ && k_sparsity_received != 0_UZ) {
    // TODO: Allocation based on the number of k-sparse filters
    // and not the total units.
    if (++this->total_k_sparse_layers == 1_UZ &&
        this->Allocate__Sparse_K_Filter() == false) {
      ERR(L"An error has been triggered from the "
          L"`Allocate__Sparse_K_Filter()` function.");
      --this->total_k_sparse_layers;
      return false;
    }
  } else if (ptr_layer_received->k_sparsity != 0_UZ &&
             k_sparsity_received == 0_UZ) {
    if (this->total_k_sparse_layers != 0_UZ &&
        --this->total_k_sparse_layers == 0_UZ) {
      this->Deallocate__Sparse_K_Filter();
    }
  }

  ptr_layer_received->k_sparsity = k_sparsity_received;

  return true;
}

bool Model::Set__Alpha_Sparsity(size_t const index_layer_received,
                                real const alpha_sparsity_received) {
  if (index_layer_received >= this->total_layers) {
    ERR(L"Layer received (%zu) overflow the "
        L"number of layers (%zu) in the neural network.",
        index_layer_received, this->total_layers);
    return false;
  } else if (this->ptr_array_layers == nullptr) {
    ERR(L"`ptr_array_layers` is a nullptr.");
    return false;
  }

  return (this->Set__Alpha_Sparsity(
      this->ptr_array_layers + index_layer_received, alpha_sparsity_received));
}

bool Model::Set__Alpha_Sparsity(Layer *const ptr_layer_received,
                                real const alpha_sparsity_received) {
  if (ptr_layer_received == nullptr) {
    ERR(L"`ptr_layer_received` is a nullptr.");
    return false;
  } else if (ptr_layer_received == this->ptr_array_layers) {
    ERR(L"Layer received as argument is the input layer.");
    return false;
  } else if (ptr_layer_received == this->ptr_last_layer - 1) {
    ERR(L"Layer received as argument is the output layer.");
    return false;
  } else if (alpha_sparsity_received < 0_r) {
    ERR(L"alpha k-sparse (%f) cannot be less than zero.",
        alpha_sparsity_received);
    return false;
  } else if (static_cast<size_t>(
                 static_cast<real>(ptr_layer_received->k_sparsity) *
                 alpha_sparsity_received) >
             *ptr_layer_received->ptr_number_outputs) {
    ERR(L"alpha k-sparse cannot be %f because an overflow (%zu * %f = %zu) "
        L"occur (limit=%zu).",
        alpha_sparsity_received, ptr_layer_received->k_sparsity,
        alpha_sparsity_received,
        static_cast<size_t>(static_cast<real>(ptr_layer_received->k_sparsity) *
                            alpha_sparsity_received),
        *ptr_layer_received->ptr_number_outputs);
    return false;
  }

  ptr_layer_received->alpha_sparsity = alpha_sparsity_received;

  return true;
}

void Model::Assign__Sparsity_Activities(size_t const number_threads_received) {
  Layer const *const last_layer(this->ptr_last_layer);
  Layer *layer_it(this->ptr_array_layers);

  std::pair<size_t, var> *k_sparse_activities(
      this->ptr_array_k_sparse_activities);

  // Assign array position to each layers.
  for (; layer_it != last_layer; ++layer_it) {
    layer_it->ptr_array_k_sparse_activities = k_sparse_activities;

    switch (layer_it->type_layer) {
      case LAYER::AVERAGE_POOLING:
      case LAYER::MAX_POOLING:
      case LAYER::LSTM:
      case LAYER::RESIDUAL:
        k_sparse_activities +=
            number_threads_received * *layer_it->ptr_number_outputs;
        break;
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_RECURRENT:
        k_sparse_activities +=
            number_threads_received *
            static_cast<size_t>(layer_it->ptr_last_AF_unit -
                                layer_it->ptr_array_AF_units);
        break;
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        k_sparse_activities +=
            number_threads_received *
            static_cast<size_t>(layer_it->ptr_last_AF_Ind_recurrent_unit -
                                layer_it->ptr_array_AF_Ind_recurrent_units);
        break;
      default:
        ERR(L"Type layer (%d | %ls) is not managed in the switch.",
            layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
        return;
    }
  }
}

void Model::Sparse_K_Filter(
    size_t const time_step_index_received, size_t const batch_size,
    size_t const input_unit_size_received, size_t const k_sparsity_received,
    std::pair<size_t, var> *const ptr_array_k_sparses_received,
    var *const ptr_array_inputs_received) {
  if (this->use_mp && this->is_mp_initialized) {
    this->Sparse_K_Filter__Loop(time_step_index_received, batch_size,
                                input_unit_size_received, k_sparsity_received,
                                ptr_array_k_sparses_received,
                                ptr_array_inputs_received);
  } else {
    this->Sparse_K_Filter__OpenMP(time_step_index_received, batch_size,
                                  input_unit_size_received, k_sparsity_received,
                                  ptr_array_k_sparses_received,
                                  ptr_array_inputs_received);
  }
}

void Model::Sparse_K_Filter__Loop(
    size_t const time_step_index_received, size_t const batch_size,
    size_t const input_unit_size_received, size_t const k_sparsity_received,
    std::pair<size_t, var> *const ptr_array_k_sparses_received,
    var *const ptr_array_inputs_received) {
  if (k_sparsity_received == input_unit_size_received) {
    return;
  }

  size_t const tmp_unit_timed_index(input_unit_size_received *
                                    time_step_index_received);
  size_t tmp_example_index, tmp_unit_index, tmp_unit_data_timed_index;

  // Custom sorting.
  auto tmp_Sort_Pair([](std::pair<size_t, var> &a_received,
                        std::pair<size_t, var> &b_received) -> bool {
    return a_received.second < b_received.second;
  });

  for (tmp_example_index = 0_UZ; tmp_example_index != batch_size;
       ++tmp_example_index) {
    tmp_unit_data_timed_index = tmp_example_index * input_unit_size_received +
                                this->batch_size * tmp_unit_timed_index;

    // Initialize array of pairs.
    for (tmp_unit_index = 0_UZ; tmp_unit_index != k_sparsity_received;
         ++tmp_unit_index) {
      ptr_array_k_sparses_received[tmp_unit_index].first = tmp_unit_index;

      ptr_array_k_sparses_received[tmp_unit_index].second =
          ptr_array_inputs_received[tmp_unit_data_timed_index + tmp_unit_index];
    }

    // Sort the array of pairs.
    std::sort(ptr_array_k_sparses_received,
              ptr_array_k_sparses_received + k_sparsity_received,
              tmp_Sort_Pair);

    // Compute the highest input into the array of pair.
    for (tmp_unit_index = k_sparsity_received;
         tmp_unit_index != input_unit_size_received; ++tmp_unit_index) {
      if (ptr_array_k_sparses_received[0].second <
          ptr_array_inputs_received[tmp_unit_data_timed_index +
                                    tmp_unit_index]) {
        ptr_array_k_sparses_received[0].first = tmp_unit_index;

        ptr_array_k_sparses_received[0].second =
            ptr_array_inputs_received[tmp_unit_data_timed_index +
                                      tmp_unit_index];

        std::sort(ptr_array_k_sparses_received,
                  ptr_array_k_sparses_received + k_sparsity_received,
                  tmp_Sort_Pair);
      }
    }

    // Zero out array of inputs.
    VARZERO(ptr_array_inputs_received + tmp_unit_data_timed_index,
            input_unit_size_received * sizeof(var));

    // Keep the k-sparses input(s).
    for (tmp_unit_index = 0_UZ; tmp_unit_index != k_sparsity_received;
         ++tmp_unit_index) {
      ptr_array_inputs_received[tmp_unit_data_timed_index +
                                ptr_array_k_sparses_received[tmp_unit_index]
                                    .first] =
          ptr_array_k_sparses_received[tmp_unit_index].second;
    }
  }
}

void Model::Sparse_K_Filter__OpenMP(
    size_t const time_step_index_received, size_t const batch_size,
    size_t const input_unit_size_received, size_t const k_sparsity_received,
    std::pair<size_t, var> *const ptr_array_k_sparses_received,
    var *const ptr_array_inputs_received) {
  if (k_sparsity_received == input_unit_size_received) return;

  int batch_size_received__int(static_cast<int>(batch_size)),
      tmp_example_index__int;

  size_t const tmp_unit_timed_index(input_unit_size_received *
                                    time_step_index_received);
  size_t tmp_unit_index(0_UZ), tmp_unit_data_timed_index(0_UZ);

  std::pair<size_t, var> *k_sparse_activities(nullptr);

  // Custom sorting.
  auto tmp_Sort_Pair([](std::pair<size_t, var> &a_received,
                        std::pair<size_t, var> &b_received) -> bool {
    return a_received.second < b_received.second;
  });

#pragma omp parallel for schedule(static) private( \
    tmp_unit_index, tmp_unit_data_timed_index, k_sparse_activities)
  for (tmp_example_index__int = 0;
       tmp_example_index__int < batch_size_received__int;
       ++tmp_example_index__int) {
    tmp_unit_data_timed_index =
        static_cast<size_t>(tmp_example_index__int) * input_unit_size_received +
        this->batch_size * tmp_unit_timed_index;

    k_sparse_activities =
        ptr_array_k_sparses_received +
        static_cast<size_t>(omp_get_thread_num()) * k_sparsity_received;

    // Initialize array of pairs.
    for (tmp_unit_index = 0_UZ; tmp_unit_index != k_sparsity_received;
         ++tmp_unit_index) {
      k_sparse_activities[tmp_unit_index].first = tmp_unit_index;

      k_sparse_activities[tmp_unit_index].second =
          ptr_array_inputs_received[tmp_unit_data_timed_index + tmp_unit_index];
    }

    // Sort the array of pairs.
    std::sort(k_sparse_activities,
              k_sparse_activities + k_sparsity_received, tmp_Sort_Pair);

    // Compute the highest input into the array of pair.
    for (tmp_unit_index = k_sparsity_received;
         tmp_unit_index != input_unit_size_received; ++tmp_unit_index) {
      if (k_sparse_activities[0].second <
          ptr_array_inputs_received[tmp_unit_data_timed_index +
                                    tmp_unit_index]) {
        k_sparse_activities[0].first = tmp_unit_index;

        k_sparse_activities[0].second =
            ptr_array_inputs_received[tmp_unit_data_timed_index +
                                      tmp_unit_index];

        std::sort(k_sparse_activities,
                  k_sparse_activities + k_sparsity_received, tmp_Sort_Pair);
      }
    }

    // Zero out array of inputs.
    VARZERO(ptr_array_inputs_received + tmp_unit_data_timed_index,
            input_unit_size_received * sizeof(var));

    // Keep the k-sparses input(s).
    for (tmp_unit_index = 0_UZ; tmp_unit_index != k_sparsity_received;
         ++tmp_unit_index) {
      ptr_array_inputs_received[tmp_unit_data_timed_index +
                                k_sparse_activities[tmp_unit_index].first] =
          k_sparse_activities[tmp_unit_index].second;
    }
  }
}
}  // namespace DL::v1