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

#include "pch.hpp"

#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/v1/learner/model.hpp"
#include "deep-learning/v1/data/datasets.hpp"

#include <omp.h>

namespace DL::v1 {
bool Model::Initialize__LSUV(size_t const max_n_trials,
                             size_t const max_batch_size, real const bias,
                             real const var_target, real const var_tolerance) {
  if (max_batch_size == 0_UZ) {
    ERR(L"Maximum batch size can not be equal to zero.");
    return false;
  } else if (var_target < 0_r) {
    ERR(L"Variance target (%f) can not be less than zero.", var_target);
    return false;
  } else if (var_tolerance < 0_r) {
    ERR(L"Variance tolerance (%f) can not be less than zero.", var_tolerance);
    return false;
  }

  this->_LSUV_Parameters.maximum_number_trials = max_n_trials;
  this->_LSUV_Parameters.maximum_batch_size = max_batch_size;
  this->_LSUV_Parameters.initial_bias = bias;
  this->_LSUV_Parameters.variance_target = var_target;
  this->_LSUV_Parameters.variance_tolerance = var_tolerance;

  this->_initialized__weight = false;
  this->_type_weights_initializer = INITIALIZER::LSUV;

  return true;
}

bool Model::Initialization__LSUV(DatasetV1 const *const dataset) {
  // Pre-initialize network with orthonormal matrices as in Saxe et al. (2014).
  this->layers_initialize_orthogonal(true, this->_LSUV_Parameters.initial_bias);

  size_t const n_data(std::min(this->_LSUV_Parameters.maximum_batch_size,
                               dataset->DatasetV1::get_n_data()));

  if (this->update_mem_batch_size(n_data) == false) {
    ERR(L"An error has been triggered from the \"update_mem_batch_size(%zu)\" "
        L"function.",
        n_data);
    return false;
  }

  if (this->use_mp && this->is_mp_initialized) {
    if (this->update_mem_thread_size(n_data) == false) {
      ERR(L"An error has been triggered from the \"update_mem_thread_size(%zu)\" "
          L"function.",
          n_data);
      return false;
    }

    omp_set_num_threads(static_cast<int>(this->number_threads));

    if (this->Initialization__LSUV__OpenMP(dataset) == false) {
      ERR(L"An error has been triggered from the "
          L"\"Initialization__LSUV__OpenMP(ptr)\" function.");
      return false;
    }
  } else if (this->Initialization__LSUV__Loop(dataset) == false) {
    ERR(L"An error has been triggered from the "
        L"\"Initialization__LSUV__Loop(ptr)\" function.");
    return false;
  }

  // Independently recurrent neural network.
  if (this->seq_w > 1_UZ && this->n_time_delay + 1_UZ == this->seq_w)
    this->indrec_initialize_uniform_ltm();

  if (this->ptr_array_derivatives_parameters != nullptr)
    this->clear_training_arrays();

  if (this->Use__Normalization()) this->Clear__Parameter__Normalized_Unit();

  this->_initialized__weight = true;

  return true;
}

void scale_parameters_st(var *parameter_it, var const *const parameter_last,
                         real const scale) {
  for (; parameter_it != parameter_last; ++parameter_it) *parameter_it *= scale;
}

// TODO: Intermediate propagation.
bool Model::Initialization__LSUV__Loop(DatasetV1 const *const dataset) {
  size_t const max_n_trials(this->_LSUV_Parameters.maximum_number_trials);
  size_t layer_index, trial_index;

  real const epsilon(this->_LSUV_Parameters.epsilon),
      var_target(std::max(this->_LSUV_Parameters.variance_target, epsilon)),
      var_tolerance(this->_LSUV_Parameters.variance_tolerance);
  real variance;

  Layer const *layer;

  auto compute_variance_st_fn([&](size_t const layer_index_received) -> real {
    size_t const n_data(std::min(this->_LSUV_Parameters.maximum_batch_size,
                                 dataset->DatasetV1::get_n_data())),
        batch_size_max(this->batch_size),
        n_batch(static_cast<size_t>(ceil(static_cast<double>(n_data) /
                                         static_cast<double>(batch_size_max))));
    size_t batch_size, i;

    real variance(0_r);

    for (i = 0_UZ; i != n_batch; ++i) {
      batch_size =
          i + 1_UZ != n_batch ? batch_size_max : n_data - i * batch_size_max;

      this->forward_pass(
          batch_size, dataset->DatasetV1::Get__Input_Array() + i * batch_size_max,
          0ll, static_cast<long long int>(layer_index_received) + 1ll);

      variance += this->get_layer_variance(layer, batch_size);
    }

    return (variance <= 0_r ? epsilon : variance);
  });

  for (layer_index = 1_UZ; layer_index != this->total_layers; ++layer_index) {
    layer = this->ptr_array_layers + layer_index;

    if (layer->type_layer == LAYER::CONVOLUTION ||
        layer->type_layer == LAYER::FULLY_CONNECTED) {
      trial_index = 0_UZ;

      variance = compute_variance_st_fn(layer_index);

      while (std::abs(variance - var_target) >= var_tolerance) {
        scale_parameters_st(
            this->ptr_array_parameters + *layer->ptr_first_connection_index,
            this->ptr_array_parameters + *layer->ptr_last_connection_index,
            1_r / (std::sqrt(variance) / std::sqrt(var_target)));

        if (++trial_index < max_n_trials)
          variance = compute_variance_st_fn(layer_index);
        else
          break;
      }
    }
  }

  return true;
}

void scale_parameters_mp(int const length, var *const parameters,
                         real const scale) {
  int i;

#pragma omp parallel for schedule(static)
  for (i = 0; i < length; ++i) {
    parameters[i] *= scale;
  }
}

bool Model::Initialization__LSUV__OpenMP(DatasetV1 const *const dataset) {
  size_t const max_n_trials(this->_LSUV_Parameters.maximum_number_trials);
  size_t layer_index, trial_index;

  real const epsilon(this->_LSUV_Parameters.epsilon),
      var_target(std::max(this->_LSUV_Parameters.variance_target, epsilon)),
      var_tolerance(this->_LSUV_Parameters.variance_tolerance);
  real variance;

  Layer const *layer;

  auto compute_variance_mp_fn([&](size_t const layer_index_received) -> real {
    size_t const n_data(std::min(this->_LSUV_Parameters.maximum_batch_size,
                                 dataset->DatasetV1::get_n_data())),
        batch_size_max(this->batch_size),
        n_batch(static_cast<size_t>(ceil(static_cast<double>(n_data) /
                                         static_cast<double>(batch_size_max))));
    size_t batch_size(0_UZ), i(0_UZ);

    real variance(0_r);

#pragma omp parallel private(i, batch_size)
    for (i = 0_UZ; i != n_batch; ++i) {
      batch_size =
          i + 1_UZ != n_batch ? batch_size_max : n_data - i * batch_size_max;

      this->forward_pass(
          batch_size, dataset->DatasetV1::Get__Input_Array() + i * batch_size_max,
          0ll, static_cast<long long int>(layer_index_received) + 1ll);

#pragma omp barrier
#pragma omp single
      variance += this->get_layer_variance(layer, batch_size);
    }

    return (variance <= 0_r ? epsilon : variance);
  });

  for (layer_index = 1_UZ; layer_index != this->total_layers; ++layer_index) {
    layer = this->ptr_array_layers + layer_index;

    if (layer->type_layer == LAYER::CONVOLUTION ||
        layer->type_layer == LAYER::FULLY_CONNECTED) {
      trial_index = 0_UZ;

      variance = compute_variance_mp_fn(layer_index);

      while (std::abs(variance - var_target) >= var_tolerance) {
        scale_parameters_mp(
            static_cast<int>(*layer->ptr_last_connection_index -
                             *layer->ptr_first_connection_index),
            this->ptr_array_parameters + *layer->ptr_first_connection_index,
            1_r / (std::sqrt(variance) / std::sqrt(var_target)));

        if (++trial_index < max_n_trials)
          variance = compute_variance_mp_fn(layer_index);
        else
          break;
      }
    }
  }

  return true;
}
}  // namespace DL