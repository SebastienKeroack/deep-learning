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

#include <omp.h>

namespace DL::v1 {
void Model::compute_r_fwp_mp(size_t const batch_size,
                             real const *const *const Ym) {
  Layer const *const layer(this->Get__Output_Layer());

  int const batch_size_(static_cast<int>(batch_size));
  int i, thread_id;

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t i_k;

  double const Y_mean(this->ptr_array_accuracy_values[0][0]),
      Q_mean(this->ptr_array_accuracy_values[1][0]);
  double *const numerators(this->ptr_array_accuracy_values[2]),
      *const denominators_Y(this->ptr_array_accuracy_values[3]),
          *const denominators_y(this->ptr_array_accuracy_values[4]),
      y_i_k_mean_diff, q_i_k_mean_diff;

  real const *Y_i;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

#pragma omp for schedule(static)
  for (i = 0; i < batch_size_; ++i) {
    thread_id = omp_get_thread_num();

    Y_i = Ym[i];

    i_k = i * n_out;

    unit_it = layer->ptr_array_AF_units;

    for (; unit_it != last_unit; ++unit_it, ++Y_i) {
      y_i_k_mean_diff = castd(*Y_i) - Y_mean;
      q_i_k_mean_diff = castd(unit_it->ptr_array_values[i_k]) - Q_mean;

      numerators[thread_id] += y_i_k_mean_diff * q_i_k_mean_diff;
      denominators_Y[thread_id] += y_i_k_mean_diff * y_i_k_mean_diff;
      denominators_y[thread_id] += q_i_k_mean_diff * q_i_k_mean_diff;
    }
  }
}

void Model::compute_r_rec_mp(size_t const batch_size,
                             real const *const *const Ym) {
  Layer const *const layer(this->Get__Output_Layer());

  int const batch_size_(static_cast<int>(batch_size));
  int i, thread_id;

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t t, i_t_k;

  double const Y_mean(this->ptr_array_accuracy_values[0][0]),
      Q_mean(this->ptr_array_accuracy_values[1][0]);
  double *const numerators(this->ptr_array_accuracy_values[2]),
      *const denominators_Y(this->ptr_array_accuracy_values[3]),
          *const denominators_y(this->ptr_array_accuracy_values[4]),
      y_i_t_k_mean_diff, q_i_t_k_mean_diff;

  real const *Y_i_t;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

#pragma omp for schedule(static)
  for (i = 0; i < batch_size_; ++i) {
    thread_id = omp_get_thread_num();

    for (t = this->n_time_delay; t != this->seq_w; ++t) {
      Y_i_t = Ym[i] + t * this->n_out;

      i_t_k = this->batch_size * n_out * t + i * n_out;

      unit_it = layer->ptr_array_AF_units;

      for (; unit_it != last_unit; ++unit_it, ++Y_i_t) {
        y_i_t_k_mean_diff = castd(*Y_i_t) - Y_mean;
        q_i_t_k_mean_diff = castd(unit_it->ptr_array_values[i_t_k]) - Q_mean;

        numerators[thread_id] += y_i_t_k_mean_diff * q_i_t_k_mean_diff;
        denominators_Y[thread_id] += y_i_t_k_mean_diff * y_i_t_k_mean_diff;
        denominators_y[thread_id] += q_i_t_k_mean_diff * q_i_t_k_mean_diff;
      }
    }
  }
}

void Model::compute_r_pre_train_fwp_mp(size_t const batch_size) {
  Layer const *const layer(this->Get__Output_Layer());

  int const batch_size_(static_cast<int>(batch_size));
  int i, thread_id;

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t i_k;

  double const Y_mean(this->ptr_array_accuracy_values[0][0]),
      Q_mean(this->ptr_array_accuracy_values[1][0]);
  double *const numerators(this->ptr_array_accuracy_values[2]),
      *const denominators_Y(this->ptr_array_accuracy_values[3]),
          *const denominators_y(this->ptr_array_accuracy_values[4]),
      y_i_k_mean_diff, q_i_k_mean_diff;
  var const *Y_i;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

#pragma omp for schedule(static)
  for (i = 0; i < batch_size_; ++i) {
    thread_id = omp_get_thread_num();

    Y_i = this->get_out(
        this->ptr_array_layers + (this->pre_training_level - 1_UZ), i);

    i_k = i * n_out;

    unit_it = layer->ptr_array_AF_units;

    for (; unit_it != last_unit; ++unit_it, ++Y_i) {
      y_i_k_mean_diff = castd(*Y_i) - Y_mean;
      q_i_k_mean_diff = castd(unit_it->ptr_array_values[i_k]) - Q_mean;

      numerators[thread_id] += y_i_k_mean_diff * q_i_k_mean_diff;
      denominators_Y[thread_id] += y_i_k_mean_diff * y_i_k_mean_diff;
      denominators_y[thread_id] += q_i_k_mean_diff * q_i_k_mean_diff;
    }
  }
}

void Model::compute_r_pre_train_rec_mp(size_t const batch_size) {
  Layer const *const layer(this->Get__Output_Layer());

  int const batch_size_(static_cast<int>(batch_size));
  int i, thread_id;

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t t, i_t_k;

  double const Y_mean(this->ptr_array_accuracy_values[0][0]),
      Q_mean(this->ptr_array_accuracy_values[1][0]);
  double *const numerators(this->ptr_array_accuracy_values[2]),
      *const denominators_Y(this->ptr_array_accuracy_values[3]),
          *const denominators_y(this->ptr_array_accuracy_values[4]),
      y_i_t_k_mean_diff, q_i_t_k_mean_diff;
  var const *Y_i_t;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

#pragma omp for schedule(static)
  for (i = 0; i < batch_size_; ++i) {
    thread_id = omp_get_thread_num();

    for (t = this->n_time_delay; t != this->seq_w; ++t) {
      Y_i_t = this->get_out(
          this->ptr_array_layers + (this->pre_training_level - 1_UZ), i, t);

      i_t_k = this->batch_size * n_out * t + i * n_out;

      unit_it = layer->ptr_array_AF_units;

      for (; unit_it != last_unit; ++unit_it, ++Y_i_t) {
        y_i_t_k_mean_diff = castd(*Y_i_t) - Y_mean;
        q_i_t_k_mean_diff = castd(unit_it->ptr_array_values[i_t_k]) - Q_mean;

        numerators[thread_id] += y_i_t_k_mean_diff * q_i_t_k_mean_diff;
        denominators_Y[thread_id] += y_i_t_k_mean_diff * y_i_t_k_mean_diff;
        denominators_y[thread_id] += q_i_t_k_mean_diff * q_i_t_k_mean_diff;
      }
    }
  }
}
}  // namespace DL
