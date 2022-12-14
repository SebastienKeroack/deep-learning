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
#include "deep-learning/io/logger.hpp"
#include "deep-learning/ops/math.hpp"

using namespace DL::Math;

namespace DL::v1 {
void Model::compute_grad_adept(size_t const batch_size,
                               real const *const *const Ym) {
  switch (this->type) {
    case MODEL::AUTOENCODER:
      if (this->pre_training_level != 0_UZ && this->_initialized__weight) {
        this->compute_grad_adept_pre_train(batch_size);
        break;
      }
    default:
      if (this->seq_w > 1_UZ)
        this->compute_grad_adept_rec_st(batch_size, Ym);
      else
        this->compute_grad_adept_fwp_st(batch_size, Ym);
      break;
  }
}

void Model::compute_grad_adept_pre_train(size_t const batch_size) {
  if (this->seq_w > 1_UZ)
    this->compute_grad_adept_pre_train_rec_st(batch_size);
  else
    this->compute_grad_adept_pre_train_fwp_st(batch_size);
}

void Model::compute_grad_adept_fwp_st(size_t const batch_size,
                                      real const *const *const Ym) {
  Layer const *const layer(this->Get__Output_Layer());

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t i, i_k;

  real const *Y_i;
  real q, y;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

  for (i = 0_UZ; i != batch_size; ++i) {
    Y_i = Ym[i];

    i_k = i * n_out;

    unit_it = layer->ptr_array_AF_units;

    for (; unit_it != last_unit; ++unit_it, ++Y_i) {
      q = cast(unit_it->ptr_array_values[i_k]);
      y = *Y_i;

      unit_it->ptr_array_values[i_k].set_gradient(
          this->loss_fn_derivative(q, y, batch_size, n_out));
    }
  }
}

void Model::compute_grad_adept_rec_st(size_t const batch_size,
                                      real const *const *const Ym) {
  Layer const *const layer(this->Get__Output_Layer());

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t i, t, i_t_k;

  real const *Y_i_t;
  real q, y;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

  for (i = 0_UZ; i != batch_size; ++i) {
    for (t = this->n_time_delay; t != this->seq_w; ++t) {
      Y_i_t = Ym[i] + t * n_out;

      i_t_k = this->batch_size * n_out * t + i * n_out;

      unit_it = layer->ptr_array_AF_units;

      for (; unit_it != last_unit; ++unit_it, ++Y_i_t) {
        q = cast(unit_it->ptr_array_values[i_t_k]);
        y = *Y_i_t;

        unit_it->ptr_array_values[i_t_k].set_gradient(
            this->loss_fn_derivative(q, y, batch_size, n_out));
      }
    }
  }
}

void Model::compute_grad_adept_pre_train_fwp_st(size_t const batch_size) {
  Layer const *const layer(this->Get__Output_Layer());

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t i, i_k;

  real q, y;
  var const *Y_i;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

  for (i = 0_UZ; i != batch_size; ++i) {
    Y_i = this->get_out(
        this->ptr_array_layers + (this->pre_training_level - 1_UZ), i);

    i_k = i * n_out;

    unit_it = layer->ptr_array_AF_units;

    for (; unit_it != last_unit; ++unit_it, ++Y_i) {
      q = cast(unit_it->ptr_array_values[i_k]);
      y = cast(*Y_i);

      unit_it->ptr_array_values[i_k].set_gradient(
          this->loss_fn_derivative(q, y, batch_size, n_out));
    }
  }
}

void Model::compute_grad_adept_pre_train_rec_st(size_t const batch_size) {
  Layer const *const layer(this->Get__Output_Layer());

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t i, t, i_t_k;

  real q, y;
  var const *Y_i_t;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

  for (i = 0_UZ; i != batch_size; ++i) {
    for (t = this->n_time_delay; t != this->seq_w; ++t) {
      Y_i_t = this->get_out(
          this->ptr_array_layers + (this->pre_training_level - 1_UZ), i, t);

      i_t_k = this->batch_size * n_out * t + i * n_out;

      unit_it = layer->ptr_array_AF_units;

      for (; unit_it != last_unit; ++unit_it, ++Y_i_t) {
        q = cast(unit_it->ptr_array_values[i_t_k]);
        y = cast(*Y_i_t);

        unit_it->ptr_array_values[i_t_k].set_gradient(
            this->loss_fn_derivative(q, y, batch_size, n_out));
      }
    }
  }
}

void Model::update_derivatives_adept(void) {
  real *const derivatives(this->ptr_array_derivatives_parameters);
  var const *const parameters(this->ptr_array_parameters);

  for (size_t i(0_UZ); i != this->total_parameters_allocated; ++i)
    derivatives[i] += parameters[i].get_gradient();
}
}  // namespace DL::v1
