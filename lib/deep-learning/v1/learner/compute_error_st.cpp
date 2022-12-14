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

#include "deep-learning/io/logger.hpp"
#include "deep-learning/v1/learner/model.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"
#include "deep-learning/ops/math.hpp"

using namespace DL::Math;

namespace DL::v1 {
void Model::compute_error_fwp_st(size_t const batch_size,
                                 real const *const *const Ym) {
  Layer const *const layer(this->Get__Output_Layer());

  bool const cce_accu(ACCU_FN::CROSS_ENTROPY == this->type_accuracy_function);

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t i, k, k_target(n_out), k_predict(0_UZ), i_k;

  real const *Y_i;
  real q, q_max, y;
  var const *X;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

  for (i = 0_UZ; i != batch_size; ++i) {
    Y_i = Ym[i];

    i_k = i * n_out;

    X = layer->ptr_array_pre_activation_functions + i_k;

    unit_it = layer->ptr_array_AF_units;

    if (cce_accu) {
      // reset predicted maximum at -Inf.
      q_max = -(std::numeric_limits<real>::max)();

      // Loop through each predicted output to find the largest output.
      for (k = 0_UZ; k != n_out; ++k) {
        // If the oIdx is the largest output.
        if ((q = cast(unit_it->ptr_array_values[i_k + k])) >= q_max) {
          // Store the largest output index.
          k_predict = k;

          // State the maximum predicted output.
          q_max = q;
        }

        if (Y_i[k] == 1_r) k_target = k;
      }

      // If the desired output index equal the largest predicted output index.
      this->ptr_array_accuracy_values[0][0] +=
          static_cast<double>(k_target == k_predict);
    }

    for (; unit_it != last_unit; ++unit_it, ++Y_i, ++X) {
      q = cast(unit_it->ptr_array_values[i_k]);
      y = *Y_i;
      
      this->update_loss(q, y);

      switch (this->type_accuracy_function) {
        case ACCU_FN::CROSS_ENTROPY:
          break;
        case ACCU_FN::DISTANCE:
          if (std::abs(q - y) <= this->acc_var)
            this->ptr_array_accuracy_values[0][0] += 1.0;
          break;
        case ACCU_FN::DIRECTIONAL:
          this->ptr_array_accuracy_values[0][0] += 1.0;
          break;
        case ACCU_FN::R:
          this->ptr_array_accuracy_values[0][0] += castd(y);
          this->ptr_array_accuracy_values[1][0] += castd(q);
          break;
        case ACCU_FN::SIGN:
          if (sign(y) == sign(q) ||
              (sign(y) == 0_r && q >= -EPSILON && q <= EPSILON))
            this->ptr_array_accuracy_values[0][0] += 1.0;
          break;
        default:
          ERR(L"Accuracy type (%d | %ls) is not managed in the switch",
              this->type_accuracy_function,
              ACC_FN_NAME[this->type_accuracy_function].c_str());
          break;
      }

      if (ACTIVATION::SOFTMAX == *unit_it->ptr_type_activation_function) {
        unit_it->ptr_array_errors[i_k] = 0_r;
        for (k = 0_UZ; k != n_out; ++k) {
          unit_it->ptr_array_errors[i_k] +=
              (q * static_cast<real>(unit_it == (layer->ptr_array_AF_units + k)) -
               cast(layer->ptr_array_AF_units[k].ptr_array_values[i_k]) * q) *
              this->loss_fn_derivative(
                  cast(layer->ptr_array_AF_units[k].ptr_array_values[i_k]),
                  Ym[i][k], batch_size, n_out);
        }
      } else {
        unit_it->ptr_array_errors[i_k] =
            this->activation_fn_derivative(*unit_it->ptr_type_activation_function,
                                           cast(*X), q, y) *
            this->loss_fn_derivative(q, y, batch_size, n_out);
      }
    }
  }

  // copy derivative AFs to derivative neurons.
  memcpy(layer->ptr_array_neuron_units->ptr_array_errors,
         layer->ptr_array_AF_units->ptr_array_errors,
         this->batch_size * n_out * sizeof(real));
  // |END| copy derivative AFs to derivative neurons. |END|
}

void Model::compute_error_rec_st(size_t const batch_size,
                                 real const *const *const Ym) {
  Layer const *const layer(this->Get__Output_Layer());

  bool const cce_accu(ACCU_FN::CROSS_ENTROPY == this->type_accuracy_function);

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t i, t, k, k_target(n_out), k_predict(0_UZ), i_t_k;

  real const *Y_i_t, *Y_i_tm1;
  real q, q_max, y;
  var const *X;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

  for (i = 0_UZ; i != batch_size; ++i) {
    for (t = this->n_time_delay; t != this->seq_w; ++t) {
      Y_i_t = Ym[i] + t * n_out;

      Y_i_tm1 = Ym[i] + (t - 1_UZ) * n_out;

      i_t_k = this->batch_size * n_out * t + i * n_out;

      X = layer->ptr_array_pre_activation_functions + i_t_k;

      unit_it = layer->ptr_array_AF_units;

      if (cce_accu) {
        // reset predicted maximum at -Inf.
        q_max = -(std::numeric_limits<real>::max)();

        // Loop through each predicted output to find the largest output.
        for (k = 0_UZ; k != n_out; ++k) {
          // If the oIdx is the largest output.
          if ((q = cast(unit_it->ptr_array_values[i_t_k + k])) >= q_max) {
            // Store the largest output index.
            k_predict = k;

            // State the maximum predicted output.
            q_max = q;
          }

          if (Y_i_t[k] == 1_r) k_target = k;
        }

        // If the desired output index equal the largest predicted output index.
        this->ptr_array_accuracy_values[0][0] +=
            static_cast<double>(k_target == k_predict);
      }

      for (; unit_it != last_unit; ++unit_it, ++Y_i_t, ++Y_i_tm1, ++X) {
        q = cast(unit_it->ptr_array_values[i_t_k]);
        y = *Y_i_t;

        this->update_loss(q, y);

        switch (this->type_accuracy_function) {
          case ACCU_FN::CROSS_ENTROPY:
            break;
          case ACCU_FN::DISTANCE:
            if (std::abs(q - y) <= this->acc_var)
              this->ptr_array_accuracy_values[0][0] += 1.0;
            break;
          case ACCU_FN::DIRECTIONAL:
            if (t == 0_UZ || sign(y - cast(*Y_i_tm1)) == sign(q - cast(*Y_i_tm1)))
              this->ptr_array_accuracy_values[0][0] += 1.0;
            break;
          case ACCU_FN::R:
            this->ptr_array_accuracy_values[0][0] += castd(y);
            this->ptr_array_accuracy_values[1][0] += castd(q);
            break;
          case ACCU_FN::SIGN:
            if (sign(y) == sign(q) ||
                (sign(y) == 0_r && q >= -EPSILON && q <= EPSILON))
              this->ptr_array_accuracy_values[0][0] += 1.0;
            break;
          default:
            ERR(L"Accuracy type (%d | %ls) is not managed in the switch",
                this->type_accuracy_function,
                ACC_FN_NAME[this->type_accuracy_function].c_str());
            break;
        }

        if (ACTIVATION::SOFTMAX == *unit_it->ptr_type_activation_function) {
          unit_it->ptr_array_errors[i_t_k] = 0_r;
          for (k = 0_UZ; k != n_out; ++k) {
            unit_it->ptr_array_errors[i_t_k] +=
                (q * static_cast<real>(unit_it ==
                                       (layer->ptr_array_AF_units + k)) -
                 cast(layer->ptr_array_AF_units[k].ptr_array_values[i_t_k]) * q) *
                this->loss_fn_derivative(
                    cast(layer->ptr_array_AF_units[k].ptr_array_values[i_t_k]),
                    Ym[i][k], batch_size, n_out);
          }
        } else {
          unit_it->ptr_array_errors[i_t_k] =
              this->activation_fn_derivative(
                  *unit_it->ptr_type_activation_function, cast(*X), q, y) *
              this->loss_fn_derivative(q, y, batch_size, n_out);
        }
      }
    }
  }

  // copy derivative AFs to derivative neurons.
  memcpy(layer->ptr_array_neuron_units->ptr_array_errors +
             this->batch_size * n_out * this->n_time_delay,
         layer->ptr_array_AF_units->ptr_array_errors +
             this->batch_size * n_out * this->n_time_delay,
         this->batch_size * n_out * (this->seq_w - this->n_time_delay) *
             sizeof(real));
  // |END| copy derivative AFs to derivative neurons. |END|
}

void Model::compute_error_pre_train_fwp_st(size_t const batch_size) {
  Layer const *const layer(this->Get__Output_Layer());

  bool const cce_accu(ACCU_FN::CROSS_ENTROPY == this->type_accuracy_function);

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t i, k, k_target(n_out), k_predict(0_UZ), i_k;

  real q, q_max, y;
  var const *X, *Y_i;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

  for (i = 0_UZ; i != batch_size; ++i) {
    Y_i = this->get_out(
        this->ptr_array_layers + (this->pre_training_level - 1_UZ), i);

    i_k = i * n_out;

    X = layer->ptr_array_pre_activation_functions + i_k;

    unit_it = layer->ptr_array_AF_units;

    if (cce_accu) {
      // reset predicted maximum at -Inf.
      q_max = -(std::numeric_limits<real>::max)();

      // Loop through each predicted output to find the largest output.
      for (k = 0_UZ; k != n_out; ++k) {
        // If the oIdx is the largest output.
        if ((q = cast(unit_it->ptr_array_values[i_k + k])) >= q_max) {
          // Store the largest output index.
          k_predict = k;

          // State the maximum predicted output.
          q_max = q;
        }

        if (Y_i[k] == 1_r) k_target = k;
      }

      // If the desired output index equal the largest predicted output index.
      this->ptr_array_accuracy_values[0][0] +=
          static_cast<double>(k_target == k_predict);
    }

    for (; unit_it != last_unit; ++unit_it, ++Y_i, ++X) {
      q = cast(unit_it->ptr_array_values[i_k]);
      y = cast(*Y_i);

      this->update_loss(q, y);

      switch (this->type_accuracy_function) {
        case ACCU_FN::CROSS_ENTROPY:
          break;
        case ACCU_FN::DISTANCE:
          if (std::abs(q - y) <= this->acc_var)
            this->ptr_array_accuracy_values[0][0] += 1.0;
          break;
        case ACCU_FN::DIRECTIONAL:
          this->ptr_array_accuracy_values[0][0] += 1.0;
          break;
        case ACCU_FN::R:
          this->ptr_array_accuracy_values[0][0] += castd(y);
          this->ptr_array_accuracy_values[1][0] += castd(q);
          break;
        case ACCU_FN::SIGN:
          if (sign(y) == sign(q) ||
              (sign(y) == 0_r && q >= -EPSILON && q <= EPSILON))
            this->ptr_array_accuracy_values[0][0] += 1.0;
          break;
        default:
          ERR(L"Accuracy type (%d | %ls) is not managed in the switch",
              this->type_accuracy_function,
              ACC_FN_NAME[this->type_accuracy_function].c_str());
          break;
      }

      unit_it->ptr_array_errors[i_k] =
          this->activation_fn_derivative(
              *unit_it->ptr_type_activation_function, cast(*X), q, y) *
          this->loss_fn_derivative(q, y, batch_size, n_out);
    }
  }

  // copy derivative AFs to derivative neurons.
  memcpy(layer->ptr_array_neuron_units->ptr_array_errors,
         layer->ptr_array_AF_units->ptr_array_errors,
         this->batch_size * n_out * sizeof(real));
  // |END| copy derivative AFs to derivative neurons. |END|
}

void Model::compute_error_pre_train_rec_st(size_t const batch_size) {
  Layer const *const layer(this->Get__Output_Layer());

  bool const cce_accu(ACCU_FN::CROSS_ENTROPY == this->type_accuracy_function);

  size_t const n_out(
      static_cast<size_t>(layer->ptr_last_AF_unit - layer->ptr_array_AF_units));
  size_t i, t, k, k_target(n_out), k_predict(0_UZ), i_t_k;

  real q, q_max, y;
  var const *X, *Y_i_t, *Y_i_tm1;

  AF_unit const *const last_unit(layer->ptr_last_AF_unit);
  AF_unit *unit_it;

  for (i = 0_UZ; i != batch_size; ++i) {
    for (t = this->n_time_delay; t != this->seq_w; ++t) {
      Y_i_t = this->get_out(
          this->ptr_array_layers + (this->pre_training_level - 1_UZ), i, t);

      Y_i_tm1 = this->get_out(
          this->ptr_array_layers + (this->pre_training_level - 1_UZ), i,
          t - 1_UZ);

      i_t_k = this->batch_size * n_out * t + i * n_out;

      X = layer->ptr_array_pre_activation_functions + i_t_k;

      unit_it = layer->ptr_array_AF_units;

      if (cce_accu) {
        // reset predicted maximum at -Inf.
        q_max = -(std::numeric_limits<real>::max)();

        // Loop through each predicted output to find the largest output.
        for (k = 0_UZ; k != n_out; ++k) {
          // If the oIdx is the largest output.
          if ((q = cast(unit_it->ptr_array_values[i_t_k + k])) >= q_max) {
            // Store the largest output index.
            k_predict = k;

            // State the maximum predicted output.
            q_max = q;
          }

          if (Y_i_t[k] == 1_r) k_target = k;
        }

        // If the desired output index equal the largest predicted output index.
        this->ptr_array_accuracy_values[0][0] +=
            static_cast<double>(k_target == k_predict);
      }

      for (; unit_it != last_unit; ++unit_it, ++Y_i_t, ++Y_i_tm1, ++X) {
        q = cast(unit_it->ptr_array_values[i_t_k]);
        y = cast(*Y_i_t);

        this->update_loss(q, y);

        switch (this->type_accuracy_function) {
          case ACCU_FN::CROSS_ENTROPY:
            break;
          case ACCU_FN::DISTANCE:
            if (std::abs(q - y) <= this->acc_var)
              this->ptr_array_accuracy_values[0][0] += 1.0;
            break;
          case ACCU_FN::DIRECTIONAL:
            if (t == 0_UZ || sign(y - cast(*Y_i_tm1)) == sign(q - cast(*Y_i_tm1)))
              this->ptr_array_accuracy_values[0][0] += 1.0;
            break;
          case ACCU_FN::R:
            this->ptr_array_accuracy_values[0][0] += castd(y);
            this->ptr_array_accuracy_values[1][0] += castd(q);
            break;
          case ACCU_FN::SIGN:
            if (sign(y) == sign(q) ||
                (sign(y) == 0_r && q >= -EPSILON && q <= EPSILON))
              this->ptr_array_accuracy_values[0][0] += 1.0;
            break;
          default:
            ERR(L"Accuracy type (%d | %ls) is not managed in the switch",
                this->type_accuracy_function,
                ACC_FN_NAME[this->type_accuracy_function].c_str());
            break;
        }

        unit_it->ptr_array_errors[i_t_k] =
            this->activation_fn_derivative(
                *unit_it->ptr_type_activation_function, cast(*X), q, y) *
            this->loss_fn_derivative(q, y, batch_size, n_out);
      }
    }
  }

  // copy derivative AFs to derivative neurons.
  memcpy(layer->ptr_array_neuron_units->ptr_array_errors +
             this->batch_size * n_out * this->n_time_delay,
         layer->ptr_array_AF_units->ptr_array_errors +
             this->batch_size * n_out * this->n_time_delay,
         this->batch_size * n_out * (this->seq_w - this->n_time_delay) *
             sizeof(real));
  // |END| copy derivative AFs to derivative neurons. |END|
}
}  // namespace DL
