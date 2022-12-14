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
#include "deep-learning/ops/math.hpp"
#include "deep-learning/v1/learner/model.hpp"

using namespace DL::Math;

namespace DL::v1 {
void Model::update_loss(real const q, real const y, int const thread_id) {
  double const loss(static_cast<double>(this->loss_fn(q, y)));

  switch (this->type_loss_function) {
    case LOSS_FN::ME:
    case LOSS_FN::L1:
    case LOSS_FN::MAE:
    case LOSS_FN::L2:
    case LOSS_FN::MSE:
    case LOSS_FN::RMSE:
    case LOSS_FN::MAPE:
    case LOSS_FN::SMAPE:
    case LOSS_FN::CROSS_ENTROPY:
      break;
    case LOSS_FN::BIT:
      if (std::abs(loss) >= this->bit_fail_limit)
        ++this->ptr_array_number_bit_fail[thread_id];
      return;
    default:
      ERR(L"Loss type (%d | %ls) is not managed in the switch",
          this->type_loss_function,
          LOSS_FN_NAME[this->type_loss_function].c_str());
      return;
  }

  this->ptr_array_loss_values[thread_id] += loss;

  ++this->ptr_array_number_loss[thread_id];
}

double Model::loss_fn(real const q, real const y) {
  real loss;

  switch (this->type_loss_function) {
    case LOSS_FN::BIT:
    case LOSS_FN::ME:
    case LOSS_FN::L1:
      loss = q - y;
      break;
    case LOSS_FN::MAE:
      loss = std::abs(q - y);
      break;
    case LOSS_FN::L2:
    case LOSS_FN::MSE:
    case LOSS_FN::RMSE:
      // E=(X - Y)2, square the difference
      loss = std::pow(q - y, 2_r);
      break;
    case LOSS_FN::MAPE:
      // Numerical stability.
      loss = q != 0_r ? q : std::max(q, ::EPSILON);
      loss = (q - y) / loss;
      loss = std::abs(loss);
      break;
    case LOSS_FN::SMAPE:
      loss = std::abs(q - y);
      loss /= std::abs(y) + std::abs(q);
      break;
    case LOSS_FN::CROSS_ENTROPY:
      loss = clip(q, ::EPSILON, 1_r - ::EPSILON);
      loss = y == 1_r ? -(y * std::log(loss)) : 0_r;
      // if (this->Use__Multi_Label() || this->n_out == 1_UZ)
      //   loss = -(y * log(loss) + (1_r - y) * log(1_r - loss));
      // else
      //   loss = -(y * log(loss));
      break;
    default:
      ERR(L"Loss type (%d | %ls) is not managed in the switch",
          this->type_loss_function,
          LOSS_FN_NAME[this->type_loss_function].c_str());
      return HUGE_VAL;
  }

  return static_cast<double>(loss);
}

real Model::loss_fn_derivative(real const q, real const y, size_t const bs,
                               size_t const n_out) {
  real dl;

  switch (this->type_loss_function) {
    case LOSS_FN::ME:
    case LOSS_FN::L1:
      dl = -1_r;
      break;
    case LOSS_FN::MAE:
      dl = q > y ? 1_r : (q < y ? -1_r : 0_r);
      break;
    case LOSS_FN::L2:
    case LOSS_FN::MSE:
    case LOSS_FN::RMSE:
      dl = 2_r * (q - y);
      break;
    case LOSS_FN::CROSS_ENTROPY:
      dl = clip(q, ::EPSILON, 1_r - ::EPSILON);
      dl = y == 1_r ? -1_r / dl : 0_r;
      break;
    case LOSS_FN::BIT:
      dl = q - y;
      break;
    default:
      ERR(L"Loss type (%d | %ls) is not managed in the switch",
          this->type_loss_function,
          LOSS_FN_NAME[this->type_loss_function].c_str());
      return HUGE_VAL;
  }

  if (LOSS_FN::CROSS_ENTROPY == this->type_loss_function)
    dl /= static_cast<real>(bs);
  else
    dl /= static_cast<real>(bs * n_out);

  return dl;
}

void Model::compute_loss(size_t const batch_size, real const *const *const Ym) {
  switch (this->type) {
    case MODEL::AUTOENCODER:
      if (this->pre_training_level != 0_UZ && this->_initialized__weight) {
        this->compute_loss_pre_train(batch_size);
        break;
      }
    default:
      if (this->seq_w > 1_UZ) {
        if (this->use_mp && this->is_mp_initialized)
          this->compute_loss_rec_mp(batch_size, Ym);
        else
          this->compute_loss_rec_st(batch_size, Ym);
      } else {
        if (this->use_mp && this->is_mp_initialized)
          this->compute_loss_fwp_mp(batch_size, Ym);
        else
          this->compute_loss_fwp_st(batch_size, Ym);
      }
      break;
  }
}

void Model::compute_loss_pre_train(size_t const batch_size) {
  if (this->seq_w > 1_UZ) {
    if (this->use_mp && this->is_mp_initialized)
      this->compute_loss_pre_train_rec_mp(batch_size);
    else
      this->compute_loss_pre_train_rec_st(batch_size);
  } else {
    if (this->use_mp && this->is_mp_initialized)
      this->compute_loss_pre_train_fwp_mp(batch_size);
    else
      this->compute_loss_pre_train_fwp_st(batch_size);
  }
}
}  // namespace DL
