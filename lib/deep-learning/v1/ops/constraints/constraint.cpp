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

#include "deep-learning/v1/learner/model.hpp"
#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/ops/math.hpp"

#include <omp.h>

using namespace DL::Math;

namespace DL::v1 {
bool Layer::Use__Regularization__Constraint_Recurrent_Weight(void) const {
  return this->constraint_recurrent_weight_lower_bound != 0_r ||
         this->constraint_recurrent_weight_upper_bound != 0_r;
}

bool Model::Check__Use__Regularization__Constraint_Recurrent_Weight__Default(
    size_t const index_layer_received) const {
  return this->Check__Use__Regularization__Constraint_Recurrent_Weight__Default(
      this->ptr_array_layers + index_layer_received);
}

bool Model::Check__Use__Regularization__Constraint_Recurrent_Weight__Default(
    Layer *const layer) const {
  std::pair<real, real> const l_u_bound(
      this->Compute__Regularization__Constraint_Recurrent_Weight__Default(
          layer));
  real const lower_bound(l_u_bound.first), upper_bound(l_u_bound.second);

  return layer->constraint_recurrent_weight_lower_bound <=
             lower_bound + 1e-8_r &&
         layer->constraint_recurrent_weight_lower_bound >=
             lower_bound - 1e-8_r &&
         layer->constraint_recurrent_weight_upper_bound <=
             upper_bound + 1e-8_r &&
         layer->constraint_recurrent_weight_upper_bound >= upper_bound - 1e-8_r;
}

std::pair<real, real>
Model::Compute__Regularization__Constraint_Recurrent_Weight__Default(
    size_t const index_layer_received) const {
  return this->Compute__Regularization__Constraint_Recurrent_Weight__Default(
      this->ptr_array_layers + index_layer_received);
}

std::pair<real, real>
Model::Compute__Regularization__Constraint_Recurrent_Weight__Default(
    Layer *const layer) const {
  real const mag(clip(this->clip_gradient, 2_r, 10_r)),
      seq_w(static_cast<real>(this->seq_w));
  real lower_bound, upper_bound;

  switch (layer->type_activation) {
    case LAYER_ACTIVATION::NONE:
      lower_bound = -1_r;
      upper_bound = 1_r;
      break;
    case LAYER_ACTIVATION::SYMMETRIC:
      lower_bound = -std::pow(mag / std::pow(0.9_r, seq_w / 10_r), 1_r / seq_w);
      upper_bound = std::pow(mag / std::pow(0.9_r, seq_w / 10_r), 1_r / seq_w);
      break;
    case LAYER_ACTIVATION::ASYMMETRIC:
    case LAYER_ACTIVATION::SOFTMAX:
    case LAYER_ACTIVATION::RECTIFIER:
    case LAYER_ACTIVATION::SELF_NORMALIZATION:
      lower_bound = -std::pow(mag, 1_r / seq_w);
      upper_bound = std::pow(mag, 1_r / seq_w);
      break;
    default:
      ERR(L"Layer activation type (%d | %ls) is not managed in "
          L"the switch.",
          layer->type_activation,
          LAYER_ACTIVATION_NAME[layer->type_activation].c_str());
      return std::make_pair(0_r, 0_r);
  }

  return std::make_pair(lower_bound, upper_bound);
}

bool Model::Set__Regularization__Constraint_Recurrent_Weight__Default(
    size_t const index_layer_received) {
  if (index_layer_received >= this->total_layers) {
    ERR(L"Layer received (%zu) overflow the number of layers (%zu) "
        L"in the neural network.",
        index_layer_received, this->total_layers);
    return false;
  } else if (this->ptr_array_layers == nullptr) {
    ERR(L"`ptr_array_layers` is a nullptr.");
    return false;
  }

  return this->Set__Regularization__Constraint_Recurrent_Weight__Default(
      this->ptr_array_layers + index_layer_received);
}

bool Model::Set__Regularization__Constraint_Recurrent_Weight__Default(
    Layer *const layer) {
  if (layer == nullptr) {
    ERR(L"`layer` is a nullptr.");
    return false;
  } else if (layer == this->ptr_array_layers) {
    ERR(L"Layer received as argument is the input layer.");
    return false;
  } else if (layer == this->ptr_last_layer - 1) {
    ERR(L"Layer received as argument is the output layer.");
    return false;
  }

  switch (layer->type_layer) {
    case LAYER::FULLY_CONNECTED_RECURRENT:
    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
    case LAYER::LSTM:
      break;
    default:
      ERR(L"Can not constraining the recurrent weight in the "
          L"layer %zu with (%d | %ls) as the type layer.",
          static_cast<size_t>(layer - this->ptr_array_layers),
          layer->type_layer, LAYER_NAME[layer->type_layer].c_str());
      return false;
  }

  std::pair<real, real> const l_u_bound(
      this->Compute__Regularization__Constraint_Recurrent_Weight__Default(
          layer));
  real const lower_bound(l_u_bound.first), upper_bound(l_u_bound.second);

  return this->Set__Regularization__Constraint_Recurrent_Weight(
      layer, lower_bound, upper_bound);
}

bool Model::Set__Regularization__Constraint_Recurrent_Weight(
    size_t const index_layer_received,
    real const constraint_recurrent_weight_lower_bound_received,
    real const constraint_recurrent_weight_upper_bound_received) {
  if (index_layer_received >= this->total_layers) {
    ERR(L"Layer received (%zu) overflow the number of layers (%zu) "
        L"in the neural network.",
        index_layer_received, this->total_layers);
    return false;
  } else if (this->ptr_array_layers == nullptr) {
    ERR(L"\"ptr_array_layers\" is a nullptr.");
    return false;
  }

  return this->Set__Regularization__Constraint_Recurrent_Weight(
      this->ptr_array_layers + index_layer_received,
      constraint_recurrent_weight_lower_bound_received,
      constraint_recurrent_weight_upper_bound_received);
}

bool Model::Set__Regularization__Constraint_Recurrent_Weight(
    Layer *const layer,
    real const constraint_recurrent_weight_lower_bound_received,
    real const constraint_recurrent_weight_upper_bound_received) {
  if (layer == nullptr) {
    ERR(L"`layer` is a nullptr.");
    return false;
  } else if (layer == this->ptr_array_layers) {
    ERR(L"Layer received as argument is the input layer.");
    return false;
  } else if (layer == this->ptr_last_layer - 1) {
    ERR(L"Layer received as argument is the output layer.");
    return false;
  } else if (constraint_recurrent_weight_lower_bound_received >
             constraint_recurrent_weight_upper_bound_received) {
    ERR(L"Lower l_u_bound (%f) can not be greater than upper l_u_bound "
        L"(%f).",
        constraint_recurrent_weight_lower_bound_received,
        constraint_recurrent_weight_upper_bound_received);
    return false;
  } else if (layer->constraint_recurrent_weight_lower_bound ==
                 constraint_recurrent_weight_lower_bound_received &&
             layer->constraint_recurrent_weight_upper_bound ==
                 constraint_recurrent_weight_upper_bound_received) {
    return true;
  }

  switch (layer->type_layer) {
    case LAYER::FULLY_CONNECTED_RECURRENT:
    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
    case LAYER::LSTM:
      break;
    default:
      ERR(L"Can not constraining the recurrent weight in the "
          L"layer %zu with (%d | %ls) as the type layer.",
          static_cast<size_t>(layer - this->ptr_array_layers),
          layer->type_layer, LAYER_NAME[layer->type_layer].c_str());
      return false;
  }

  if (layer->constraint_recurrent_weight_lower_bound == 0_r &&
      layer->constraint_recurrent_weight_upper_bound == 0_r &&
      (constraint_recurrent_weight_lower_bound_received != 0_r ||
       constraint_recurrent_weight_upper_bound_received != 0_r)) {
    ++this->total_constraint_recurrent_weight_layers;
  } else if ((layer->constraint_recurrent_weight_lower_bound != 0_r ||
              layer->constraint_recurrent_weight_upper_bound != 0_r) &&
             constraint_recurrent_weight_lower_bound_received == 0_r &&
             constraint_recurrent_weight_upper_bound_received == 0_r) {
    --this->total_constraint_recurrent_weight_layers;
  }

  layer->constraint_recurrent_weight_lower_bound =
      constraint_recurrent_weight_lower_bound_received;
  layer->constraint_recurrent_weight_upper_bound =
      constraint_recurrent_weight_upper_bound_received;

  // Mirror layer.
  if (this->type == MODEL::AUTOENCODER &&
      layer < this->Get__End_Layer__Active() - 1  // Get last active layer.
      && this->Set__Regularization__Constraint_Recurrent_Weight(
             this->ptr_last_layer -
                 static_cast<size_t>(layer - this->ptr_array_layers) - 1,
             constraint_recurrent_weight_lower_bound_received,
             constraint_recurrent_weight_upper_bound_received)) {
    ERR(L"An error has been triggered from the "
        L"`Set__Regularization__Constraint_Recurrent_Weight(ptr, %f, %f)` "
        L"function.",
        constraint_recurrent_weight_lower_bound_received,
        constraint_recurrent_weight_upper_bound_received);
    return false;
  }
  // |END| Mirror layer. |END|

  return true;
}

void Model::Update_Weight_Regularization__Constraint_Recurrent_Weight(
    size_t const start_index_received, size_t const end_index_received) {
  Layer const *const last_layer(this->ptr_last_layer - 1);
  Layer *layer_it(this->ptr_array_layers + 1);

  for (; layer_it != last_layer; ++layer_it) {
    if (*layer_it->ptr_first_connection_index < start_index_received)
      continue;
    else if (*layer_it->ptr_last_connection_index > end_index_received)
      break;

    if (layer_it->Use__Regularization__Constraint_Recurrent_Weight()) {
      switch (layer_it->type_layer) {
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          this->Update_Weight_Regularization__Constraint_Recurrent_Weight__FC_Ind_RNN(
              layer_it);
          break;
        default:
          ERR(L"Type layer (%d | %ls) is not managed in the switch.",
              layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
          return;
      }
    }
  }
}

void Model::
    Update_Weight_Regularization__Constraint_Recurrent_Weight__FC_Ind_RNN(
        Layer const *const layer_it) {
  AF_Ind_recurrent_unit const *const unit(
      layer_it->ptr_array_AF_Ind_recurrent_units);

  size_t const n_units(
      static_cast<size_t>(layer_it->ptr_last_AF_Ind_recurrent_unit - unit));

  real const lower_bound(layer_it->constraint_recurrent_weight_lower_bound),
      upper_bound(layer_it->constraint_recurrent_weight_upper_bound);

  var *parameter_it(this->ptr_array_parameters +
                    *unit->ptr_recurrent_connection_index);
  var const *parameter_end(parameter_it + n_units);

  for (; parameter_it != parameter_end; ++parameter_it)
    *parameter_it = clip(cast(*parameter_it), lower_bound, upper_bound);
}
}  // namespace DL
