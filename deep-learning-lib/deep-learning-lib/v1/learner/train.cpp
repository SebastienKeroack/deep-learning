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
#include "deep-learning-lib/io/file.hpp"
#include "deep-learning-lib/ops/math.hpp"
#include "deep-learning-lib/v1/mem/reallocate.hpp"

#include <omp.h>

namespace DL::v1 {
void Model::set_bit_fail_limit(double const bit_fail_limit) {
  if (this->bit_fail_limit == bit_fail_limit) {
    return;
  }

  this->bit_fail_limit = bit_fail_limit;

#ifdef COMPILE_CUDA
  if (this->is_cu_initialized) {
    this->cumodel->set_bit_fail_limit(bit_fail_limit);
  }
#endif
}

void Model::Set__Maximum_Allowable_Memory(
    size_t const maximum_allowable_memory_bytes_received) {
  this->maximum_allowable_memory_bytes =
      maximum_allowable_memory_bytes_received;
}

void Model::set_loss_fn(LOSS_FN::TYPE const type) {
  if (this->type_loss_function == type) {
    return;
  }

  this->type_loss_function = type;

#ifdef COMPILE_CUDA
  if (this->is_cu_initialized) {
    this->cumodel->set_loss_fn(type);
  }
#endif
}

void Model::set_accu_fn(ACCU_FN::TYPE const type) {
  if (this->type_accuracy_function == type) {
    return;
  }

  this->type_accuracy_function = type;

#ifdef COMPILE_CUDA
  if (this->is_cu_initialized) {
    this->cumodel->set_accu_fn(type);
  }
#endif
}

void Model::set_optimizer(OPTIMIZER::TYPE const type) {
  if (this->type_optimizer_function == type) {
    return;
  }

  // Deallocate old optimizer array.
  if (this->type_optimizer_function != OPTIMIZER::NONE) {
    this->Deallocate__Parameter__Optimizer();
  }
  // |END| Deallocate old optimizer array. |END|

  // Store type optimizer function.
  this->type_optimizer_function = type;

  // allocate optimizer array(s).
  if (this->Allocate__Parameter__Optimizer() == false) {
    ERR(L"An error has been triggered from the "
        L"\"Allocate__Parameter__Optimizer()\" function.", );

    return;
  }
  // |END| allocate optimizer array(s). |END|

  // clear all derivative array(s).
  this->clear_training_arrays();

#ifdef COMPILE_CUDA
  if (this->is_cu_initialized) {
    this->cumodel->set_optimizer(type);
  }
#endif
}

void Model::clear_training_arrays(void) {
#ifdef COMPILE_CUDA
  if (this->use_cu && this->is_cu_initialized) {
    this->Clear_Training_Arrays__CUDA();
  } else
#endif
  {
    if (this->ptr_array_derivatives_parameters == nullptr) {
      this->ptr_array_derivatives_parameters =
          new real[this->number_threads * this->total_parameters_allocated];

      if (this->Use__Normalization()) {
        this->Reset__Derivative_Parameter__Normalized_Unit();
      }
    }
    memset(
        this->ptr_array_derivatives_parameters, 0,
        this->number_threads * this->total_parameters_allocated * sizeof(real));

    this->Clear_Optimizer();

    this->warm_restarts_maximum_learning_rate =
        this->warm_restarts_initial_maximum_learning_rate;
    this->warm_restarts_T_i = this->warm_restarts_initial_T_i;
  }
}

void Model::Clear_Optimizer(void) {
  size_t i;

  switch (this->type_optimizer_function) {
    case OPTIMIZER::NONE:
      break;
    case OPTIMIZER::GD:
      if (this->learning_momentum != 0_r &&
          this->ptr_array_previous_delta_parameters != nullptr) {
        memset(this->ptr_array_previous_delta_parameters, 0,
               this->total_parameters_allocated * sizeof(real));
      }
      break;
    case OPTIMIZER::IRPROP_MINUS:
      // Previous train slopes.
      if (this->ptr_array_previous_derivatives_parameters != nullptr) {
        memset(this->ptr_array_previous_derivatives_parameters, 0,
               this->total_parameters_allocated * sizeof(real));
      }
      // |END| Previous train slopes. |END|

      // Previous steps.
      if (this->ptr_array_previous_steps != nullptr) {
        for (i = 0u; i != this->total_parameters_allocated; ++i) {
          this->ptr_array_previous_steps[i] = this->rprop_delta_zero;
        }
      }
      // |END| Previous steps. |END|
      break;
    case OPTIMIZER::IRPROP_PLUS:
      this->loss_rprop = (std::numeric_limits<real>::max)();
      this->loss_rprop_tm1 = (std::numeric_limits<real>::max)();

      // Previous train slopes.
      if (this->ptr_array_previous_derivatives_parameters != nullptr) {
        memset(this->ptr_array_previous_derivatives_parameters, 0,
               this->total_parameters_allocated * sizeof(real));
      }
      // |END| Previous train slopes. |END|

      // Previous steps.
      if (this->ptr_array_previous_steps != nullptr) {
        for (i = 0u; i != this->total_parameters_allocated; ++i) {
          this->ptr_array_previous_steps[i] = this->rprop_delta_zero;
        }
      }
      // |END| Previous steps. |END|

      // Previous delta weights.
      if (this->ptr_array_previous_delta_parameters != nullptr) {
        memset(this->ptr_array_previous_delta_parameters, 0,
               this->total_parameters_allocated * sizeof(real));
      }
      // |END| Previous delta weights. |END|
      break;
    case OPTIMIZER::QUICKPROP:
      break;
    case OPTIMIZER::SARPROP:
      break;
    case OPTIMIZER::AMSBOUND:
    case OPTIMIZER::AMSGRAD:
      if (this->ptr_array_previous_biased_first_moment != nullptr) {
        memset(this->ptr_array_previous_biased_first_moment, 0,
               this->total_parameters_allocated * sizeof(real));
      }

      if (this->ptr_array_previous_biased_second_moment != nullptr) {
        memset(this->ptr_array_previous_biased_second_moment, 0,
               this->total_parameters_allocated * sizeof(real));
      }

      if (this->ptr_array_previous_biased_second_moment_hat != nullptr) {
        memset(this->ptr_array_previous_biased_second_moment_hat, 0,
               this->total_parameters_allocated * sizeof(real));
      }
      break;
    case OPTIMIZER::ADABOUND:
    case OPTIMIZER::ADAM:
    case OPTIMIZER::ADAMAX:
      if (this->ptr_array_previous_biased_first_moment != nullptr) {
        memset(this->ptr_array_previous_biased_first_moment, 0,
               this->total_parameters_allocated * sizeof(real));
      }

      if (this->ptr_array_previous_biased_second_moment != nullptr) {
        memset(this->ptr_array_previous_biased_second_moment, 0,
               this->total_parameters_allocated * sizeof(real));
      }
      break;
    case OPTIMIZER::NOSADAM:
      if (this->ptr_array_previous_biased_first_moment != nullptr) {
        memset(this->ptr_array_previous_biased_first_moment, 0,
               this->total_parameters_allocated * sizeof(real));
      }

      if (this->ptr_array_previous_biased_second_moment != nullptr) {
        memset(this->ptr_array_previous_biased_second_moment, 0,
               this->total_parameters_allocated * sizeof(real));
      }

      this->adam_previous_beta2 = 0_r;
      break;
    default:
      ERR(L"Can not reset the optimizer parameters with (%d | %ls) as the "
          L"current optimizer.",
          this->type_optimizer_function,
          OPTIMIZER_NAME[this->type_optimizer_function].c_str());
      break;
  }

  this->optimizer_time_step = 0_r;
  this->epoch_time_step = 1_r;
}

real Model::warm_restarts_decay(void) {
  real const lr_decayed(
      this->warm_restarts_minimum_learning_rate +
      0.5_r *
          (this->warm_restarts_maximum_learning_rate -
           this->warm_restarts_minimum_learning_rate) *
          (1_r + cos(this->optimizer_time_step / this->warm_restarts_T_i *
                     Math::PI<real>)));

  if (this->optimizer_time_step >= this->warm_restarts_T_i) {
    this->Clear_Optimizer();

    this->warm_restarts_T_i *= this->warm_restarts_multiplier;

    this->warm_restarts_maximum_learning_rate *=
        this->warm_restarts_decay_learning_rate;

    this->warm_restarts_maximum_learning_rate =
        std::max(this->warm_restarts_maximum_learning_rate,
                 this->warm_restarts_minimum_learning_rate);
  }

  return lr_decayed;
}

// https://arxiv.org/pdf/1711.05101.pdf: Fixing Weight Decay Regularization in
// Adam
real Model::normalized_wd(size_t const batch_size,
                          size_t const training_size) {
  return (this->weight_decay * sqrt(static_cast<real>(batch_size) /
                                    (static_cast<real>(training_size) *
                                     this->epoch_time_step)));
}

bool Model::set_layer_activation_function(
    size_t const index_layer_received,
    ACTIVATION::TYPE const type_activation_function_received) {
  if (type_activation_function_received == ACTIVATION::NONE ||
      type_activation_function_received == ACTIVATION::LENGTH) {
    ERR(L"Type activation function can not be set to (%u | %ls).",
        type_activation_function_received,
        ACTIVATION_NAME[type_activation_function_received].c_str());

    return false;
  } else if (this->ptr_array_layers == nullptr) {
    ERR(L"\"ptr_array_layers\" is a nullptr.", );

    return false;
  } else if (index_layer_received >= this->total_layers) {
    ERR(L"Layer index (%zu) overflow the number of layers in the network "
        L"(%zu).",
        index_layer_received, this->total_layers);

    return false;
  }

  return (this->set_layer_activation_function(
      this->ptr_array_layers + index_layer_received,
      type_activation_function_received));
}

LAYER_ACTIVATION::TYPE
Model::Activation_Function__To__Class_Activation_Function(
    ACTIVATION::TYPE const type_activation_function_received) const {
  LAYER_ACTIVATION::TYPE tmp_class_activation_function;

  switch (type_activation_function_received) {
    case ACTIVATION::COSINE_SYMMETRIC:
    case ACTIVATION::ELLIOT_SYMMETRIC:
    case ACTIVATION::GAUSSIAN_SYMMETRIC:
    case ACTIVATION::LINEAR_PIECE_SYMMETRIC:
    case ACTIVATION::SINE_SYMMETRIC:
    case ACTIVATION::TANH:
    case ACTIVATION::TANH_STEPWISE:
    case ACTIVATION::THRESHOLD_SYMMETRIC:
      tmp_class_activation_function = LAYER_ACTIVATION::SYMMETRIC;
      break;
    case ACTIVATION::COSINE:
    case ACTIVATION::ELU:
    case ACTIVATION::ELLIOT:
    case ACTIVATION::GAUSSIAN:
    case ACTIVATION::GAUSSIAN_STEPWISE:
    case ACTIVATION::ISRU:
    case ACTIVATION::ISRLU:
    case ACTIVATION::LINEAR:
    case ACTIVATION::LINEAR_PIECE:
    case ACTIVATION::SIGMOID:
    case ACTIVATION::SINE:
    case ACTIVATION::SIGMOID_STEPWISE:
    case ACTIVATION::THRESHOLD:
      tmp_class_activation_function = LAYER_ACTIVATION::ASYMMETRIC;
      break;
    case ACTIVATION::LEAKY_RELU:
    case ACTIVATION::PARAMETRIC_RELU:
    case ACTIVATION::RELU:
      tmp_class_activation_function = LAYER_ACTIVATION::RECTIFIER;
      break;
    case ACTIVATION::SELU:
      tmp_class_activation_function = LAYER_ACTIVATION::SELF_NORMALIZATION;
      break;
    case ACTIVATION::SOFTMAX:
      tmp_class_activation_function = LAYER_ACTIVATION::SOFTMAX;
      break;
    default:
      ERR(L"Activation function type (%d | %ls) is not managed in the switch.",
          type_activation_function_received,
          ACTIVATION_NAME[type_activation_function_received].c_str());
      tmp_class_activation_function = LAYER_ACTIVATION::NONE;
  }

  return (tmp_class_activation_function);
}

bool Model::set_layer_activation_function(
    struct Layer *const layer_it,
    ACTIVATION::TYPE const type_activation_function_received) {
  if (type_activation_function_received == ACTIVATION::NONE ||
      type_activation_function_received >= ACTIVATION::LENGTH) {
    ERR(L"Type activation function can not be set to (%u | %ls).",
        type_activation_function_received,
        ACTIVATION_NAME[type_activation_function_received].c_str());

    return false;
  } else if (layer_it == nullptr) {
    ERR(L"\"layer_it\" is a nullptr.", );

    return false;
  } else if (layer_it->type_layer == LAYER::AVERAGE_POOLING ||
             layer_it->type_layer == LAYER::MAX_POOLING ||
             layer_it->type_layer == LAYER::RESIDUAL) {
    WARN(L"Type layer (%u | %ls) is useless in this function.",
         layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());

    return true;
  }

  // Regularization on recurrent connection(s) (Independently RNN).
  bool const tmp_layer_use_regularization_constraint_recurrent_weight_default(
      layer_it->type_layer == LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT &&
      this->Check__Use__Regularization__Constraint_Recurrent_Weight__Default(
          layer_it));

  switch ((layer_it->type_activation =
               this->Activation_Function__To__Class_Activation_Function(
                   type_activation_function_received))) {
    case LAYER_ACTIVATION::ASYMMETRIC:
    case LAYER_ACTIVATION::RECTIFIER:
    case LAYER_ACTIVATION::SELF_NORMALIZATION:
    case LAYER_ACTIVATION::SYMMETRIC:
      break;
    case LAYER_ACTIVATION::SOFTMAX:
      if (layer_it != this->ptr_last_layer - 1)  // If is not the output layer
      {
        ERR(L"Can not use a softmax layer in a hidden layer.", );

        return false;
      } else if (*layer_it->ptr_number_outputs == 1u) {
        ERR(L"Softmax activation functions is only for multi class.", );

        return false;
      }
      break;
    default:
      ERR(L"Layer activation type (%d | %ls) is not managed in the switch.",
          layer_it->type_activation,
          LAYER_ACTIVATION_NAME[layer_it->type_activation].c_str());
      return false;
  }

  switch (layer_it->type_layer) {
    case LAYER::FULLY_CONNECTED:
    case LAYER::FULLY_CONNECTED_RECURRENT:
    case LAYER::SHORTCUT:
      if (this->set_layer_activation_function__AF(
              layer_it, type_activation_function_received) == false) {
        ERR(L"An error has been triggered from the "
            L"\"set_layer_activation_function__AF(ptr, %u)\" function.",
            type_activation_function_received);

        return false;
      }
      break;
    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
      if (this->set_layer_activation_function__AF_Ind_Recurrent(
              layer_it, type_activation_function_received) == false) {
        ERR(L"An error has been triggered from the "
            L"\"set_layer_activation_function__AF_Ind_Recurrent(ptr, %u)\" "
            L"function.",
            type_activation_function_received);

        return false;
      }
      break;
    case LAYER::LSTM:
      if (this->set_layer_activation_function__LSTM(
              layer_it, type_activation_function_received) == false) {
        ERR(L"An error has been triggered from the "
            L"\"set_layer_activation_function__LSTM(ptr, %u)\" function.",
            type_activation_function_received);

        return false;
      }
      break;
    default:
      ERR(L"Type layer (%u | %ls) is not managed in the switch.",
          layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
      return false;
  }

  // Regularization on recurrent connection(s) (Independently RNN).
  if (tmp_layer_use_regularization_constraint_recurrent_weight_default &&
      this->Set__Regularization__Constraint_Recurrent_Weight__Default(
          layer_it) == false) {
    ERR(L"An error has been triggered from the "
        L"\"Set__Regularization__Constraint_Recurrent_Weight__Default(ptr)\" "
        L"function.", );

    return false;
  }

  return true;
}

bool Model::set_layer_activation_function__AF(
    struct Layer *const layer_it,
    ACTIVATION::TYPE const type_activation_function_received) {
  struct AF_unit const *const tmp_ptr_last_AF_unit(layer_it->ptr_last_AF_unit);
  struct AF_unit *tmp_ptr_AF_unit_it(layer_it->ptr_array_AF_units);

  for (; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it) {
    *tmp_ptr_AF_unit_it->ptr_type_activation_function =
        type_activation_function_received;
  }

  return true;
}

bool Model::set_layer_activation_function__AF_Ind_Recurrent(
    struct Layer *const layer_it,
    ACTIVATION::TYPE const type_activation_function_received) {
  struct AF_Ind_recurrent_unit const
      *const tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit(
          layer_it->ptr_last_AF_Ind_recurrent_unit);
  struct AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it(
      layer_it->ptr_array_AF_Ind_recurrent_units);

  for (; tmp_ptr_AF_Ind_recurrent_unit_it !=
         tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit;
       ++tmp_ptr_AF_Ind_recurrent_unit_it) {
    *tmp_ptr_AF_Ind_recurrent_unit_it->ptr_type_activation_function =
        type_activation_function_received;
  }

  return true;
}

bool Model::set_layer_activation_function__LSTM(
    struct Layer *const layer_it,
    ACTIVATION::TYPE const type_activation_function_received) {
  struct BlockUnit const *const tmp_ptr_last_block_unit(
      layer_it->ptr_last_block_unit);
  struct BlockUnit *tmp_ptr_block_unit_it(layer_it->ptr_array_block_units);

  for (; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit;
       ++tmp_ptr_block_unit_it) {
    tmp_ptr_block_unit_it->activation_function_io =
        type_activation_function_received;
  }

  return true;
}
}  // namespace DL