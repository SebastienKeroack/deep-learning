/* Copyright 2022 Sébastien Kéroack. All Rights Reserved.

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
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/drivers/driver.hpp"

// Deep learning lib:
#include "deep-learning-lib/data/string.hpp"
#include "deep-learning-lib/io/logger.hpp"

// Standard:
#include <omp.h>

#include <algorithm>
#include <iostream>

namespace DL {
#if defined(_DEBUG) && defined(COMPILE_ADEPT)
void compare_adept_derivatives(v1::Model const &model) {
  std::wcout << L"Adept dx/dw:" << std::endl;
  for (size_t w(0_UZ); w != std::min(model.total_parameters, 128_UZ); ++w) {
    std::wcout << Str::to_wstring(model.ptr_array_parameters[w].get_gradient(),
                                  8, true, std::ios_base::fixed)
               << L" ";
    if (w != 0_UZ && (w + 1_UZ) % 8 == 0_UZ) std::wcout << std::endl;
  }

  std::wcout << L"Backpropagation algorithm dx/dw:" << std::endl;
  for (size_t w(0_UZ); w != std::min(model.total_parameters, 128_UZ); ++w) {
    std::wcout << Str::to_wstring(model.ptr_array_derivatives_parameters[w], 8,
                                  true, std::ios_base::fixed)
               << L" ";
    if (w != 0_UZ && (w + 1_UZ) % 8 == 0_UZ) std::wcout << std::endl;
  }
}
#endif

Driver::Driver(Dataset &dataset, ENV::TYPE const &env_type, v1::Model &model,
               size_t const minibatch_size)
    : dataset(dataset),
      env_type(env_type),
      model(model),
      minibatch_size(minibatch_size) {}

double Driver::evalt(void) {
  this->model.reset_loss();

  this->model.type_state_propagation = v1::PROPAGATION::INFERENCE;

  if (this->model.update_mem_batch_size(this->dataset.n_data) == false) {
    ERR(L"An error has been triggered from the "
        L"`Model::update_mem_batch_size(%zu)` "
        L"function.",
        this->dataset.n_data);
    return HUGE_VAL;
  } else if (this->model.weights_initialized() == false) {
    // TODO: Implement LSUV based on DatasetV1.
    ERR(L"Model weights are not initialized which most be because the model "
        L"request the weight to be initialze with `LSUV` which has not been "
        L"implemented yet using the new `Dataset`.");
    return HUGE_VAL;
  }

  if (this->model.use_mp && this->model.is_mp_initialized) {
    if (this->model.update_mem_thread_size(this->dataset.n_data) == false) {
      ERR(L"An error has been triggered from the "
          L"`Model::update_mem_thread_size(%zu)` "
          L"function.",
          this->dataset.n_data);
      return HUGE_VAL;
    }

    omp_set_num_threads(static_cast<int>(this->model.number_threads));

    this->evalt_mp();
  } else {
    this->evalt_st();
  }

  this->model.n_acc_trial =
      this->dataset.n_data * (this->dataset.seq_w - this->model.n_time_delay) *
      (v1::ACCU_FN::CROSS_ENTROPY == this->model.type_accuracy_function
           ? 1_UZ
           : this->dataset.n_out);

  double const loss(this->model.get_loss(ENV::NONE));
  this->model.set_loss(this->env_type, loss);
  this->model.set_accu(this->env_type, this->model.get_accu(ENV::NONE));
  return loss;
}

double Driver::train(void) {
  if (nullptr == this->model.ptr_array_derivatives_parameters)
    this->model.clear_training_arrays();

  this->model.type_state_propagation = v1::PROPAGATION::TRAINING;

  if (this->model.update_mem_batch_size(this->dataset.n_data) == false) {
    ERR(L"An error has been triggered from the "
        L"`Model::update_mem_batch_size(%zu)` "
        L"function.",
        this->dataset.n_data);
    return HUGE_VAL;
  } else if (this->model.weights_initialized() == false) {
    // TODO: Implement LSUV based on DatasetV1.
    ERR(L"Model weights are not initialized which most be because the model "
        L"request the weight to be initialze with `LSUV` which has not been "
        L"implemented yet using the new `Dataset`.");
    return HUGE_VAL;
  }

  if (this->model.Use__Dropout__Bernoulli() ||
      this->model.Use__Dropout__Bernoulli__Inverted() ||
      this->model.Use__Dropout__Alpha())
    this->model.Dropout_Bernoulli();
  else if (this->model.Use__Dropout__Zoneout())
    this->model.Dropout_Zoneout();

  this->model.reset_loss();

#ifdef COMPILE_ADEPT
  this->train_st();
#else
  if (this->model.use_mp && this->model.is_mp_initialized) {
    if (this->model.update_mem_thread_size(this->dataset.n_data) == false) {
      ERR(L"An error has been triggered from the "
          L"`Model::update_mem_thread_size(%zu)` "
          L"function.",
          this->dataset.n_data);
      return HUGE_VAL;
    }

    omp_set_num_threads(static_cast<int>(this->model.number_threads));

    this->train_mp();
  } else {
    this->train_st();
  }
#endif

  this->model.type_state_propagation = v1::PROPAGATION::INFERENCE;

  this->model.epoch_time_step += 1_r;

  this->model.n_acc_trial =
      this->dataset.n_data * (this->dataset.seq_w - this->model.n_time_delay) *
      (v1::ACCU_FN::CROSS_ENTROPY == this->model.type_accuracy_function
           ? 1_UZ
           : this->dataset.n_out);

  double const loss(this->model.get_loss(ENV::NONE));
  this->model.set_loss(this->env_type, loss);
  this->model.set_accu(this->env_type, this->model.get_accu(ENV::NONE));
  return loss;
}

void Driver::evalt_mp(void) {
  size_t const n_data(this->dataset.n_data),
      batch_size_max(this->model.batch_size);
  size_t batch_size(0_UZ);
  int const n_batch(static_cast<int>(
      ceil(static_cast<double>(n_data) / static_cast<double>(batch_size_max))));
  int i(0);

#ifdef COMPILE_ADEPT
  adept::Stack &global_stack(*adept::active_stack());
  global_stack.deactivate();
#endif

#pragma omp parallel private(i, batch_size)
  for (i = 0; i < n_batch; ++i) {
    batch_size =
        i + 1 != n_batch ? batch_size_max : n_data - i * batch_size_max;

#ifdef COMPILE_ADEPT
    adept::Stack stack;
    stack.pause_recording();
#endif

    this->model.forward_pass(batch_size, this->dataset.Xm + i * batch_size_max);

    this->model.compute_loss(batch_size, this->dataset.Ym + i * batch_size_max);
  }

#ifdef COMPILE_ADEPT
  global_stack.activate();
#endif

  this->model.merge_mp_accu_loss();
}

void Driver::evalt_st(void) {
  size_t const n_data(this->dataset.n_data),
      batch_size_max(this->model.batch_size),
      n_batch(static_cast<size_t>(ceil(static_cast<double>(n_data) /
                                       static_cast<double>(batch_size_max))));
  size_t i, batch_size;

  for (i = 0_UZ; i != n_batch; ++i) {
    batch_size =
        i + 1_UZ != n_batch ? batch_size_max : n_data - i * batch_size_max;

    this->model.forward_pass(batch_size, this->dataset.Xm + i * batch_size_max);

    this->model.compute_loss(batch_size, this->dataset.Ym + i * batch_size_max);
  }
}

void Driver::train_mp(void) {
  bool const irprop(
      v1::OPTIMIZER::IRPROP_MINUS == this->model.type_optimizer_function ||
      v1::OPTIMIZER::IRPROP_PLUS == this->model.type_optimizer_function);

  if (irprop) this->model.loss_rprop_tm1 = this->model.loss_rprop;

  size_t const n_data(this->dataset.n_data),
      batch_size_max(this->model.batch_size);
  size_t batch_size(0_UZ);
  int const n_batch(static_cast<int>(
      ceil(static_cast<double>(n_data) / static_cast<double>(batch_size_max))));
  int i(0);

#pragma omp parallel private(i, batch_size)
  for (i = 0; i < n_batch; ++i) {
    batch_size =
        i + 1 != n_batch ? batch_size_max : n_data - i * batch_size_max;

    this->model.forward_pass(batch_size, this->dataset.Xm + i * batch_size_max);

    this->model.compute_error(batch_size,
                              this->dataset.Ym + i * batch_size_max);

    this->model.backward_pass(batch_size);

    this->model.update_derivatives(batch_size, this->model.ptr_array_layers + 1,
                                   this->model.ptr_last_layer);
  }

  this->model.merge_mp_derivatives(0_UZ, this->model.total_parameters);

  this->model.merge_mp_accu_loss();

  if (irprop) this->model.loss_rprop = abs(this->model.get_loss(ENV::NONE));

  this->model.update_weights_mp(this->dataset.n_data, this->dataset.n_data);
}

void Driver::train_st(void) {
  bool const irprop(
      v1::OPTIMIZER::IRPROP_MINUS == this->model.type_optimizer_function ||
      v1::OPTIMIZER::IRPROP_PLUS == this->model.type_optimizer_function);

  if (irprop) this->model.loss_rprop_tm1 = this->model.loss_rprop;

  size_t const n_data(this->dataset.n_data),
      batch_size_max(this->model.batch_size),
      n_batch(static_cast<size_t>(ceil(static_cast<double>(n_data) /
                                       static_cast<double>(batch_size_max))));
  size_t i, batch_size;

#ifdef COMPILE_ADEPT
  for (i = 0_UZ; i != n_batch; ++i) {
    batch_size =
        i + 1_UZ != n_batch ? batch_size_max : n_data - i * batch_size_max;

    adept::active_stack()->new_recording();
    adept::active_stack()->continue_recording();

    this->model.forward_pass(batch_size, this->dataset.Xm + i * batch_size_max);

    this->model.compute_grad_adept(batch_size,
                                   this->dataset.Ym + i * batch_size_max);

    adept::active_stack()->reverse();
    adept::active_stack()->pause_recording();

#ifdef _DEBUG
    if (i == 0_UZ) {
      this->model.compute_error(batch_size,
                                this->dataset.Ym + i * batch_size_max);

      this->model.backward_pass(batch_size);

      this->model.update_derivatives(batch_size,
                                     this->model.ptr_array_layers + 1,
                                     this->model.ptr_last_layer);

      INFO(
          L"Compare derivatives between Adept and backpropagation algorithm on "
          L"batch #1:");
      compare_adept_derivatives(this->model);
      continue;
    }
#endif

    this->model.compute_loss(batch_size, this->dataset.Ym + i * batch_size_max);

    this->model.update_derivatives_adept();
  }
#else
  for (i = 0_UZ; i != n_batch; ++i) {
    batch_size =
        i + 1_UZ != n_batch ? batch_size_max : n_data - i * batch_size_max;

    this->model.forward_pass(batch_size, this->dataset.Xm + i * batch_size_max);

    this->model.compute_error(batch_size,
                              this->dataset.Ym + i * batch_size_max);

    this->model.backward_pass(batch_size);

    this->model.update_derivatives(batch_size, this->model.ptr_array_layers + 1,
                                   this->model.ptr_last_layer);
  }
#endif

  if (irprop) this->model.loss_rprop = abs(this->model.get_loss(ENV::NONE));

  this->model.update_weights_st(this->dataset.n_data, this->dataset.n_data);
}
}  // namespace DL