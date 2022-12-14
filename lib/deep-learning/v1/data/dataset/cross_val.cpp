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

#include "deep-learning/v1/data/datasets.hpp"
#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"

#include <iostream>

namespace DL::v1 {
CrossVal::CrossVal(void) : DatasetV1() {
  this->p_type_dataset_process = DATASET::CROSS_VAL;
}

void CrossVal::Shuffle(void) {
  size_t tmp_swap, i;
  size_t tmp_randomize_index;

  for (i = this->p_n_data; i-- != this->p_str_i;) {
    this->Generator_Random.range(this->p_str_i, i);

    tmp_randomize_index = this->Generator_Random();

    // Store the index to swap from the remaining index at "tmp_randomize_index"
    tmp_swap = this->ptr_array_stochastic_index[tmp_randomize_index];

    // Get remaining index starting at index "i"
    // And store it to the remaining index at "tmp_randomize_index"
    this->ptr_array_stochastic_index[tmp_randomize_index] =
        this->ptr_array_stochastic_index[i];

    // Store the swapped index at the index "i"
    this->ptr_array_stochastic_index[i] = tmp_swap;
  }
}

void CrossVal::Set__Use__Shuffle(bool const use_shuffle_received) {
  this->use_shuffle = use_shuffle_received;

  if (use_shuffle_received == false)
    for (size_t tmp_index(0_UZ); tmp_index != this->p_n_data;
         ++tmp_index)
      this->ptr_array_stochastic_index[tmp_index] = tmp_index;
}

void CrossVal::reset(void) {
  this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold;

  this->ptr_array_outputs_array_k_sub_fold =
      this->ptr_array_outputs_array_k_fold;

  this->number_examples = this->number_examples_validating;
}

bool CrossVal::Initialize(void) {
  this->DatasetV1::Initialize();

  this->p_type_dataset_process = DATASET::CROSS_VAL;

  return true;
}

bool CrossVal::Initialize__Fold(bool const use_shuffle_received,
                                   size_t const number_k_fold_received,
                                   size_t const number_k_sub_fold_received) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (number_k_fold_received < 2_UZ) {
    ERR(L"K-fold must be at least 2.");
    return false;
  }

  if (this->Set__Desired_K_Fold(number_k_fold_received,
                                number_k_sub_fold_received) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Desired_Data_Per_Batch(%zu, %zu)` function.", number_k_fold_received,
        number_k_sub_fold_received);
    return false;
  }

  this->ptr_array_stochastic_index = new size_t[this->p_n_data];

  this->Set__Use__Shuffle(use_shuffle_received);

  if (use_shuffle_received)
    for (size_t i(0_UZ); i != this->p_n_data; ++i)
      this->ptr_array_stochastic_index[i] = i;

  this->Generator_Random.seed(static_cast<unsigned int>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count()));

  return true;
}

bool CrossVal::Set__Desired_K_Fold(size_t const number_k_fold_received,
                                   size_t const number_k_sub_fold_received) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (number_k_fold_received < 2_UZ) {
    ERR(L"K-fold must be at least 2.");
    return false;
  } else if (number_k_fold_received >
             this->DatasetV1::get_n_data()) {
    ERR(L"K-fold (%zu) must not be greater than total number of data (%zu).",
        number_k_fold_received,
        this->DatasetV1::get_n_data());
    return false;
  }

  this->number_k_fold = number_k_fold_received;
  this->number_examples_per_fold =
      this->DatasetV1::get_n_data() / number_k_fold_received;

  if (this->ptr_array_inputs_array_k_fold == nullptr) {
    this->ptr_array_inputs_array_k_fold = new real const
        *[(number_k_fold_received - 1_UZ) * this->number_examples_per_fold];
  } else {
    this->ptr_array_inputs_array_k_fold =
        Mem::reallocate_ptofpt<real const *, false>(
            this->ptr_array_inputs_array_k_fold,
            (number_k_fold_received - 1_UZ) * this->number_examples_per_fold,
            this->number_examples_training);
    if (this->ptr_array_inputs_array_k_fold == nullptr) {
      ERR(L"Can not allocate %zu bytes.", (number_k_fold_received - 1_UZ) *
                                              this->number_examples_per_fold *
                                              sizeof(real *));
      return false;
    }
  }
  this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold;

  if (this->ptr_array_outputs_array_k_fold == nullptr) {
    this->ptr_array_outputs_array_k_fold = new real const
        *[(number_k_fold_received - 1_UZ) * this->number_examples_per_fold];
  } else {
    this->ptr_array_outputs_array_k_fold =
        Mem::reallocate_ptofpt<real const *, false>(
            this->ptr_array_outputs_array_k_fold,
            (number_k_fold_received - 1_UZ) * this->number_examples_per_fold,
            this->number_examples_training);
    if (this->ptr_array_outputs_array_k_fold == nullptr) {
      ERR(L"Can not allocate %zu bytes.",
             (number_k_fold_received - 1_UZ) * this->number_examples_per_fold *
                 sizeof(real *));
      return false;
    }
  }
  this->ptr_array_outputs_array_k_sub_fold =
      this->ptr_array_outputs_array_k_fold;

  this->number_examples_training =
      (number_k_fold_received - 1_UZ) * this->number_examples_per_fold;

  if (this->ptr_array_inputs_array_validation == nullptr) {
    this->ptr_array_inputs_array_validation = new real const
        *[this->DatasetV1::get_n_data() - this->number_examples_training];
  } else {
    this->ptr_array_inputs_array_validation =
        Mem::reallocate_ptofpt<real const *, false>(
            this->ptr_array_inputs_array_validation,
            this->DatasetV1::get_n_data() -
                this->number_examples_training,
            this->number_examples_validating);
    if (this->ptr_array_inputs_array_validation == nullptr) {
      ERR(L"Can not allocate %zu bytes.",
             this->DatasetV1::get_n_data() -
                 this->number_examples_training * sizeof(real *));
      return false;
    }
  }

  if (this->ptr_array_outputs_array_validation == nullptr) {
    this->ptr_array_outputs_array_validation = new real const
        *[this->DatasetV1::get_n_data() - this->number_examples_training];
  } else {
    this->ptr_array_outputs_array_validation =
        Mem::reallocate_ptofpt<real const *, false>(
            this->ptr_array_outputs_array_validation,
            this->DatasetV1::get_n_data() -
                this->number_examples_training,
            this->number_examples_validating);
    if (this->ptr_array_outputs_array_validation == nullptr) {
      ERR(L"Can not allocate %zu bytes.",
             this->DatasetV1::get_n_data() -
                 this->number_examples_training * sizeof(real *));
      return false;
    }
  }

  this->number_examples = this->number_examples_validating =
      this->DatasetV1::get_n_data() - this->number_examples_training;

  if (number_k_sub_fold_received > this->number_examples_training) {
    ERR(L"K-sub-fold (%zu) > (%zu) amount of data from the training set.",
        number_k_sub_fold_received,
        this->number_examples_training);
    return false;
  }

  this->number_k_sub_fold =
      number_k_sub_fold_received == 0_UZ ? 1_UZ : number_k_sub_fold_received;

  // 8 / 2 = 4
  // 31383 / 240 = 130.7625
  double const tmp_number_examples_per_sub_fold(
      static_cast<double>(this->number_examples_training) /
      static_cast<double>(this->number_k_sub_fold));

  // 4
  // 130
  this->number_examples_per_sub_iteration =
      static_cast<size_t>(tmp_number_examples_per_sub_fold);

  // 4 + (4 - 4) * 2 = 0
  // 130 + (130.7625 - 130) * 240 = 183
  this->number_examples_last_sub_iteration =
      this->number_examples_per_sub_iteration +
      static_cast<size_t>(
          (tmp_number_examples_per_sub_fold -
           static_cast<double>(this->number_examples_per_sub_iteration)) *
          static_cast<double>(this->number_k_sub_fold));

  return true;
}

bool CrossVal::Increment_Fold(size_t const fold_received) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  }

  if (fold_received >= this->number_k_fold)
    return false;

  size_t const tmp_number_examples_training_per_fold(
      this->number_examples_per_fold),
      tmp_number_examples_validating(this->number_examples_validating),
      tmp_validating_index_start(fold_received *
                                 tmp_number_examples_training_per_fold),
      *tmp_ptr_array_stochastic_index(this->ptr_array_stochastic_index);
  size_t tmp_example_index;

  if (tmp_validating_index_start == 0_UZ)  // First iteration.
  {
    // Validation sample.
    // (0, 1, 2)   [3, 4, 5   6, 7, 8   9, 10, 11]
    for (tmp_example_index = 0_UZ;
         tmp_example_index != tmp_number_examples_validating;
         ++tmp_example_index) {
      this->ptr_array_inputs_array_validation[tmp_example_index] =
          this->Xm
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
      this->ptr_array_outputs_array_validation[tmp_example_index] =
          this->Ym
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
    }
    // |END| Validation sample. |END|

    // Training sample.
    tmp_ptr_array_stochastic_index += tmp_number_examples_validating;

    // (0, 1, 2)   [3, 4, 5   6, 7, 8   9, 10, 11]
    for (tmp_example_index = 0_UZ;
         tmp_example_index != this->number_examples_training;
         ++tmp_example_index) {
      this->ptr_array_inputs_array_k_fold[tmp_example_index] =
          this->Xm
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
      this->ptr_array_outputs_array_k_fold[tmp_example_index] =
          this->Ym
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
    }
    // |END| Training sample. |END|
  } else if (tmp_validating_index_start ==
             this->number_examples_training)  // Last iteration.
  {
    // Training sample.
    // [0, 1, 2   3, 4, 5   6, 7, 8]   (9, 10, 11)
    for (tmp_example_index = 0_UZ;
         tmp_example_index != this->number_examples_training;
         ++tmp_example_index) {
      this->ptr_array_inputs_array_k_fold[tmp_example_index] =
          this->Xm
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
      this->ptr_array_outputs_array_k_fold[tmp_example_index] =
          this->Ym
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
    }
    // |END| Training sample. |END|

    // Validation sample.
    tmp_ptr_array_stochastic_index += this->number_examples_training;

    // [0, 1, 2   3, 4, 5   6, 7, 8]   (9, 10, 11)
    for (tmp_example_index = 0_UZ;
         tmp_example_index != tmp_number_examples_validating;
         ++tmp_example_index) {
      this->ptr_array_inputs_array_validation[tmp_example_index] =
          this->Xm
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
      this->ptr_array_outputs_array_validation[tmp_example_index] =
          this->Ym
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
    }
    // |END| Validation sample. |END|
  } else  // The remaining iterations.
  {
    // Training sample.
    // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
    for (tmp_example_index = 0_UZ;
         tmp_example_index != tmp_validating_index_start; ++tmp_example_index) {
      this->ptr_array_inputs_array_k_fold[tmp_example_index] =
          this->Xm
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
      this->ptr_array_outputs_array_k_fold[tmp_example_index] =
          this->Ym
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
    }

    // Validation sample.
    tmp_ptr_array_stochastic_index += tmp_validating_index_start;

    // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
    for (tmp_example_index = 0_UZ;
         tmp_example_index != tmp_number_examples_validating;
         ++tmp_example_index) {
      this->ptr_array_inputs_array_validation[tmp_example_index] =
          this->Xm
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
      this->ptr_array_outputs_array_validation[tmp_example_index] =
          this->Ym
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
    }
    // |END| Validation sample. |END|

    // Training sample.
    tmp_ptr_array_stochastic_index =
        this->ptr_array_stochastic_index + tmp_number_examples_validating;

    // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
    for (tmp_example_index = tmp_validating_index_start;
         tmp_example_index != this->number_examples_training;
         ++tmp_example_index) {
      this->ptr_array_inputs_array_k_fold[tmp_example_index] =
          this->Xm
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
      this->ptr_array_outputs_array_k_fold[tmp_example_index] =
          this->Ym
              [tmp_ptr_array_stochastic_index[tmp_example_index +
                                              this->p_str_i]];
    }
    // |END| Training sample. |END|
  }

  return true;
}

bool CrossVal::Increment_Sub_Fold(size_t const sub_fold_received) {
  if (this->number_k_sub_fold == 1_UZ)
    return true;
  else if (sub_fold_received >= this->number_k_sub_fold)
    return false;

  size_t const tmp_data_per_sub_fold(
      sub_fold_received + 1_UZ != this->number_k_sub_fold
          ? this->number_examples_per_sub_iteration
          : this->number_examples_last_sub_iteration);

  this->ptr_array_inputs_array_k_sub_fold =
      this->ptr_array_inputs_array_k_fold +
      sub_fold_received * this->number_examples_per_sub_iteration;

  this->ptr_array_outputs_array_k_sub_fold =
      this->ptr_array_outputs_array_k_fold +
      sub_fold_received * this->number_examples_per_sub_iteration;

  this->number_examples = tmp_data_per_sub_fold;

  return true;
}

bool CrossVal::Get__Use__Shuffle(void) const {
  return this->use_shuffle;
}

bool CrossVal::Deallocate(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_inputs_array_k_fold);
  SAFE_DELETE_ARRAY(this->ptr_array_outputs_array_k_fold);
  SAFE_DELETE_ARRAY(this->ptr_array_inputs_array_validation);
  SAFE_DELETE_ARRAY(this->ptr_array_outputs_array_validation);

  SAFE_DELETE_ARRAY(this->ptr_array_stochastic_index);

  if (this->DatasetV1::Deallocate() == false) {
    ERR(L"An error has been triggered from the "
        L"`Deallocate()` function.");
    return false;
  }

  return true;
}

size_t CrossVal::get_n_data(void) const {
  return this->number_examples;
}

size_t CrossVal::get_n_batch(void) const {
  return this->number_k_fold;
}

size_t CrossVal::Get__Number_Sub_Batch(void) const {
  return this->number_k_sub_fold;
}

size_t CrossVal::Get__Number_Examples_Training(void) const {
  return this->number_examples_training;
}

size_t CrossVal::Get__Number_Examples_Validating(void) const {
  return this->number_examples_validating;
}

size_t CrossVal::Get__Number_Examples_Per_Fold(void) const {
  return this->number_examples_per_fold;
}

size_t CrossVal::Get__Number_Examples_Per_Sub_Iteration(void) const {
  return this->number_examples_per_sub_iteration;
}

size_t CrossVal::Get__Number_Examples_Last_Sub_Iteration(void) const {
  return this->number_examples_last_sub_iteration;
}

real CrossVal::get_inp(size_t const index_received,
                             size_t const sub_index_received) const {
  return this->ptr_array_inputs_array_k_sub_fold[index_received]
                                                 [sub_index_received];
}

real CrossVal::get_out(size_t const index_received,
                              size_t const sub_index_received) const {
  return this->ptr_array_outputs_array_k_sub_fold[index_received]
                                                  [sub_index_received];
}

real const *const CrossVal::get_inp(size_t const index_received) const {
  return this->ptr_array_inputs_array_k_sub_fold[index_received];
}

real const *const CrossVal::get_out(size_t const index_received) const {
  return this->ptr_array_outputs_array_k_sub_fold[index_received];
}

real const *const *const CrossVal::Get__Input_Array(void) const {
  return this->ptr_array_inputs_array_k_sub_fold;
}

real const *const *const CrossVal::Get__Output_Array(void) const {
  return this->ptr_array_outputs_array_k_sub_fold;
}

double CrossVal::train_mp(Model *const model) {
  double tmp_summation_loss(0.0), tmp_summation_accurancy(0.0);

  if (this->use_shuffle)
    this->Shuffle();

  size_t tmp_fold_index, tmp_sub_fold_index;

  for (tmp_fold_index = 0_UZ; tmp_fold_index != this->number_k_fold;
       ++tmp_fold_index) {
    if (this->Increment_Fold(tmp_fold_index)) {
      for (tmp_sub_fold_index = 0_UZ;
           tmp_sub_fold_index != this->number_k_sub_fold;
           ++tmp_sub_fold_index) {
        if (this->Increment_Sub_Fold(tmp_sub_fold_index)) {
          this->Train_Epoch_OpenMP(model);

          model->update_weights_mp(
              this->get_n_data(),
              this->DatasetV1::get_n_data());
        } else {
          ERR(L"An error has been triggered from the "
              L"`Increment_Sub_Fold(%zu)` function.",
              tmp_sub_fold_index);
          return false;
        }
      }

      this->ptr_array_inputs_array_k_sub_fold =
          this->ptr_array_inputs_array_validation;
      this->ptr_array_outputs_array_k_sub_fold =
          this->ptr_array_outputs_array_validation;
      this->number_examples = this->number_examples_validating;

      tmp_summation_loss += this->Test_Epoch_OpenMP(model);
      tmp_summation_accurancy += this->compute_accuracy(
          this->get_n_data(), this->Get__Input_Array(),
          this->Get__Output_Array(), model);
    } else {
      ERR(L"An error has been triggered from the "
          L"`Increment_Fold(%zu)` function.", tmp_fold_index);
      return false;
    }
  }

  this->reset();

  model->epoch_time_step += 1_r;

  tmp_summation_loss /= static_cast<double>(this->number_k_fold);
  tmp_summation_accurancy /= static_cast<double>(this->number_k_fold);

  model->set_loss(ENV::TRAIN, tmp_summation_loss);
  model->set_accu(ENV::TRAIN, tmp_summation_accurancy);

  return model->get_loss(ENV::TRAIN);
}

double CrossVal::train_st(Model *const model) {
  double tmp_summation_loss(0.0), tmp_summation_accurancy(0.0);

  if (this->use_shuffle)
    this->Shuffle();

  size_t tmp_fold_index, tmp_sub_fold_index;

  for (tmp_fold_index = 0_UZ; tmp_fold_index != this->number_k_fold;
       ++tmp_fold_index) {
    if (this->Increment_Fold(tmp_fold_index)) {
      for (tmp_sub_fold_index = 0_UZ;
           tmp_sub_fold_index != this->number_k_sub_fold;
           ++tmp_sub_fold_index) {
        if (this->Increment_Sub_Fold(tmp_sub_fold_index)) {
          this->Train_Epoch_Loop(model);

          model->update_weights_st(
              this->get_n_data(),
              this->DatasetV1::get_n_data());
        } else {
          ERR(L"An error has been triggered from the "
              L"`Increment_Sub_Fold(%zu)` function.",
              tmp_sub_fold_index);
          return false;
        }
      }

      this->ptr_array_inputs_array_k_sub_fold =
          this->ptr_array_inputs_array_validation;
      this->ptr_array_outputs_array_k_sub_fold =
          this->ptr_array_outputs_array_validation;
      this->number_examples = this->number_examples_validating;

      tmp_summation_loss += this->Test_Epoch_Loop(model);
      tmp_summation_accurancy += this->compute_accuracy(
          this->get_n_data(), this->Get__Input_Array(),
          this->Get__Output_Array(), model);
    } else {
      ERR(L"An error has been triggered from the "
          L"`Increment_Fold(%zu)` function.", tmp_fold_index);
      return false;
    }
  }

  this->reset();

  model->epoch_time_step += 1_r;

  tmp_summation_loss /= static_cast<double>(this->number_k_fold);
  tmp_summation_accurancy /= static_cast<double>(this->number_k_fold);

  model->set_loss(ENV::TRAIN, tmp_summation_loss);
  model->set_accu(ENV::TRAIN, tmp_summation_accurancy);

  return model->get_loss(ENV::TRAIN);
}

double CrossVal::Test_Epoch_OpenMP(Model *const model) {
  size_t const n_data(this->get_n_data()),
      tmp_maximum_batch_size(model->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_index(0_UZ), tmp_batch_size(0_UZ);

  model->reset_loss();

  model->type_state_propagation = PROPAGATION::INFERENCE;

#pragma omp parallel private(tmp_batch_index, tmp_batch_size)
  for (tmp_batch_index = 0_UZ; tmp_batch_index != tmp_number_batchs;
       ++tmp_batch_index) {
    tmp_batch_size =
        tmp_batch_index + 1_UZ != tmp_number_batchs
            ? tmp_maximum_batch_size
            : n_data - tmp_batch_index * tmp_maximum_batch_size;

    model->forward_pass(
        tmp_batch_size,
        this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);

    model->compute_loss(
        tmp_batch_size,
        this->Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);
  }

  model->type_state_propagation = PROPAGATION::TRAINING;

  model->n_acc_trial =
      n_data *
      (this->get_seq_w() - model->n_time_delay) *
      (model->type_accuracy_function == ACCU_FN::CROSS_ENTROPY
           ? 1_UZ
           : model->get_n_out());

  model->merge_mp_accu_loss();

  return model->get_loss(ENV::NONE);
}

double CrossVal::Test_Epoch_Loop(Model *const model) {
  size_t const n_data(this->get_n_data()),
      tmp_maximum_batch_size(model->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_size, tmp_batch_index;

  model->reset_loss();

  model->type_state_propagation = PROPAGATION::INFERENCE;

  for (tmp_batch_index = 0_UZ; tmp_batch_index != tmp_number_batchs;
       ++tmp_batch_index) {
    tmp_batch_size =
        tmp_batch_index + 1_UZ != tmp_number_batchs
            ? tmp_maximum_batch_size
            : n_data - tmp_batch_index * tmp_maximum_batch_size;

    model->forward_pass(
        tmp_batch_size,
        this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);

    model->compute_loss(
        tmp_batch_size,
        this->Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);
  }

  model->type_state_propagation = PROPAGATION::TRAINING;

  model->n_acc_trial =
      n_data *
      (this->get_seq_w() - model->n_time_delay) *
      (model->type_accuracy_function == ACCU_FN::CROSS_ENTROPY
           ? 1_UZ
           : model->get_n_out());

  return model->get_loss(ENV::NONE);
}

CrossVal::~CrossVal(void) {
  this->Deallocate();
}
}