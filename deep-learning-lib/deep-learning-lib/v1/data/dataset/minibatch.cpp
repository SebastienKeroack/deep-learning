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

#include "deep-learning-lib/v1/data/datasets.hpp"
#include "deep-learning-lib/data/string.hpp"
#include "deep-learning-lib/data/time.hpp"
#include "deep-learning-lib/io/logger.hpp"
#include "deep-learning-lib/v1/mem/reallocate.hpp"

#include <iostream>

namespace DL::v1 {
MiniBatch::MiniBatch(void) : DatasetV1() {
  this->p_type_dataset_process = DATASET::MINIBATCH;
}

MiniBatch::MiniBatch(
    bool const use_shuffle_received,
    size_t const desired_number_examples_per_mini_batch_received,
    size_t const number_mini_batch_maximum_received, DatasetV1 &dataset)
    : DatasetV1(dataset) {
  this->Initialize(use_shuffle_received,
                   desired_number_examples_per_mini_batch_received,
                   number_mini_batch_maximum_received);

  this->p_type_dataset_process = DATASET::MINIBATCH;
}

void MiniBatch::Shuffle(void) {
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

void MiniBatch::Set__Use__Shuffle(bool const use_shuffle_received) {
  this->use_shuffle = use_shuffle_received;

  if (use_shuffle_received == false) {
    for (size_t tmp_index(0_UZ); tmp_index != this->p_n_data;
         ++tmp_index) {
      this->ptr_array_stochastic_index[tmp_index] = tmp_index;
    }
  }
}

void MiniBatch::reset(void) {
  this->number_examples = this->number_examples_last_iteration;
}

bool MiniBatch::Initialize(void) {
  this->DatasetV1::Initialize();

  this->p_type_dataset_process = DATASET::MINIBATCH;

  return true;
}

bool MiniBatch::Initialize(
    bool const use_shuffle_received,
    size_t const desired_number_examples_per_mini_batch_received,
    size_t const number_mini_batch_maximum_received) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.",);

    return false;
  } else if (desired_number_examples_per_mini_batch_received == 0_UZ) {
    ERR(
        L"Desired number data per mini-batch equal zero.",);

    return false;
  } else if (desired_number_examples_per_mini_batch_received >
             this->p_n_data) {
    ERR(
        L"Desired number data per mini-batch (%zu) greater than "
        "total number of data (%zu).",
        desired_number_examples_per_mini_batch_received,
        this->p_n_data);

    return false;
  }

  if (this->Set__Desired_Data_Per_Batch(
          desired_number_examples_per_mini_batch_received,
          number_mini_batch_maximum_received) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Set__Desired_Data_Per_Batch(%zu, %zu)\" function.",
        desired_number_examples_per_mini_batch_received,
        number_mini_batch_maximum_received);

    return false;
  }

  this->ptr_array_stochastic_index = new size_t[this->p_n_data];
  if (this->ptr_array_stochastic_index == nullptr) {
    ERR(L"Can not allocate %zu bytes.",
           this->p_n_data * sizeof(size_t));

    return false;
  }

  this->Set__Use__Shuffle(use_shuffle_received);

  if (use_shuffle_received) {
    for (size_t i(0); i != this->p_n_data; ++i) {
      this->ptr_array_stochastic_index[i] = i;
    }
  }

  this->Generator_Random.seed(static_cast<unsigned int>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count()));

  return true;
}

bool MiniBatch::Set__Desired_Data_Per_Batch(
    size_t const desired_number_examples_per_mini_batch_received,
    size_t const number_mini_batch_maximum_received) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.",);

    return false;
  } else if (desired_number_examples_per_mini_batch_received == 0_UZ) {
    ERR(
        L"Desired number data per mini-batch equal zero.",);

    return false;
  } else if (desired_number_examples_per_mini_batch_received >
             this->DatasetV1::get_n_data()) {
    ERR(
        L"Desired number data per mini-batch (%zu) greater than "
        "total number of data (%zu).",
        desired_number_examples_per_mini_batch_received,
        this->DatasetV1::get_n_data());

    return false;
  }

  double const tmp_number_mini_batch__real(
      static_cast<double>(this->DatasetV1::get_n_data()) /
      static_cast<double>(desired_number_examples_per_mini_batch_received));
  size_t tmp_number_mini_batch(
      static_cast<size_t>(tmp_number_mini_batch__real));

  if (number_mini_batch_maximum_received != 0_UZ) {
    tmp_number_mini_batch =
        tmp_number_mini_batch > number_mini_batch_maximum_received
            ? number_mini_batch_maximum_received
            : tmp_number_mini_batch;
  }

  if (tmp_number_mini_batch <= 1_UZ) {
    ERR(L"Invalid number of mini-batch.",);

    return false;
  }

  if (this->number_mini_batch == tmp_number_mini_batch) {
    return true;
  } else {
    this->number_mini_batch = tmp_number_mini_batch;
  }

  this->number_examples_per_iteration =
      desired_number_examples_per_mini_batch_received;
  this->number_examples =
      this->number_examples_per_iteration +
      static_cast<size_t>(
          (tmp_number_mini_batch__real -
           static_cast<double>(this->number_mini_batch)) *
          static_cast<double>(this->number_examples_per_iteration));

  if (this->ptr_array_inputs_array_stochastic == nullptr) {
    if ((this->ptr_array_inputs_array_stochastic =
             new real const *[this->number_examples]) == nullptr) {
      ERR(L"Can not allocate %zu bytes.",
             this->number_examples * sizeof(real const *));

      return false;
    }
  } else {
    this->ptr_array_inputs_array_stochastic =
        Mem::reallocate_ptofpt<real const *, false>(
            this->ptr_array_inputs_array_stochastic, this->number_examples,
            this->number_examples_last_iteration);
    if (this->ptr_array_inputs_array_stochastic == nullptr) {
      ERR(L"Can not allocate %zu bytes.",
             this->number_examples * sizeof(real const *));

      return false;
    }
  }

  if (this->ptr_array_outputs_array_stochastic == nullptr) {
    if ((this->ptr_array_outputs_array_stochastic =
             new real const *[this->number_examples]) == nullptr) {
      ERR(L"Can not allocate %zu bytes.",
             this->number_examples * sizeof(real const *));

      return false;
    }
  } else {
    this->ptr_array_outputs_array_stochastic =
        Mem::reallocate_ptofpt<real const *, false>(
            this->ptr_array_outputs_array_stochastic, this->number_examples,
            this->number_examples_last_iteration);
    if (this->ptr_array_outputs_array_stochastic == nullptr) {
      ERR(L"Can not allocate %zu bytes.",
             this->number_examples * sizeof(real const *));

      return false;
    }
  }

  this->number_examples_last_iteration = this->number_examples;

  return true;
}

bool MiniBatch::Increment_Mini_Batch(
    size_t const mini_batch_iteration_received) {
  size_t const tmp_data_per_mini_batch(
      mini_batch_iteration_received + 1_UZ != this->number_mini_batch
          ? this->number_examples_per_iteration
          : this->number_examples_last_iteration);
  size_t tmp_last_element_start_index, tmp_last_element_end_index,
      tmp_shift_index, tmp_index;

  tmp_last_element_start_index =
      mini_batch_iteration_received * this->number_examples_per_iteration;
  tmp_last_element_end_index =
      tmp_last_element_start_index + tmp_data_per_mini_batch;

  // Index global inputs to local inputs.
  for (tmp_index = 0_UZ, tmp_shift_index = tmp_last_element_start_index;
       tmp_shift_index != tmp_last_element_end_index;
       ++tmp_shift_index, ++tmp_index) {
    this->ptr_array_inputs_array_stochastic[tmp_index] =
        this->Xm
            [this->ptr_array_stochastic_index[tmp_shift_index +
                                              this->p_str_i]];
    this->ptr_array_outputs_array_stochastic[tmp_index] =
        this->Ym
            [this->ptr_array_stochastic_index[tmp_shift_index +
                                              this->p_str_i]];
  }
  // |END| Index global inputs to local inputs. |END|

  this->number_examples = tmp_data_per_mini_batch;

  return true;
}

bool MiniBatch::Get__Use__Shuffle(void) const {
  return (this->use_shuffle);
}

bool MiniBatch::Deallocate(void) {
  SAFE_DELETE_ARRAY(this->ptr_array_inputs_array_stochastic);
  SAFE_DELETE_ARRAY(this->ptr_array_outputs_array_stochastic);

  SAFE_DELETE_ARRAY(this->ptr_array_stochastic_index);

  if (this->DatasetV1::Deallocate() == false) {
    ERR(
        L"An error has been triggered from the \"Deallocate()\" "
        "function.",);

    return false;
  }

  return true;
}

size_t MiniBatch::get_n_data(void) const {
  return (this->number_examples);
}

size_t MiniBatch::get_n_batch(void) const {
  return (this->number_mini_batch);
}

size_t MiniBatch::Get__Number_Examples_Per_Batch(void) const {
  return (this->number_examples_per_iteration);
}

size_t MiniBatch::Get__Number_Examples_Last_Batch(void) const {
  return (this->number_examples_last_iteration);
}

double MiniBatch::train_mp(Model *const model) {
  double tmp_summation_loss(0.0), tmp_summation_accurancy(0.0);

  if (this->use_shuffle) {
    this->Shuffle();
  }

  for (size_t tmp_mini_batch_index(0_UZ);
       tmp_mini_batch_index != this->number_mini_batch;
       ++tmp_mini_batch_index) {
    if (this->Increment_Mini_Batch(tmp_mini_batch_index)) {
      this->Train_Epoch_OpenMP(model);

      tmp_summation_loss += model->get_loss(ENV::NONE);
      tmp_summation_accurancy += this->compute_accuracy(
          this->get_n_data(), this->Get__Input_Array(),
          this->Get__Output_Array(), model);

      model->update_weights_mp(this->get_n_data(),
                                      this->DatasetV1::get_n_data());
    } else {
      ERR(L"'Increment_Mini_Batch' Fail.",);

      return false;
    }
  }

  this->reset();

  model->epoch_time_step += 1_r;

  tmp_summation_loss /= static_cast<double>(this->number_mini_batch);
  tmp_summation_accurancy /= static_cast<double>(this->number_mini_batch);

  model->set_loss(ENV::TRAIN, tmp_summation_loss);
  model->set_accu(ENV::TRAIN, tmp_summation_accurancy);

  return model->get_loss(ENV::TRAIN);
}

double MiniBatch::train_st(Model *const model) {
  double tmp_summation_loss(0.0), tmp_summation_accurancy(0.0);

  if (this->use_shuffle) {
    this->Shuffle();
  }

  for (size_t tmp_mini_batch_index(0_UZ);
       tmp_mini_batch_index != this->number_mini_batch;
       ++tmp_mini_batch_index) {
    if (this->Increment_Mini_Batch(tmp_mini_batch_index)) {
      this->Train_Epoch_Loop(model);

      tmp_summation_loss += model->get_loss(ENV::NONE);
      tmp_summation_accurancy += this->compute_accuracy(
          this->get_n_data(), this->Get__Input_Array(),
          this->Get__Output_Array(), model);

      model->update_weights_st(this->get_n_data(),
                                    this->DatasetV1::get_n_data());
    } else {
      ERR(L"'Increment_Mini_Batch' Fail.",);

      return false;
    }
  }

  this->reset();

  model->epoch_time_step += 1_r;

  tmp_summation_loss /= static_cast<double>(this->number_mini_batch);
  tmp_summation_accurancy /= static_cast<double>(this->number_mini_batch);

  model->set_loss(ENV::TRAIN, tmp_summation_loss);
  model->set_accu(ENV::TRAIN, tmp_summation_accurancy);

  return model->get_loss(ENV::TRAIN);
}

real MiniBatch::get_inp(size_t const index_received,
                              size_t const sub_index_received) const {
  return (this->ptr_array_inputs_array_stochastic[index_received]
                                                 [sub_index_received]);
}

real MiniBatch::get_out(size_t const index_received,
                               size_t const sub_index_received) const {
  return (this->ptr_array_outputs_array_stochastic[index_received]
                                                  [sub_index_received]);
}

real const *const MiniBatch::get_inp(size_t const index_received) const {
  return (this->ptr_array_inputs_array_stochastic[index_received]);
}

real const *const MiniBatch::get_out(size_t const index_received) const {
  return (this->ptr_array_outputs_array_stochastic[index_received]);
}

real const *const *const MiniBatch::Get__Input_Array(void) const {
  return (this->ptr_array_inputs_array_stochastic);
}

real const *const *const MiniBatch::Get__Output_Array(void) const {
  return (this->ptr_array_outputs_array_stochastic);
}

MiniBatch::~MiniBatch(void) {
  this->Deallocate();
}
}