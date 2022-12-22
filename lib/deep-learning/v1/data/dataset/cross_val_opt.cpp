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

#include <iostream>
#include <array>

namespace DL::v1 {
CrossValOpt::CrossValOpt(void) : CrossVal(), HyperOpt() {
  this->p_type_dataset_process = DATASET::CROSS_VAL_OPT;
}

[[deprecated("Not properly implemented.")]] double CrossValOpt::train_mp(
    Model *const model) {
  // if(++this->p_iteration < this->p_number_iterations_per_search) {
  // return(this->CrossVal::train_mp(model));
  // } else { this->p_iteration = 0_UZ; }

  if (this->use_shuffle)
    this->Shuffle();

  /*
  size_t tmp_fold_index,
            tmp_sub_fold_index;

  Model *tmp_ptr_Neural_Network;

  if(this->Update__Size__Population(this->get_n_batch()) == false)
  {
      ERR(L"An error has been triggered from the
  \"Update__Size__Population(%zu)\" function.",
                               this->get_n_batch());

      return false;
  }
  else if(this->update(model) == false)
  {
      ERR(L"An error has been triggered from the
  \"Update(ptr)\" function.",);

      return false;
  }
  else if(this->update_mem_thread_size(this->get_n_data()) == false)
  {
      ERR(L"An error has been triggered from the
  \"update_mem_thread_size(%zu)\" function.",
                               this->get_n_data());

      return false;
  }
  else if(this->update_mem_batch_size(this->get_n_data()) == false)
  {
      ERR(L"An error has been triggered from the
  \"update_mem_batch_size(%zu)\" function.",
                               this->get_n_data());

      return false;
  }
  else if(this->Initialize__Hyper_Parameters(model) ==
  false)
  {
      ERR(L"An error has been triggered from the
  \"Initialize__Hyper_Parameters(ptr)\" function.",);

      return false;
  }
  else if(this->Shuffle__Hyper_Parameter() == false)
  {
      ERR(L"An error has been triggered from the
  \"Shuffle__Hyper_Parameter()\" function.",);

      return false;
  }
  else if(this->Feed__Hyper_Parameter() == false)
  {
      ERR(L"An error has been triggered from the
  \"Feed__Hyper_Parameter()\" function.",);

      return false;
  }

  for(tmp_fold_index = 0_UZ; tmp_fold_index != this->number_k_fold;
  ++tmp_fold_index)
  {
      tmp_ptr_Neural_Network =
  this->p_ptr_array_ptr_Neural_Networks[tmp_fold_index];

      if(this->Increment_Fold(tmp_fold_index))
      {
          for(tmp_sub_fold_index = 0_UZ; tmp_sub_fold_index !=
  this->number_k_sub_fold; ++tmp_sub_fold_index)
          {
              if(this->Increment_Sub_Fold(tmp_sub_fold_index))
              {
                  this->Train_Epoch_OpenMP(tmp_ptr_Neural_Network);

                  tmp_ptr_Neural_Network->update_weights_mp(this->get_n_data(),
  this->DatasetV1::get_n_data());
              }
              else
              {
                  ERR(L"An error has been triggered from
  the \"Increment_Sub_Fold(%zu)\" function.",
                                              tmp_sub_fold_index);

                  return false;
              }
          }

          this->ptr_array_inputs_array_k_sub_fold =
  this->ptr_array_inputs_array_validation;
          this->ptr_array_outputs_array_k_sub_fold =
  this->ptr_array_outputs_array_validation; this->number_examples =
  this->number_examples_validating;

          this->Test_Epoch_OpenMP(tmp_ptr_Neural_Network);
          this->compute_accuracy(this->get_n_data(),
                                               this->Get__Input_Array(),
                                               this->Get__Output_Array(),
                                               tmp_ptr_Neural_Network);
      }
      else
      {
          ERR(L"An error has been triggered from the
  \"Increment_Fold(%zu)\" function.",
                                   tmp_fold_index);

          return false;
      }
  }
  */

  this->CrossVal::reset();

  if (this->Evaluation() == false) {
    ERR(L"An error has been triggered from the "
        L"`Evaluation()` function.");
    return false;
  }

  model->epoch_time_step += 1_r;

  model->set_loss(ENV::TRAIN, model->get_loss(ENV::NONE));
  model->set_accu(ENV::TRAIN, model->get_accu(ENV::NONE));

  return model->get_loss(ENV::TRAIN);
}

[[deprecated("Not properly implemented.")]] double CrossValOpt::train_st(
    Model *const model) {
  // if(++this->p_iteration < this->p_number_iterations_per_search) {
  // return(this->CrossVal::train_st(model));
  // } else { this->p_iteration = 0_UZ; }

  if (this->use_shuffle)
    this->Shuffle();

  /*
  size_t tmp_fold_index,
            tmp_sub_fold_index;

  Model *tmp_ptr_Neural_Network;

  if(this->Update__Size__Population(this->get_n_batch()) == false)
  {
      ERR(L"An error has been triggered from the
  \"Update__Size__Population(%zu)\" function.",
                               this->get_n_batch());

      return false;
  }
  else if(this->update(model) == false)
  {
      ERR(L"An error has been triggered from the
  \"Update(ptr)\" function.",);

      return false;
  }
  else if(this->update_mem_batch_size(this->get_n_data()) == false)
  {
      ERR(L"An error has been triggered from the
  \"update_mem_batch_size(%zu)\" function.",
                               this->get_n_data());

      return false;
  }
  else if(this->Initialize__Hyper_Parameters(model) ==
  false)
  {
      ERR(L"An error has been triggered from the
  \"Initialize__Hyper_Parameters(ptr)\" function.",);

      return false;
  }
  else if(this->Shuffle__Hyper_Parameter() == false)
  {
      ERR(L"An error has been triggered from the
  \"Shuffle__Hyper_Parameter()\" function.",);

      return false;
  }
  else if(this->Feed__Hyper_Parameter() == false)
  {
      ERR(L"An error has been triggered from the
  \"Feed__Hyper_Parameter()\" function.",);

      return false;
  }

  for(tmp_fold_index = 0_UZ; tmp_fold_index != this->number_k_fold;
  ++tmp_fold_index)
  {
      tmp_ptr_Neural_Network =
  this->p_ptr_array_ptr_Neural_Networks[tmp_fold_index];

      if(this->Increment_Fold(tmp_fold_index))
      {
          for(tmp_sub_fold_index = 0_UZ; tmp_sub_fold_index !=
  this->number_k_sub_fold; ++tmp_sub_fold_index)
          {
              if(this->Increment_Sub_Fold(tmp_sub_fold_index))
              {
                  this->Train_Epoch_Loop(tmp_ptr_Neural_Network);

                  if(tmp_ptr_Neural_Network->Is_Online_Training() == false) {
  tmp_ptr_Neural_Network->update_weights_st(this->get_n_data(),
  this->DatasetV1::get_n_data()); }
              }
              else
              {
                  ERR(L"An error has been triggered from
  the \"Increment_Sub_Fold(%zu)\" function.",
                                              tmp_sub_fold_index);

                  return false;
              }
          }

          this->ptr_array_inputs_array_k_sub_fold =
  this->ptr_array_inputs_array_validation;
          this->ptr_array_outputs_array_k_sub_fold =
  this->ptr_array_outputs_array_validation; this->number_examples =
  this->number_examples_validating;

          this->Test_Epoch_Loop(tmp_ptr_Neural_Network);
          this->compute_accuracy(this->get_n_data(),
                                               this->Get__Input_Array(),
                                               this->Get__Output_Array(),
                                               tmp_ptr_Neural_Network);
      }
      else
      {
          ERR(L"An error has been triggered from the
  \"Increment_Fold(%zu)\" function.",
                                   tmp_fold_index);

          return false;
      }
  }
  */

  this->CrossVal::reset();

  if (this->Evaluation() == false) {
    ERR(L"An error has been triggered from the "
        L"`Evaluation()` function.");
    return false;
  }

  model->epoch_time_step += 1_r;

  model->set_loss(ENV::TRAIN, model->get_loss(ENV::NONE));
  model->set_accu(ENV::TRAIN, model->get_accu(ENV::NONE));

  return model->get_loss(ENV::TRAIN);
}

bool CrossValOpt::Deallocate(void) {
  if (this->CrossVal::Deallocate() == false) {
    ERR(L"An error has been triggered from the "
        L"`CrossVal::Deallocate()` function.");
    return false;
  }

  if (this->HyperOpt::Deallocate() == false) {
    ERR(L"An error has been triggered from the "
        L"`HyperOpt::Deallocate()` function.");
    return false;
  }

  return true;
}

CrossValOpt::~CrossValOpt(void) { this->Deallocate(); }
}  // namespace DL
