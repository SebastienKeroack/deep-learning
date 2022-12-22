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

// PCH:
#include "pch.hpp"

// File header:
#include "deep-learning/v1/data/datasets.hpp"

// Deep learning:
#ifdef COMPILE_CUDA
#include "deep-learning/v1/data/datasets.cuh"
#endif

#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/device/system/shutdown_block.hpp"
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/io/term/input.hpp"
#include "deep-learning/io/term/keyboard.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"

// Standard:
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

using namespace DL::Str;
using namespace DL::Term;

namespace DL::v1 {
Datasets::Datasets(void) : DatasetV1(), HyperOpt() {}

Datasets::Datasets(DATASET_FORMAT::TYPE const dset_fmt,
                   std::wstring const &path_name)
    : DatasetV1(dset_fmt, path_name), HyperOpt() {}

void Datasets::evaluate_envs(Model *const model) {
  TIME_POINT time_str, time_end;

  // Train set.
  INFO(L"");
  INFO(L"Evaluate %zu example(s) from the training set.",
       this->get_dataset(ENV::TRAIN)->DatasetV1::get_n_data());

  time_str = std::chrono::high_resolution_clock::now();

#ifdef COMPILE_CUDA
  if (model->Use__CUDA()) {
    this->Get__CUDA()->Type_Testing(ENV::TRAIN, model);
  } else
#endif
  {
    this->Type_Testing(ENV::TRAIN, model);
  }

  time_end = std::chrono::high_resolution_clock::now();

  INFO(L"%.1f example(s) per second.",
       std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_str)
                   .count() == 0ll
           ? 0.0
           : static_cast<double>(
                 this->get_dataset(ENV::TRAIN)->DatasetV1::get_n_data()) /
                 (static_cast<double>(
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                          time_end - time_str)
                          .count()) /
                  1e+9));
  INFO(L"Loss at training: %.9f", model->get_loss(ENV::TRAIN));
  INFO(L"Accuracy at training: %.5f", model->get_accu(ENV::TRAIN));
  // |END| Train set. |END|

  // Valid set.
  if (this->get_storage_type() >=
      ENUM_TYPE_DATASET_MANAGER_STORAGE::
          TYPE_STORAGE_TRAINING_VALIDATION_TESTING) {
    INFO(L"");
    INFO(L"Evaluate %zu example(s) from the validation set.",
         this->get_dataset(ENV::VALID)->DatasetV1::get_n_data());

    time_str = std::chrono::high_resolution_clock::now();

#ifdef COMPILE_CUDA
    if (model->Use__CUDA()) {
      this->Get__CUDA()->Type_Testing(ENV::VALID, model);
    } else
#endif
    {
      this->Type_Testing(ENV::VALID, model);
    }

    time_end = std::chrono::high_resolution_clock::now();

    INFO(L"%.1f example(s) per second.",
         std::chrono::duration_cast<std::chrono::nanoseconds>(time_end -
                                                              time_str)
                     .count() == 0ll
             ? 0.0
             : static_cast<double>(
                   this->get_dataset(ENV::VALID)->DatasetV1::get_n_data()) /
                   (static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            time_end - time_str)
                            .count()) /
                    1e+9));
    INFO(L"Loss at validating: %.9f", model->get_loss(ENV::VALID));
    INFO(L"Accuracy at validating: %.5f", model->get_accu(ENV::VALID));
  } else {
    model->set_loss(ENV::VALID, model->get_loss(ENV::TRAIN));
    model->set_accu(ENV::VALID, model->get_accu(ENV::TRAIN));
  }
  // |END| Valid set. |END|

  // Test set.
  if (this->get_storage_type() >=
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING) {
    INFO(L"");
    INFO(L"Evaluate %zu example(s) from the testing set.",
         this->get_dataset(ENV::TESTG)->DatasetV1::get_n_data());

    time_str = std::chrono::high_resolution_clock::now();

#ifdef COMPILE_CUDA
    if (model->Use__CUDA()) {
      this->Get__CUDA()->Type_Testing(ENV::TESTG, model);
    } else
#endif
    {
      this->Type_Testing(ENV::TESTG, model);
    }

    time_end = std::chrono::high_resolution_clock::now();

    INFO(L"%.1f example(s) per second.",
         std::chrono::duration_cast<std::chrono::nanoseconds>(time_end -
                                                              time_str)
                     .count() == 0ll
             ? 0.0
             : static_cast<double>(
                   this->get_dataset(ENV::TESTG)->DatasetV1::get_n_data()) /
                   (static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            time_end - time_str)
                            .count()) /
                    1e+9));
    INFO(L"Loss at testing: %.9f", model->get_loss(ENV::TESTG));
    INFO(L"Accuracy at testing: %.5f", model->get_accu(ENV::TESTG));
  } else {
    model->set_loss(ENV::TESTG, model->get_loss(ENV::TRAIN));
    model->set_accu(ENV::TESTG, model->get_accu(ENV::TRAIN));
  }
  // |END| Test set. |END|
}

void Datasets::set_eval_env(ENV::TYPE const type_evaluation_received) {
  this->_type_evaluation = type_evaluation_received;
}

void Datasets::set_desired_optimization_time_between_reports(
    double const desired_optimization_time_between_reports_received) {
  this->_desired_optimization_time_between_reports =
      desired_optimization_time_between_reports_received;
}

bool Datasets::user_controls(void) {
  while (true) {
    INFO(L"");
    INFO(L"User controls:");
    INFO(L"[0]: optimize, processing parameters.");
    INFO(L"[1]: Change evaluation type (%ls).",
         ENV_NAME[this->_type_evaluation].c_str());
    INFO(L"[2]: Change metric comparison (%ls).",
         this->_use_metric_loss ? "Loss" : "Accuracy");
    INFO(L"[3]: Maximum example(s) (%zu).", this->_maximum_examples);
    INFO(
        L"[4]: Desired optimization time between reports "
        L"(%f seconds).",
        this->_desired_optimization_time_between_reports);
    INFO(L"[5]: Minimum dataset out loss accepted (%f).",
         this->_minimum_loss_holdout_accepted);
    INFO(
        L"[6]: DatasetV1 in equal or less dataset out accepted "
        "(%ls).",
        this->_dataset_in_equal_less_holdout_accepted ? "true" : "false");
    INFO(L"[7]: Hyperparameter optimization.");
    INFO(L"[8]: Quit.");

    switch (parse_discrete(0, 8, L"Option: ")) {
      case 0:
        if (this->User_Controls__Optimization_Processing_Parameters() ==
            false) {
          ERR(L"An error has been triggered from the "
              L"`User_Controls__Optimization_Processing_Parameters()` "
              L"function.");
          return false;
        }
        break;
      case 1:
        if (this->User_Controls__Type_Evaluation() == false) {
          ERR(L"An error has been triggered from the "
              L"`User_Controls__Type_Evaluation()` function.");
          return false;
        }
        break;
      case 2:
        if (this->User_Controls__Type_Metric() == false) {
          ERR(L"An error has been triggered from the "
              L"`User_Controls__Type_Metric()` function.");
          return false;
        }
        break;
      case 3:
        if (this->User_Controls__Set__Maximum_Data() == false) {
          ERR(L"An error has been triggered from the "
              L"`User_Controls__Set__Maximum_Data()` function.");
          return false;
        }
        break;
      case 4:
        this->set_desired_optimization_time_between_reports(parse_real<double>(
            0.0, L"Desired optimization time between reports (seconds): "));
        break;
      case 5:
        this->_minimum_loss_holdout_accepted =
            parse_real(0_r, L"Minimum dataset out loss accepted: ");
        break;
      case 6:
        this->_dataset_in_equal_less_holdout_accepted =
            accept(L"DatasetV1 in equal or less dataset out accepted?");
        break;
      case 7:
        if (this->HyperOpt::user_controls() == false) {
          ERR(L"An error has been triggered from the "
              L"`HyperOpt::user_controls()` function.");
          return false;
        }
        break;
      case 8:
        return true;
      default:
        ERR(L"An error has been triggered from the "
            L"`parse_discrete(%d, %d)` function.",
            0, 8);
        break;
    }
  }

  return false;
}

bool Datasets::User_Controls__Set__Maximum_Data(void) {
  INFO(L"");
  INFO(L"Maximum example(s).");
  INFO(L"default=%zu || 0.", this->p_n_data);

  if (this->Set__Maximum_Data(parse_discrete(
          1_UZ, this->p_n_data, L"Maximum example(s): ")) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Maximum_Data()` function.");
    return false;
  }

  return true;
}

bool Datasets::User_Controls__Type_Evaluation(void) {
  INFO(L"");
  INFO(L"Type evaluation.");
  for (unsigned int tmp_type_dataset_index(1u);
       tmp_type_dataset_index != ENV::LENGTH; ++tmp_type_dataset_index) {
    INFO(L"[%d]: %ls.", tmp_type_dataset_index,
         ENV_NAME[static_cast<ENV::TYPE>(tmp_type_dataset_index)].c_str());
  }
  INFO(L"default=%ls.", ENV_NAME[ENV::TESTG].c_str());

  this->set_eval_env(static_cast<ENV::TYPE>(
      parse_discrete(1, ENV::LENGTH - 1, L"Type: ")));

  return true;
}

bool Datasets::User_Controls__Type_Metric(void) {
  INFO(L"");
  INFO(L"Type metric.");
  INFO(L"default=Loss.");

  this->_use_metric_loss = accept(L"(Yes=Loss, No=Accuracy): ");

  return true;
}

bool Datasets::User_Controls__Optimization_Processing_Parameters(void) {
  switch (this->get_dataset(ENV::TRAIN)->Get__Type_Dataset_Process()) {
    case DATASET::BATCH:
      if (this->User_Controls__Optimization_Processing_Parameters__Batch() ==
          false) {
        ERR(L"An error has been triggered from the "
            L"`User_Controls__Optimization_Processing_Parameters__Batch()` "
            L"function.");
        return false;
      }
      break;
    case DATASET::MINIBATCH:
      if (this->User_Controls__Optimization_Processing_Parameters__Mini_Batch() ==
          false) {
        ERR(L"An error has been triggered from the "
            L"`User_Controls__Optimization_Processing_Parameters__Mini_Batch()` "
            L"function.");
        return false;
      }
      break;
    case DATASET::CROSS_VAL:
      if (this->User_Controls__Optimization_Processing_Parameters__Cross_Validation() ==
          false) {
        ERR(L"An error has been triggered from the "
            L"`User_Controls__Optimization_Processing_Parameters__Cross_Validation()` "
            L"function.");
        return false;
      }
      break;
    case DATASET::CROSS_VAL_OPT:
      if (this->User_Controls__Optimization_Processing_Parameters__Cross_Validation__Gaussian_Search() ==
          false) {
        ERR(L"An error has been triggered from the "
            L"`User_Controls__Optimization_Processing_Parameters__Cross_Validation__Gaussian_Search()` "
            L"function.");
        return false;
      }
      break;
    default:
      ERR(L"DatasetV1 process type (%d | %ls) is not managed "
          L"in the switch.",
          this->get_dataset(ENV::TRAIN)->Get__Type_Dataset_Process(),
          DATASET_NAME[this->get_dataset(ENV::TRAIN)
                           ->Get__Type_Dataset_Process()]
              .c_str());
      return false;
  }

  return true;
}

bool Datasets::User_Controls__Optimization_Processing_Parameters__Batch(void) {
  while (true) {
    INFO(L"");
    INFO(L"User controls, %ls:", DATASET_NAME[DATASET::BATCH].c_str());
    INFO(L"[0]: Quit.");

    switch (parse_discrete(0, 0, L"Option: ")) {
      case 0:
        return true;
      default:
        ERR(L"An error has been triggered from the "
            L"`parse_discrete(%d, %d)` function.",
            0, 0);
        return false;
    }
  }

  return false;
}

bool Datasets::User_Controls__Optimization_Processing_Parameters__Mini_Batch(
    void) {
  MiniBatch *tmp_ptr_Dataset_Mini_Batch(
      dynamic_cast<MiniBatch *>(this->get_dataset(ENV::TRAIN)));

  while (true) {
    INFO(L"");
    INFO(L"User controls, %ls:", DATASET_NAME[DATASET::MINIBATCH].c_str());
    INFO(L"[0]: Modify number of mini-batch (%zu).",
         tmp_ptr_Dataset_Mini_Batch->get_n_batch());
    INFO(L"[1]: Use shuffle (%ls).",
         tmp_ptr_Dataset_Mini_Batch->Get__Use__Shuffle() ? "true" : "false");
    INFO(L"[2]: Quit.");

    switch (parse_discrete(0, 2, L"Option: ")) {
      case 0:
        INFO(L"Desired-examples per batch:");
        INFO(L"Range[1, %zu].",
             tmp_ptr_Dataset_Mini_Batch->DatasetV1::get_n_data());
        if (tmp_ptr_Dataset_Mini_Batch->Set__Desired_Data_Per_Batch(
                parse_discrete(
                    1_UZ, tmp_ptr_Dataset_Mini_Batch->DatasetV1::get_n_data(),
                    L"Desired-examples per batch: ")) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Desired_Data_Per_Batch()` function.");
          return false;
        }
        break;
      case 1:
        INFO(L"Shuffle:");
        INFO(L"default=Yes.");
        tmp_ptr_Dataset_Mini_Batch->Set__Use__Shuffle(accept(L"Use shuffle: "));
        break;
      case 2:
        return true;
      default:
        ERR(L"An error has been triggered from the "
            L"`parse_discrete(%d, %d)` function.",
            0, 2);
        return false;
    }
  }

  return false;
}

bool Datasets::
    User_Controls__Optimization_Processing_Parameters__Cross_Validation(void) {
  CrossVal *tmp_ptr_Dataset_Cross_Validation(
      dynamic_cast<CrossVal *>(this->get_dataset(ENV::TRAIN)));

  size_t tmp_number_k_folds, tmp_number_k_sub_folds,
      tmp_number_examples_training;

  while (true) {
    INFO(L"");
    INFO(L"User controls, %ls:", DATASET_NAME[DATASET::CROSS_VAL].c_str());
    INFO(L"[0]: Modify number of K-Fold (%zu, %zu).",
         tmp_ptr_Dataset_Cross_Validation->get_n_batch(),
         tmp_ptr_Dataset_Cross_Validation->Get__Number_Sub_Batch());
    INFO(L"[1]: Use shuffle (%ls).",
         tmp_ptr_Dataset_Cross_Validation->Get__Use__Shuffle() ? "true"
                                                               : "false");
    INFO(L"[2]: Quit.");

    switch (parse_discrete(0, 2, L"Option: ")) {
      case 0:
        // K-fold.
        INFO(L"K-fold:");
        INFO(L"Range[2, %zu].",
             tmp_ptr_Dataset_Cross_Validation->DatasetV1::get_n_data());
        tmp_number_k_folds = parse_discrete(
            2_UZ, tmp_ptr_Dataset_Cross_Validation->DatasetV1::get_n_data(),
            L"K-fold: ");
        // |END| K-fold. |END|

        // K-sub-fold.
        tmp_number_examples_training =
            (tmp_number_k_folds - 1_UZ) *
            (tmp_ptr_Dataset_Cross_Validation->DatasetV1::get_n_data() /
             tmp_number_k_folds);

        INFO(L"K-sub-fold:");
        INFO(L"Range[0, %zu].", tmp_number_examples_training);
        INFO(L"default=%zu.", tmp_number_k_folds - 1_UZ);
        tmp_number_k_sub_folds = parse_discrete(
            0_UZ, tmp_number_examples_training, L"K-sub-fold: ");
        // |END| K-sub-fold. |END|

        if (tmp_ptr_Dataset_Cross_Validation->Set__Desired_K_Fold(
                tmp_number_k_folds, tmp_number_k_sub_folds) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Desired_Data_Per_Batch(%zu, %zu)` function.",
              tmp_number_k_folds, tmp_number_k_sub_folds);
          return false;
        }
        break;
      case 1:
        INFO(L"Shuffle:");
        INFO(L"default=Yes.");
        tmp_ptr_Dataset_Cross_Validation->Set__Use__Shuffle(
            accept(L"Use shuffle: "));
        break;
      case 2:
        return true;
      default:
        ERR(L"An error has been triggered from the "
            L"`parse_discrete(%d, %d)` function.",
            0, 2);
        return false;
    }
  }

  return false;
}

bool Datasets::
    User_Controls__Optimization_Processing_Parameters__Cross_Validation__Gaussian_Search(
        void) {
  CrossValOpt *tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization(
      dynamic_cast<CrossValOpt *>(this->get_dataset(ENV::TRAIN)));

  size_t tmp_number_k_folds, tmp_number_k_sub_folds,
      tmp_number_examples_training;

  while (true) {
    INFO(L"");
    INFO(L"User controls, %ls:", DATASET_NAME[DATASET::CROSS_VAL].c_str());
    INFO(L"[0]: Modify number of K-Fold (%zu, %zu).",
         tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
             ->get_n_batch(),
         tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
             ->Get__Number_Sub_Batch());
    INFO(L"[1]: Use shuffle (%ls).",
         tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
                 ->Get__Use__Shuffle()
             ? "true"
             : "false");
    INFO(L"[2]: Hyperparameter optimization.");
    INFO(L"[3]: Quit.");

    switch (parse_discrete(0, 3, L"Option: ")) {
      case 0:
        // K-fold.
        INFO(L"K-fold:");
        INFO(L"Range[2, %zu].",
             tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
                 ->DatasetV1::get_n_data());
        tmp_number_k_folds = parse_discrete(
            2_UZ,
            tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
                ->DatasetV1::get_n_data(),
            L"K-fold: ");
        // |END| K-fold. |END|

        // K-sub-fold.
        tmp_number_examples_training =
            (tmp_number_k_folds - 1_UZ) *
            (tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
                 ->DatasetV1::get_n_data() /
             tmp_number_k_folds);

        INFO(L"K-sub-fold:");
        INFO(L"Range[0, %zu].", tmp_number_examples_training);
        INFO(L"default=%zu.", tmp_number_k_folds - 1_UZ);
        tmp_number_k_sub_folds = parse_discrete(
            0_UZ, tmp_number_examples_training, L"K-sub-fold: ");
        // |END| K-sub-fold. |END|

        if (tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
                ->Set__Desired_K_Fold(tmp_number_k_folds,
                                      tmp_number_k_sub_folds) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Desired_Data_Per_Batch(%zu, %zu)` function.",
              tmp_number_k_folds, tmp_number_k_sub_folds);
          return false;
        }
        break;
      case 1:
        INFO(L"Shuffle:");
        INFO(L"default=Yes.");
        tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
            ->Set__Use__Shuffle(accept(L"Use shuffle: "));
        break;
      case 2:
        if (tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
                ->user_controls() == false) {
          ERR(L"An error has been triggered from the "
              L"`User_Controls__Push_Back()` function.");
          return false;
        }
        break;
      case 3:
        return true;
      default:
        ERR(L"An error has been triggered from the "
            L"`parse_discrete(%d, %d)` function.",
            0, 3);
        return false;
    }
  }

  return false;
}

bool Datasets::User_Controls__Optimization(Model *&trainer, Model *&trained) {
  while (true) {
    INFO(L"");
    INFO(L"User controls, optimization:");
    INFO(L"[0]: Trainer controls.");
    INFO(L"[1]: Trained controls.");
    INFO(L"[2]: Transfer learning.");
    INFO(L"[3]: DatasetV1 controls.");
    INFO(L"[4]: Quit.");

    switch (parse_discrete(0, 4, L"Option: ")) {
      case 0:
        if (trainer->user_controls() == false) {
          ERR(L"An error has been triggered from the "
              L"`user_controls()` function.");
          return false;
        }
        break;
      case 1:
        if (trained->user_controls() == false) {
          ERR(L"An error has been triggered from the "
              L"`user_controls()` function.");
          return false;
        }
        break;
      case 2:
        if (accept(L"Transfer to trained: ")) {
          if (trained->update(*trainer, true) == false) {
            ERR(L"An error has been triggered from the "
                "`update(ptr, true)` function.");
            return false;
          }
        } else if (accept(L"Transfer to trainer: ")) {
          if (trainer->update(*trained, true) == false) {
            ERR(L"An error has been triggered from the "
                L"`update(ptr, true)` function.");
            return false;
          }
        }
        break;
      case 3:
        if (this->user_controls() == false) {
          ERR(L"An error has been triggered from the "
              L"`user_controls()` function.");
          return false;
        }
        break;
      case 4:  // Quit.
        return true;
      default:
        ERR(L"An error has been triggered from the "
            L"`parse_discrete(%d, %d)` function.",
            0, 4);
        return false;
    }
  }

  return true;
}

bool Datasets::Get__Dataset_In_Equal_Less_Holdout_Accepted(void) const {
  return this->_dataset_in_equal_less_holdout_accepted;
}

bool Datasets::Use__Metric_Loss(void) const { return this->_use_metric_loss; }

double Datasets::Get__Minimum_Loss_Holdout_Accepted(void) const {
  return this->_minimum_loss_holdout_accepted;
}

ENV::TYPE Datasets::Get__Type_Dataset_Evaluation(void) const {
  return this->_type_evaluation;
}

void Datasets::optimize(WhileCond const &while_cond, bool const save_trainer,
                        bool const save_trained, double const desired_loss,
                        std::wstring const &path_trainer_params,
                        std::wstring const &path_trainer_spec_params,
                        std::wstring const &path_trained_params,
                        std::wstring const &path_trained_spec_params,
                        Model *&trainer, Model *&trained) {
  bool trained_updated(false), report(false), alive(true);

  unsigned long long epoch_so_far(1ull), epoch_so_far_report(0ull),
      epochs_interval(1ull);

  double train_loss(trainer->get_loss(ENV::TRAIN)),
      valid_loss(trainer->get_loss(ENV::VALID));

  TIME_POINT time_str, time_end, time_str_report, time_end_report;

  Keyboard keyboard;

  // Check if we reach the desired error.
  if (trained->get_loss(this->_type_evaluation) <= desired_loss) return;
  INFO(L"");

  time_str_report = std::chrono::high_resolution_clock::now();

  do {
    report = ++epoch_so_far_report % epochs_interval == 0ull;

    if (save_trained && trained_updated && report) {
      trained_updated = false;

      if (path_trained_params.empty() == false &&
          trained->save_params(path_trained_params) == false)
        ERR(L"An error has been triggered from the "
            L"`save_params(%ls)` function.",
            path_trained_params.c_str());

      if (path_trained_spec_params.empty() == false &&
          trained->save_spec_params(path_trained_spec_params) == false)
        ERR(L"An error has been triggered from the "
            L"`save_spec_params(%ls)` function.",
            path_trained_spec_params.c_str());
    }

    if (Sys::shutdownblock->preparing_for_shutdown()) break;

#ifdef _WIN32
    if (keyboard.trigger_key(0x51)) {
      INFO(L"A signal for stopping the training has been triggered from "
           L"the user input.");
      std::this_thread::sleep_for(std::chrono::seconds(1));

      break;
    }
#elif __linux__
    keyboard.collect_keys_pressed();

    if (keyboard.trigger_key('q')) {
      INFO(L"A signal for stopping the training has been triggered from "
           L"the user input.");

      keyboard.clear_keys_pressed();

      std::this_thread::sleep_for(std::chrono::seconds(1));

      break;
    }
#endif

    if (report) {
#ifdef _WIN32
      if (keyboard.trigger_key(0x4D))
        if (this->User_Controls__Optimization(trainer, trained) == false)
          ERR(L"An error has been triggered from the "
              L"`User_Controls__Optimization(ptr, ptr)` function.");
#elif __linux__
      if (keyboard.trigger_key('m')) {
        keyboard.clear_keys_pressed();

        if (this->User_Controls__Optimization(trainer, trained) == false)
          ERR(L"An error has been triggered from the "
              L"`User_Controls__Optimization(ptr, ptr)` function.");
      }
#endif
    }

    // Training.
    if (report) {
      INFO(L"#=========================================================#");
      INFO(L"Number of epochs between reports: %llu", epochs_interval);

      INFO(L"");
      INFO(L"[TRAINER]: Train on %zu example(s) from the training set.",
           this->get_dataset(ENV::TRAIN)->DatasetV1::get_n_data());

      time_str = std::chrono::high_resolution_clock::now();
    }

    this->Optimize(trainer);

    if (report) {
      time_end = std::chrono::high_resolution_clock::now();

      INFO(L"[TRAINER]: %.1f example(s) per second.",
           std::chrono::duration_cast<std::chrono::nanoseconds>(time_end -
                                                                time_str)
                       .count() == 0ll
               ? 0.0
               : static_cast<double>(
                     this->get_dataset(ENV::TRAIN)->DatasetV1::get_n_data()) /
                     (static_cast<double>(
                          std::chrono::duration_cast<std::chrono::nanoseconds>(
                              time_end - time_str)
                              .count()) /
                      1e9));

      INFO(L"[TRAINER]: Validate on %zu example(s) from the validation set.",
           this->get_dataset(ENV::VALID)->DatasetV1::get_n_data());

      time_str = std::chrono::high_resolution_clock::now();
    }

    this->evaluate(trainer);

    if (report) {
      time_end = std::chrono::high_resolution_clock::now();

      INFO(L"[TRAINER]: %.1f example(s) per second.",
           std::chrono::duration_cast<std::chrono::nanoseconds>(time_end -
                                                                time_str)
                       .count() == 0ll
               ? 0.0
               : static_cast<double>(
                     this->get_dataset(ENV::VALID)->DatasetV1::get_n_data()) /
                     (static_cast<double>(
                          std::chrono::duration_cast<std::chrono::nanoseconds>(
                              time_end - time_str)
                              .count()) /
                      1e9));
    }

    if (save_trainer &&
        (trainer->get_loss(ENV::TRAIN) + trainer->get_loss(ENV::VALID) <
         train_loss + valid_loss)) {
      // TODO: copy of trainer for refreshing. Faster. Need more memory. Skip
      // saving.
      if (path_trainer_params.empty() == false &&
          trainer->save_params(path_trainer_params) == false)
        ERR(L"An error has been triggered from the "
            L"`save_params(%ls)` function.",
            path_trainer_params.c_str());

      if (path_trainer_spec_params.empty() == false &&
          trainer->save_spec_params(path_trainer_spec_params) == false)
        ERR(L"An error has been triggered from the "
            L"`save_spec_params(%ls)` function.",
            path_trainer_spec_params.c_str());
    }
    // |END| Training. |END|

    // Evaluate test set.
    if (this->_type_evaluation == ENV::TESTG)
      this->Optimization__Testing(report, time_str, time_end, trainer);
    // |END| Evaluate test set. |END|

    // Compare.
    if (trained->Compare(this->_use_metric_loss,
                         this->_dataset_in_equal_less_holdout_accepted,
                         this->_type_evaluation,
                         this->_minimum_loss_holdout_accepted, trainer)) {
      trained_updated = true;

      if (this->_type_evaluation != ENV::TESTG)
        this->Optimization__Testing(false, time_str, time_end, trainer);

      if (trained->update(*trainer, true) == false)
        ERR(L"An error has been triggered from the "
            L"`update(ptr, true)` function.");
    }
    // |END| Compare. |END|

    if (report) {
      INFO(L"");
      INFO(L"Epochs %llu end.", epoch_so_far);
      INFO(L"Loss at:");
      INFO(L"[TRAINER]: Train: %.9f", trainer->get_loss(ENV::TRAIN));
      INFO(L"[TRAINED]: Train: %.9f", trained->get_loss(ENV::TRAIN));
      INFO(L"[TRAINER]: Valid: %.9f", trainer->get_loss(ENV::VALID));
      INFO(L"[TRAINED]: Valid: %.9f", trained->get_loss(ENV::VALID));
      INFO(L"[TRAINER]: Testg: %.9f", trainer->get_loss(ENV::TESTG));
      INFO(L"[TRAINED]: Testg: %.9f", trained->get_loss(ENV::TESTG));
      INFO(L"Desired loss:          %.9f", desired_loss);
      INFO(L"");
      INFO(L"Accuracy at:");
      INFO(L"[TRAINER]: Train: %.5f", trainer->get_accu(ENV::TRAIN));
      INFO(L"[TRAINED]: Train: %.5f", trained->get_accu(ENV::TRAIN));
      INFO(L"[TRAINER]: Valid: %.5f", trainer->get_accu(ENV::VALID));
      INFO(L"[TRAINED]: Valid: %.5f", trained->get_accu(ENV::VALID));
      INFO(L"[TRAINER]: Testg: %.5f", trainer->get_accu(ENV::TESTG));
      INFO(L"[TRAINED]: Testg: %.5f", trained->get_accu(ENV::TESTG));
      INFO(L"");
    }

    // Check if we reach the desired error.
    if (trained->get_loss(this->_type_evaluation) <= desired_loss) {
      INFO(L"Desired error reach.");
      INFO(L"");
      break;
    }

#ifdef COMPILE_CUDA
    if (epoch_so_far == 1ull && trainer->Use__CUDA())
      trainer->cumodel->Set__Limit_Device_Runtime_Pending_Launch_Count();
#endif

    if (report) {
      time_end_report = std::chrono::high_resolution_clock::now();

      INFO(L"Total time performance: %ls",
           Time::time_format(
               static_cast<double>(
                   std::chrono::duration_cast<std::chrono::nanoseconds>(
                       time_end_report - time_str_report)
                       .count()) /
               1e+9)
               .c_str());
      INFO(L"");

      switch (while_cond.type) {
        case WHILE_MODE::EXPIRATION:
          // X = ceil( Max(0, Min(D, Exp - Now)) / (Te - Ts) / to_secs / N) )
          epochs_interval = static_cast<unsigned long long>(ceil(
              std::max<double>(
                  0.0, std::min<double>(
                           this->_desired_optimization_time_between_reports,
                           static_cast<double>(
                               std::chrono::system_clock::to_time_t(
                                   while_cond.expiration) -
                               std::chrono::system_clock::to_time_t(
                                   std::chrono::system_clock::now())))) /
              (static_cast<double>(
                   std::chrono::duration_cast<std::chrono::nanoseconds>(
                       time_end_report - time_str_report)
                       .count()) /
               1e+9 / static_cast<double>(epoch_so_far_report))));
          alive = epochs_interval != 0ull;
          break;
        default:
          // X = Max(1, D / (Te - Ts) / to_secs / N)
          epochs_interval = std::max<unsigned long long>(
              1ull,
              static_cast<unsigned long long>(
                  this->_desired_optimization_time_between_reports /
                  (static_cast<double>(
                       std::chrono::duration_cast<std::chrono::nanoseconds>(
                           time_end_report - time_str_report)
                           .count()) /
                   1e+9 / static_cast<double>(epoch_so_far_report))));
          break;
      }

      epoch_so_far_report = 0ull;

      time_str_report = std::chrono::high_resolution_clock::now();
    }

    switch (while_cond.type) {
      case WHILE_MODE::FOREVER:
      case WHILE_MODE::EXPIRATION:
        break;
      case WHILE_MODE::ITERATION:
        alive = epoch_so_far < while_cond.maximum_iterations;
        break;
      default:
        ERR(L"While condition type (%d | %ls) is not managed "
            L"in the switch.",
            while_cond.type, WHILE_MODE_NAME[while_cond.type].c_str());
        alive = false;
        break;
    }

    if (alive) ++epoch_so_far;
  } while (alive);

  if (save_trained && trained_updated) {
    if (path_trained_params.empty() == false &&
        trained->save_params(path_trained_params) == false)
      ERR(L"An error has been triggered from the "
          L"`save_params(%ls)` function.",
          path_trained_params.c_str());

    if (path_trained_spec_params.empty() == false &&
        trained->save_spec_params(path_trained_spec_params) == false)
      ERR(L"An error has been triggered from the "
          L"`save_spec_params(%ls)` function.",
          path_trained_spec_params.c_str());
  }

  INFO(L"");
  INFO(L"Epochs %llu end.", epoch_so_far);
  INFO(L"[TRAINER]: Train: %.9f", trainer->get_loss(ENV::TRAIN));
  INFO(L"[TRAINED]: Train: %.9f", trained->get_loss(ENV::TRAIN));
  INFO(L"[TRAINER]: Valid: %.9f", trainer->get_loss(ENV::VALID));
  INFO(L"[TRAINED]: Valid: %.9f", trained->get_loss(ENV::VALID));
  INFO(L"[TRAINER]: Testg: %.9f", trainer->get_loss(ENV::TESTG));
  INFO(L"[TRAINED]: Testg: %.9f", trained->get_loss(ENV::TESTG));
  INFO(L"Desired loss: %.9f", desired_loss);
  INFO(L"");
  INFO(L"Accuracy at:");
  INFO(L"[TRAINER]: Train: %.5f", trainer->get_accu(ENV::TRAIN));
  INFO(L"[TRAINED]: Train: %.5f", trained->get_accu(ENV::TRAIN));
  INFO(L"[TRAINER]: Valid: %.5f", trainer->get_accu(ENV::VALID));
  INFO(L"[TRAINED]: Valid: %.5f", trained->get_accu(ENV::VALID));
  INFO(L"[TRAINER]: Testg: %.5f", trainer->get_accu(ENV::TESTG));
  INFO(L"[TRAINED]: Testg: %.5f", trained->get_accu(ENV::TESTG));
  INFO(L"");
  INFO(L"#=========================================================#");
}

void Datasets::Optimization__Testing(bool const report, TIME_POINT &time_str,
                                     TIME_POINT &time_end, Model *&model) {
  if (report) {
    INFO(L"[TRAINER]: Evaluate %zu example(s) from the testing set.",
        this->get_dataset(ENV::TESTG)->DatasetV1::get_n_data());
    time_str = std::chrono::high_resolution_clock::now();
  }

#ifdef COMPILE_CUDA
  if (model->Use__CUDA()) {
    this->Get__CUDA()->Type_Testing(ENV::TESTG, model);
  } else
#endif
  {
    this->Type_Testing(ENV::TESTG, model);
  }

  if (report) {
    time_end = std::chrono::high_resolution_clock::now();

    INFO(L"[TRAINER]: %.1f example(s) per second.",
         std::chrono::duration_cast<std::chrono::nanoseconds>(time_end -
                                                              time_str)
                     .count() == 0ll
             ? 0.0
             : static_cast<double>(
                   this->get_dataset(ENV::TESTG)->DatasetV1::get_n_data()) /
                   (static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            time_end - time_str)
                            .count()) /
                    1e9));
  }
}

void Datasets::Deallocate__Storage(void) {
  if (this->_ptr_array_ptr_Dataset != nullptr) {
    if (this->_reference == false) {
      switch (this->_type_storage_data) {
        case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
          SAFE_DELETE(this->_ptr_array_ptr_Dataset[0]);
          break;
        case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
          SAFE_DELETE(this->_ptr_array_ptr_Dataset[0]);
          SAFE_DELETE(this->_ptr_array_ptr_Dataset[1]);
          break;
        case ENUM_TYPE_DATASET_MANAGER_STORAGE::
            TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
          SAFE_DELETE(this->_ptr_array_ptr_Dataset[0]);
          SAFE_DELETE(this->_ptr_array_ptr_Dataset[1]);
          SAFE_DELETE(this->_ptr_array_ptr_Dataset[2]);
          break;
        default:
          ERR(L"DatasetV1 storage type (%d | %ls) is not managed "
              L"in the switch.",
              this->_type_storage_data,
              ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[this->_type_storage_data]
                  .c_str());
          break;
      }
    }

    delete[] (this->_ptr_array_ptr_Dataset);
    this->_ptr_array_ptr_Dataset = nullptr;
  }

  this->_type_storage_data =
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE;
  this->envs_type_evalt.clear();
  this->use_valid_set = false;
  this->use_testg_set = false;
}

void Datasets::Deallocate_CUDA(void) {
#ifdef COMPILE_CUDA
  if (this->_ptr_CUDA_Dataset_Manager != NULL) {
    this->_ptr_CUDA_Dataset_Manager->Deallocate();

    CUDA__Safe_Call(cudaFree(this->_ptr_CUDA_Dataset_Manager));
  }
#else
  ERR(L"`CUDA` functionality was not built. "
      L"Pass `-DCOMPILE_CUDA` to the compiler.");
#endif
}

bool Datasets::Initialize__CUDA(void) {
#ifdef COMPILE_CUDA
  if (this->_ptr_CUDA_Dataset_Manager == NULL) {
    CUDA__Safe_Call(cudaMalloc((void **)&this->_ptr_CUDA_Dataset_Manager,
                               sizeof(cuDatasets)));

    if (this->_ptr_CUDA_Dataset_Manager->Initialize() == false) {
      ERR(L"An error has been triggered from the "
          L"`Initialize()` function.");
      return false;
    }

    this->_ptr_CUDA_Dataset_Manager->copy(this);

    switch (this->_type_storage_data) {
      case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
        if (this->Initialize_Dataset_CUDA(ENV::TRAIN) == false) {
          ERR(L"An error has been triggered from the "
              L"`Initialize_Dataset_CUDA(TRAIN)` function.");
          return false;
        }
        break;
      case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
        if (this->Initialize_Dataset_CUDA(ENV::TRAIN) == false) {
          ERR(L"An error has been triggered from the "
              L"`Initialize_Dataset_CUDA(TRAIN)` function.");
          return false;
        }

        if (this->Initialize_Dataset_CUDA(ENV::TESTG) == false) {
          ERR(L"An error has been triggered from the "
              L"`Initialize_Dataset_CUDA(TESTG)` function.");
          return false;
        }
        break;
      case ENUM_TYPE_DATASET_MANAGER_STORAGE::
          TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
        if (this->Initialize_Dataset_CUDA(ENV::TRAIN) == false) {
          ERR(L"An error has been triggered from the "
              L"`Initialize_Dataset_CUDA(TRAIN)` function.");
          return false;
        }

        if (this->Initialize_Dataset_CUDA(ENV::VALID) == false) {
          ERR(L"An error has been triggered from the "
              L"`Initialize_Dataset_CUDA(VALID)` function.");
          return false;
        }

        if (this->Initialize_Dataset_CUDA(ENV::TESTG) == false) {
          ERR(L"An error has been triggered from the "
              L"`Initialize_Dataset_CUDA(TESTG)` function.");
          return false;
        }
        break;
      default:
        ERR(L"DatasetV1 manager type (%d | %ls) is not managed "
            L"in the switch.",
            this->_type_storage_data,
            ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[this->_type_storage_data]
                .c_str());
        return false;
    }
  }
#else
  ERR(L"`CUDA` functionality was not built. "
      L"Pass `-DCOMPILE_CUDA` to the compiler.");
#endif

  return true;
}

bool Datasets::Set__Maximum_Data(size_t const number_examples_received) {
  if (this->_maximum_examples < number_examples_received ||
      number_examples_received == 0_UZ) {
    this->_maximum_examples = number_examples_received;

    return true;
  } else {
    return false;
  }
}

bool Datasets::Reallocate_Internal_Storage(void) {
  DatasetV1 const *const tmp_ptr_source_TrainingSet(
      this->_ptr_array_ptr_Dataset[0]);
  DatasetV1 *train_set(nullptr), *tmp_ptr_ValidatingSet(nullptr),
      *tmp_ptr_TestingSet(nullptr);

  DATASET::TYPE const tmp_type_dataset_process(
      tmp_ptr_source_TrainingSet->Get__Type_Dataset_Process());

  switch (this->_type_storage_data) {
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
      if ((train_set = this->Allocate__Dataset(tmp_type_dataset_process,
                                               ENV::TRAIN)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            tmp_type_dataset_process, ENV::TRAIN);
        return false;
      }
      break;
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
      if ((train_set = this->Allocate__Dataset(tmp_type_dataset_process,
                                               ENV::TRAIN)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            tmp_type_dataset_process, ENV::TRAIN);
        return false;
      }

      if ((tmp_ptr_ValidatingSet = this->Allocate__Dataset(
               tmp_type_dataset_process, ENV::VALID)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            tmp_type_dataset_process, ENV::VALID);
        return false;
      }
      break;
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::
        TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
      if ((train_set = this->Allocate__Dataset(tmp_type_dataset_process,
                                               ENV::TRAIN)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            tmp_type_dataset_process, ENV::TRAIN);
        return false;
      }

      if ((tmp_ptr_ValidatingSet = this->Allocate__Dataset(
               tmp_type_dataset_process, ENV::VALID)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            tmp_type_dataset_process, ENV::VALID);
        return false;
      }

      if ((tmp_ptr_TestingSet = this->Allocate__Dataset(
               tmp_type_dataset_process, ENV::TESTG)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            tmp_type_dataset_process, ENV::TESTG);
        return false;
      }
      break;
    default:
      ERR(L"DatasetV1 storage type (%d | %ls) is not managed "
          L"in the switch.",
          this->_type_storage_data,
          ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[this->_type_storage_data]
              .c_str());
      return false;
  }

  struct DatasetsParams datasets_params;

  switch (tmp_type_dataset_process) {
    case DATASET::BATCH:
      break;
    case DATASET::MINIBATCH: {
      MiniBatch const *const tmp_ptr_Dataset_Mini_Batch_Stochastic(
          dynamic_cast<MiniBatch const *>(tmp_ptr_source_TrainingSet));

      datasets_params.train_params.value_0 = static_cast<int>(
          tmp_ptr_Dataset_Mini_Batch_Stochastic->Get__Use__Shuffle());
      datasets_params.train_params.value_1 =
          static_cast<int>(tmp_ptr_Dataset_Mini_Batch_Stochastic
                               ->Get__Number_Examples_Per_Batch());
      datasets_params.train_params.value_2 = static_cast<int>(
          tmp_ptr_Dataset_Mini_Batch_Stochastic->get_n_batch());
    } break;
    case DATASET::CROSS_VAL:
    case DATASET::CROSS_VAL_OPT: {
      CrossVal const *const tmp_ptr_Dataset_Cross_Validation(
          dynamic_cast<CrossVal const *>(tmp_ptr_source_TrainingSet));

      datasets_params.train_params.value_0 = static_cast<int>(
          tmp_ptr_Dataset_Cross_Validation->Get__Use__Shuffle());
      datasets_params.train_params.value_1 =
          static_cast<int>(tmp_ptr_Dataset_Cross_Validation->get_n_batch());
      datasets_params.train_params.value_2 = static_cast<int>(
          tmp_ptr_Dataset_Cross_Validation->Get__Number_Sub_Batch());
    } break;
    default:
      ERR(L"DatasetV1 process type (%d | %ls) is not managed "
          L"in the switch.",
          tmp_type_dataset_process,
          DATASET_NAME[tmp_type_dataset_process].c_str());
      return false;
  }

  switch (this->_type_storage_data) {
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
      if (this->Prepare_Storage(train_set) == false) {
        ERR(L"An error has been triggered from the "
            L"`Prepare_Storage()` function.");
        return false;
      }
      break;
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
      if (this->Prepare_Storage(this->_size_dataset_training__percent,
                                this->_size_dataset_testing__percent, train_set,
                                tmp_ptr_ValidatingSet) == false) {
        ERR(L"An error has been triggered from the "
            L"`Prepare_Storage()` function.");
        return false;
      }
      break;
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::
        TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
      if (this->Prepare_Storage(this->_size_dataset_training__percent,
                                this->_size_dataset_validation__percent,
                                this->_size_dataset_testing__percent, train_set,
                                tmp_ptr_ValidatingSet,
                                tmp_ptr_TestingSet) == false) {
        ERR(L"An error has been triggered from the "
            L"`Prepare_Storage()` function.");
        return false;
      }
      break;
    default:
      ERR(L"DatasetV1 storage type (%d | %ls) is not managed "
          L"in the switch.",
          this->_type_storage_data,
          ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[this->_type_storage_data]
              .c_str());
      return false;
  }

  if (this->Initialize_Dataset(ENV::TRAIN, tmp_type_dataset_process,
                               &datasets_params.train_params) == false) {
    ERR(L"An error has been triggered from the "
        L"`Initialize_Dataset(%d, %d, ptr)` function.",
        ENV::TRAIN,
        this->_ptr_array_ptr_Dataset[0]->Get__Type_Dataset_Process());
    return false;
  }

  return true;
}

bool Datasets::push_back(real const *const ptr_array_inputs_received,
                         real const *const ptr_array_outputs_received) {
  size_t tmp_index, tmp_time_step_index, tmp_data_input_index,
      tmp_data_output_index, tmp_data_time_step_index;

  real const **tmp_ptr_matrix_inputs, **tmp_ptr_matrix_outputs;
  real *tmp_ptr_array_inputs, *tmp_ptr_array_outputs;

  if (this->_maximum_examples != 0_UZ &&
      this->p_n_data >= this->_maximum_examples) {
    size_t const tmp_number_examples_minus_one(this->p_n_data - 1_UZ);
    size_t tmp_input_index, tmp_shift_data_input_index,
        tmp_shift_data_output_index, tmp_shift_data_time_step_index;

    // shift index toward zero by one all inputs/outputs.
    for (tmp_index = 0_UZ; tmp_index != tmp_number_examples_minus_one;
         ++tmp_index) {
      tmp_data_input_index = tmp_index * this->p_n_inp * this->p_seq_w;
      tmp_data_output_index = tmp_index * this->p_n_out * this->p_seq_w;

      tmp_shift_data_input_index =
          (tmp_index + 1_UZ) * this->p_n_inp * this->p_seq_w;
      tmp_shift_data_output_index =
          (tmp_index + 1_UZ) * this->p_n_out * this->p_seq_w;

      for (tmp_time_step_index = 0_UZ; tmp_time_step_index != this->p_seq_w;
           ++tmp_time_step_index) {
        tmp_data_time_step_index =
            tmp_data_input_index + tmp_time_step_index * this->p_n_inp;

        tmp_shift_data_time_step_index =
            tmp_shift_data_input_index + tmp_time_step_index * this->p_n_inp;

        for (tmp_input_index = 0_UZ; tmp_input_index != this->p_n_inp;
             ++tmp_input_index) {
          this->X[tmp_data_time_step_index + tmp_input_index] =
              this->X[tmp_shift_data_time_step_index + tmp_input_index];
        }

        tmp_data_time_step_index =
            tmp_data_output_index + tmp_time_step_index * this->p_n_out;

        tmp_shift_data_time_step_index =
            tmp_shift_data_output_index + tmp_time_step_index * this->p_n_out;

        for (tmp_input_index = 0_UZ; tmp_input_index != this->p_n_out;
             ++tmp_input_index) {
          this->Y[tmp_data_time_step_index + tmp_input_index] =
              this->Y[tmp_shift_data_time_step_index + tmp_input_index];
        }
      }
    }
    // |END| shift index toward zero by one all inputs/outputs. |END|

    // Assign new inputs/outputs.
    tmp_data_input_index =
        tmp_number_examples_minus_one * this->p_n_inp * this->p_seq_w;
    tmp_data_output_index =
        tmp_number_examples_minus_one * this->p_n_out * this->p_seq_w;

    for (tmp_time_step_index = 0_UZ; tmp_time_step_index != this->p_seq_w;
         ++tmp_time_step_index) {
      tmp_data_time_step_index =
          tmp_data_input_index + tmp_time_step_index * this->p_n_inp;

      for (tmp_index = 0_UZ; tmp_index != this->p_n_inp; ++tmp_index) {
        this->X[tmp_data_time_step_index + tmp_index] =
            ptr_array_inputs_received[tmp_time_step_index * this->p_n_inp +
                                      tmp_index];
      }

      tmp_data_time_step_index =
          tmp_data_output_index + tmp_time_step_index * this->p_n_out;

      for (tmp_index = 0_UZ; tmp_index != this->p_n_out; ++tmp_index) {
        this->Y[tmp_data_time_step_index + tmp_index] =
            ptr_array_outputs_received[tmp_time_step_index * this->p_n_out +
                                       tmp_index];
      }
    }
    // |END| Assign new inputs/outputs. |END|
  } else {
    // TODO: reallocate preprocessing scaler...
    // TODO: allocate by chunk of memory. Keep tracking of the size.
    size_t const tmp_number_examples_plus_one(this->p_n_data + 1_UZ);

    // reallocate.
    //  Inputs.
    tmp_ptr_array_inputs = Mem::reallocate<real, true>(
        this->X, tmp_number_examples_plus_one * this->p_n_inp * this->p_seq_w,
        this->p_n_data * this->p_n_inp * this->p_seq_w);
    this->X = tmp_ptr_array_inputs;

    tmp_ptr_matrix_inputs = Mem::reallocate_ptofpt<real const *, false>(
        this->Xm, tmp_number_examples_plus_one, this->p_n_data);
    this->Xm = tmp_ptr_matrix_inputs;
    //  |END| Inputs. |END|

    //  Outputs.
    tmp_ptr_array_outputs = Mem::reallocate<real, true>(
        this->Y, tmp_number_examples_plus_one * this->p_n_out * this->p_seq_w,
        this->p_n_data * this->p_n_out * this->p_seq_w);
    this->Y = tmp_ptr_array_outputs;

    tmp_ptr_matrix_outputs = Mem::reallocate_ptofpt<real const *, false>(
        this->Ym, tmp_number_examples_plus_one, this->p_n_data);
    this->Ym = tmp_ptr_matrix_outputs;
    //  |END| Outputs. |END|
    // |END| reallocate. |END|

    // Assign new position.
    for (tmp_index = 0_UZ; tmp_index != tmp_number_examples_plus_one;
         ++tmp_index) {
      tmp_ptr_matrix_inputs[tmp_index] =
          tmp_ptr_array_inputs + tmp_index * this->p_n_inp * this->p_seq_w;

      tmp_ptr_matrix_outputs[tmp_index] =
          tmp_ptr_array_outputs + tmp_index * this->p_n_out * this->p_seq_w;
    }
    // |END| Assign new position. |END|

    // Assign new inputs/outputs.
    tmp_data_input_index = this->p_n_data * this->p_n_inp * this->p_seq_w;
    tmp_data_output_index = this->p_n_data * this->p_n_out * this->p_seq_w;

    for (tmp_time_step_index = 0_UZ; tmp_time_step_index != this->p_seq_w;
         ++tmp_time_step_index) {
      tmp_data_time_step_index =
          tmp_data_input_index + tmp_time_step_index * this->p_n_inp;

      for (tmp_index = 0_UZ; tmp_index != this->p_n_inp; ++tmp_index) {
        tmp_ptr_array_inputs[tmp_data_time_step_index + tmp_index] =
            ptr_array_inputs_received[tmp_time_step_index * this->p_n_inp +
                                      tmp_index];
      }

      tmp_data_time_step_index =
          tmp_data_output_index + tmp_time_step_index * this->p_n_out;

      for (tmp_index = 0_UZ; tmp_index != this->p_n_out; ++tmp_index) {
        tmp_ptr_array_outputs[tmp_data_time_step_index + tmp_index] =
            ptr_array_outputs_received[tmp_time_step_index * this->p_n_out +
                                       tmp_index];
      }
    }
    // |END| Assign new inputs/outputs. |END|

    ++this->p_n_data;

    this->Reallocate_Internal_Storage();
  }

  return true;
}

bool Datasets::Prepare_Storage(DatasetV1 *const ptr_TrainingSet_received) {
  if (this->get_n_data() == 0_UZ) {
    ERR(L"The number example(s) can not be equal to zero.");
    return false;
  } else if (ptr_TrainingSet_received == nullptr) {
    ERR(L"`ptr_TrainingSet_received` is a nullptr.");
    return false;
  }

  if (this->_type_storage_data !=
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE)
    this->Deallocate__Storage();

  this->_ptr_array_ptr_Dataset = new DatasetV1 *[1];
  this->_ptr_array_ptr_Dataset[0] = ptr_TrainingSet_received;
  ptr_TrainingSet_received->reference(this->p_n_data - this->p_str_i,
                                      this->Xm + this->p_str_i,
                                      this->Ym + this->p_str_i, *this);

  this->_size_dataset_training__percent = 100.0;
  this->_size_dataset_validation__percent = 0.0;
  this->_size_dataset_testing__percent = 0.0;

  this->_type_storage_data =
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING;
  this->envs_type_evalt.push_back(ENV::TRAIN);
  this->use_valid_set = false;
  this->use_testg_set = false;

  return true;
}

bool Datasets::Prepare_Storage(size_t const number_examples_training_received,
                               size_t const number_examples_testing_received,
                               DatasetV1 *const ptr_TrainingSet_received,
                               DatasetV1 *const ptr_TestingSet_received) {
  if (number_examples_training_received + number_examples_testing_received !=
      this->get_n_data()) {
    ERR(L"training(%zu) + testing(%zu) != total(%zu).",
        number_examples_training_received, number_examples_testing_received,
        this->get_n_data());
    return false;
  } else if (number_examples_training_received == 0_UZ) {
    ERR(L"The number of example(s) from the training set "
        L"can not be equal to zero.");
    return false;
  } else if (number_examples_testing_received == 0_UZ) {
    ERR(L"The number of example(s) from the testing set "
        L"can not be equal to zero.");
    return false;
  } else if (this->get_n_data() < 2_UZ) {
    ERR(L"The number of example(s) (%zu) is less than 2.", this->get_n_data());
    return false;
  } else if (ptr_TrainingSet_received == nullptr) {
    ERR(L"`ptr_TrainingSet_received` is a nullptr.");
    return false;
  } else if (ptr_TestingSet_received == nullptr) {
    ERR(L"`ptr_TestingSet_received` is a nullptr.");
    return false;
  }

  if (this->_type_storage_data !=
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE)
    this->Deallocate__Storage();

  real const **tmp_ptr_array_inputs_array(this->Xm + this->p_str_i),
      **tmp_ptr_array_outputs_array(this->Ym + this->p_str_i);

  this->_ptr_array_ptr_Dataset = new DatasetV1 *[2];

  this->_ptr_array_ptr_Dataset[0] = ptr_TrainingSet_received;
  ptr_TrainingSet_received->reference(number_examples_training_received,
                                      tmp_ptr_array_inputs_array,
                                      tmp_ptr_array_outputs_array, *this);

  tmp_ptr_array_inputs_array += number_examples_training_received;
  tmp_ptr_array_outputs_array += number_examples_training_received;

  this->_ptr_array_ptr_Dataset[1] = ptr_TestingSet_received;
  ptr_TestingSet_received->reference(number_examples_testing_received,
                                     tmp_ptr_array_inputs_array,
                                     tmp_ptr_array_outputs_array, *this);

  this->_size_dataset_training__percent =
      100.0 * static_cast<double>(number_examples_training_received) /
      static_cast<double>(this->get_n_data());
  this->_size_dataset_validation__percent = 0.0;
  this->_size_dataset_testing__percent =
      100.0 * static_cast<double>(number_examples_testing_received) /
      static_cast<double>(this->get_n_data());

  this->_type_storage_data =
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING;
  this->envs_type_evalt.push_back(ENV::TRAIN);
  this->envs_type_evalt.push_back(ENV::VALID);
  this->use_valid_set = true;
  this->use_testg_set = false;

  return true;
}

bool Datasets::Prepare_Storage(size_t const number_examples_training_received,
                               size_t const number_examples_validation_received,
                               size_t const number_examples_testing_received,
                               DatasetV1 *const ptr_TrainingSet_received,
                               DatasetV1 *const ptr_ValidatingSet_received,
                               DatasetV1 *const ptr_TestingSet_received) {
  if (number_examples_training_received + number_examples_validation_received +
          number_examples_testing_received !=
      this->get_n_data()) {
    ERR(L"training(%zu) + validating(%zu) + testing(%zu) "
        L"!= total(%zu).",
        number_examples_training_received, number_examples_validation_received,
        number_examples_testing_received, this->get_n_data());
    return false;
  } else if (number_examples_training_received == 0_UZ) {
    ERR(L"The number of example(s) from the training set "
        L"can not be equal to zero.");
    return false;
  } else if (number_examples_validation_received == 0_UZ) {
    ERR(L"The number of example(s) from the validation set "
        L"can not be equal to zero.");
    return false;
  } else if (number_examples_testing_received == 0_UZ) {
    ERR(L"The number of example(s) from the testing set "
        L"can not be equal to zero.");
    return false;
  } else if (this->get_n_data() < 3_UZ) {
    ERR(L"The number of example(s) (%zu) is less than 3.", this->get_n_data());
    return false;
  } else if (ptr_TrainingSet_received == nullptr) {
    ERR(L"`ptr_TrainingSet_received` is a nullptr.");
    return false;
  } else if (ptr_ValidatingSet_received == nullptr) {
    ERR(L"`ptr_ValidatingSet_received` is a nullptr.");
    return false;
  } else if (ptr_TestingSet_received == nullptr) {
    ERR(L"`ptr_TestingSet_received` is a nullptr.");
    return false;
  }

  if (this->_type_storage_data !=
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE)
    this->Deallocate__Storage();

  real const **tmp_ptr_array_inputs_array(this->Xm + this->p_str_i),
      **tmp_ptr_array_outputs_array(this->Ym + this->p_str_i);

  this->_ptr_array_ptr_Dataset = new DatasetV1 *[3];

  this->_ptr_array_ptr_Dataset[0] = ptr_TrainingSet_received;
  ptr_TrainingSet_received->reference(number_examples_training_received,
                                      tmp_ptr_array_inputs_array,
                                      tmp_ptr_array_outputs_array, *this);

  tmp_ptr_array_inputs_array += number_examples_training_received;
  tmp_ptr_array_outputs_array += number_examples_training_received;

  this->_ptr_array_ptr_Dataset[1] = ptr_ValidatingSet_received;
  ptr_ValidatingSet_received->reference(number_examples_validation_received,
                                        tmp_ptr_array_inputs_array,
                                        tmp_ptr_array_outputs_array, *this);

  tmp_ptr_array_inputs_array += number_examples_validation_received;
  tmp_ptr_array_outputs_array += number_examples_validation_received;

  this->_ptr_array_ptr_Dataset[2] = ptr_TestingSet_received;
  ptr_TestingSet_received->reference(number_examples_testing_received,
                                     tmp_ptr_array_inputs_array,
                                     tmp_ptr_array_outputs_array, *this);

  this->_size_dataset_training__percent =
      100.0 * static_cast<double>(number_examples_training_received) /
      static_cast<double>(this->get_n_data());
  this->_size_dataset_validation__percent =
      100.0 * static_cast<double>(number_examples_validation_received) /
      static_cast<double>(this->get_n_data());
  this->_size_dataset_testing__percent =
      100.0 * static_cast<double>(number_examples_testing_received) /
      static_cast<double>(this->get_n_data());

  this->_type_storage_data = ENUM_TYPE_DATASET_MANAGER_STORAGE::
      TYPE_STORAGE_TRAINING_VALIDATION_TESTING;
  this->envs_type_evalt.push_back(ENV::TRAIN);
  this->envs_type_evalt.push_back(ENV::VALID);
  this->envs_type_evalt.push_back(ENV::TESTG);
  this->use_valid_set = true;
  this->use_testg_set = true;

  return true;
}

bool Datasets::Prepare_Storage(
    double const number_examples_percent_training_received,
    double const number_examples_percent_testing_received,
    DatasetV1 *const ptr_TrainingSet_received,
    DatasetV1 *const ptr_TestingSet_received) {
  if (number_examples_percent_training_received +
          number_examples_percent_testing_received !=
      100.0) {
    ERR(L"training(%f%%) + testing(%f%%) != 100.0%%.",
        number_examples_percent_training_received,
        number_examples_percent_testing_received);
    return false;
  } else if (number_examples_percent_training_received == 0.0) {
    ERR(L"The number of example(s) from the training set "
        L"can not be equal to zero.");
    return false;
  } else if (number_examples_percent_testing_received == 0.0) {
    ERR(L"The number of example(s) from the testing set "
        L"can not be equal to zero.");
    return false;
  } else if (this->get_n_data() < 2_UZ) {
    ERR(L"The number of example(s) (%zu) is less than 2.", this->get_n_data());
    return false;
  } else if (ptr_TrainingSet_received == nullptr) {
    ERR(L"`ptr_TrainingSet_received` is a nullptr.");
    return false;
  } else if (ptr_TestingSet_received == nullptr) {
    ERR(L"`ptr_TestingSet_received` is a nullptr.");
    return false;
  }

  if (this->_type_storage_data !=
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE)
    this->Deallocate__Storage();

  size_t const tmp_number_examples_training(std::max<size_t>(
      static_cast<size_t>(round(static_cast<double>(this->get_n_data()) *
                                number_examples_percent_training_received /
                                100.0)),
      1_UZ)),
      tmp_number_examples_testing(this->get_n_data() -
                                  tmp_number_examples_training);

  real const **tmp_ptr_array_inputs_array(this->Xm + this->p_str_i),
      **tmp_ptr_array_outputs_array(this->Ym + this->p_str_i);

  this->_ptr_array_ptr_Dataset = new DatasetV1 *[2];

  this->_ptr_array_ptr_Dataset[0] = ptr_TrainingSet_received;
  ptr_TrainingSet_received->reference(tmp_number_examples_training,
                                      tmp_ptr_array_inputs_array,
                                      tmp_ptr_array_outputs_array, *this);

  tmp_ptr_array_inputs_array += tmp_number_examples_training;
  tmp_ptr_array_outputs_array += tmp_number_examples_training;

  this->_ptr_array_ptr_Dataset[1] = ptr_TestingSet_received;
  ptr_TestingSet_received->reference(tmp_number_examples_testing,
                                     tmp_ptr_array_inputs_array,
                                     tmp_ptr_array_outputs_array, *this);

  this->_size_dataset_training__percent =
      number_examples_percent_training_received;
  this->_size_dataset_validation__percent = 0.0;
  this->_size_dataset_testing__percent =
      number_examples_percent_testing_received;

  this->_type_storage_data =
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING;
  this->envs_type_evalt.push_back(ENV::TRAIN);
  this->envs_type_evalt.push_back(ENV::VALID);
  this->use_valid_set = true;
  this->use_testg_set = false;

  return true;
}

bool Datasets::Prepare_Storage(
    double const number_examples_percent_training_received,
    double const number_examples_percent_validation_received,
    double const number_examples_percent_testing_received,
    DatasetV1 *const ptr_TrainingSet_received,
    DatasetV1 *const ptr_ValidatingSet_received,
    DatasetV1 *const ptr_TestingSet_received) {
  if (number_examples_percent_training_received +
          number_examples_percent_validation_received +
          number_examples_percent_testing_received !=
      100.0) {
    ERR(L"training(%f%%) + validation(%f%%) + testing(%f%%) "
        "!= 100.0%%.",
        number_examples_percent_training_received,
        number_examples_percent_validation_received,
        number_examples_percent_testing_received);
    return false;
  } else if (number_examples_percent_training_received == 0.0) {
    ERR(L"The number of example(s) from the training set "
        L"can not be equal to zero.");
    return false;
  } else if (number_examples_percent_validation_received == 0.0) {
    ERR(L"The number of example(s) from the validation set "
        L"can not be equal to zero.");
    return false;
  } else if (number_examples_percent_testing_received == 0.0) {
    ERR(L"The number of example(s) from the testing set "
        L"can not be equal to zero.");
    return false;
  } else if (this->get_n_data() < 3_UZ) {
    ERR(L"The number of example(s) (%zu) is less than 3.", this->get_n_data());
    return false;
  } else if (ptr_TrainingSet_received == nullptr) {
    ERR(L"`ptr_TrainingSet_received` is a nullptr.");
    return false;
  } else if (ptr_ValidatingSet_received == nullptr) {
    ERR(L"`ptr_ValidatingSet_received` is a nullptr.");
    return false;
  } else if (ptr_TestingSet_received == nullptr) {
    ERR(L"`ptr_TestingSet_received` is a nullptr.");
    return false;
  }

  if (this->_type_storage_data !=
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE)
    this->Deallocate__Storage();

  size_t const tmp_number_examples_training(Math::clip<size_t>(
      static_cast<size_t>(round(static_cast<double>(this->get_n_data()) *
                                number_examples_percent_training_received /
                                100.0)),
      1_UZ, this->get_n_data() - 2_UZ)),
      tmp_number_examples_validation(Math::clip<size_t>(
          static_cast<size_t>(
              round(static_cast<double>(this->get_n_data()) *
                    number_examples_percent_validation_received / 100.0)),
          1_UZ, this->get_n_data() - tmp_number_examples_training - 1_UZ)),
      tmp_number_examples_testing(
          std::max<size_t>(this->get_n_data() - tmp_number_examples_training -
                               tmp_number_examples_validation,
                           1_UZ));

  real const **tmp_ptr_array_inputs_array(this->Xm + this->p_str_i),
      **tmp_ptr_array_outputs_array(this->Ym + this->p_str_i);

  this->_ptr_array_ptr_Dataset = new DatasetV1 *[3];

  this->_ptr_array_ptr_Dataset[0] = ptr_TrainingSet_received;
  ptr_TrainingSet_received->reference(tmp_number_examples_training,
                                      tmp_ptr_array_inputs_array,
                                      tmp_ptr_array_outputs_array, *this);

  tmp_ptr_array_inputs_array += tmp_number_examples_training;
  tmp_ptr_array_outputs_array += tmp_number_examples_training;

  this->_ptr_array_ptr_Dataset[1] = ptr_ValidatingSet_received;
  ptr_ValidatingSet_received->reference(tmp_number_examples_validation,
                                        tmp_ptr_array_inputs_array,
                                        tmp_ptr_array_outputs_array, *this);

  tmp_ptr_array_inputs_array += tmp_number_examples_validation;
  tmp_ptr_array_outputs_array += tmp_number_examples_validation;

  this->_ptr_array_ptr_Dataset[2] = ptr_TestingSet_received;
  ptr_TestingSet_received->reference(tmp_number_examples_testing,
                                     tmp_ptr_array_inputs_array,
                                     tmp_ptr_array_outputs_array, *this);

  this->_size_dataset_training__percent =
      number_examples_percent_training_received;
  this->_size_dataset_validation__percent =
      number_examples_percent_validation_received;
  this->_size_dataset_testing__percent =
      number_examples_percent_testing_received;

  this->_type_storage_data = ENUM_TYPE_DATASET_MANAGER_STORAGE::
      TYPE_STORAGE_TRAINING_VALIDATION_TESTING;
  this->envs_type_evalt.push_back(ENV::TRAIN);
  this->envs_type_evalt.push_back(ENV::VALID);
  this->envs_type_evalt.push_back(ENV::TESTG);
  this->use_valid_set = true;
  this->use_testg_set = true;

  return true;
}

bool Datasets::Initialize_Dataset_CUDA(ENV::TYPE const env_type) {
#ifdef COMPILE_CUDA
  DatasetV1 const *const dataset(this->get_dataset(env_type));

  if (dataset == nullptr) {
    ERR(L"An error has been triggered from the "
        L"`get_dataset(%d)` function.",
        env_type);
    return false;
  }

  if (this->_ptr_CUDA_Dataset_Manager->Set__Type_Gradient_Descent(
          env_type, dataset->Get__Type_Dataset_Process()) == false) {
    ERR(L"An error has been triggered from the "
        L"`CUDA->Set__Type_Gradient_Descent(%d, %d)` function.",
        env_type, dataset->Get__Type_Dataset_Process());
    return false;
  }

  switch (env_type) {
    case ENV::TRAIN:
      switch (dataset->Get__Type_Dataset_Process()) {
        case DATASET::BATCH:
          break;
        case DATASET::MINIBATCH: {
          MiniBatch const *const tmp_ptr_Dataset_Mini_Batch(
              dynamic_cast<MiniBatch const *const>(dataset));

          if (this->_ptr_CUDA_Dataset_Manager
                  ->Initialize_Mini_Batch_Stochastic_Gradient_Descent(
                      tmp_ptr_Dataset_Mini_Batch->Get__Use__Shuffle(),
                      tmp_ptr_Dataset_Mini_Batch
                          ->Get__Number_Examples_Per_Batch(),
                      tmp_ptr_Dataset_Mini_Batch->get_n_batch()) == false) {
            ERR(L"An error has been triggered from the "
                L"`Initialize_Mini_Batch_Stochastic_Gradient_Descent"
                L"(%ls, %zu, %zu)` function.",
                tmp_ptr_Dataset_Mini_Batch->Get__Use__Shuffle() ? L"true"
                                                                : L"false",
                tmp_ptr_Dataset_Mini_Batch->Get__Number_Examples_Per_Batch(),
                tmp_ptr_Dataset_Mini_Batch->get_n_batch());

            return false;
          }
        } break;
        case DATASET::CROSS_VAL: {
          CrossVal const *const tmp_ptr_Dataset_Cross_Validation(
              dynamic_cast<CrossVal const *const>(dataset));

          if (this->_ptr_CUDA_Dataset_Manager->Initialize__Cross_Validation(
                  tmp_ptr_Dataset_Cross_Validation->Get__Use__Shuffle(),
                  tmp_ptr_Dataset_Cross_Validation->get_n_batch(),
                  tmp_ptr_Dataset_Cross_Validation->Get__Number_Sub_Batch()) ==
              false) {
            ERR(L"An error has been triggered from the "
                L"`Initialize__Cross_Validation(%ls, %zu, %zu)` function.",
                tmp_ptr_Dataset_Cross_Validation->Get__Use__Shuffle() ? L"true"
                                                                      : L"false",
                tmp_ptr_Dataset_Cross_Validation->get_n_batch(),
                tmp_ptr_Dataset_Cross_Validation->Get__Number_Sub_Batch());

            return false;
          }
        } break;
        default:
          ERR(L"DatasetV1 process type (%d | %ls) is not managed "
              L"in the switch.",
              dataset->Get__Type_Dataset_Process(),
              DATASET_NAME[dataset->Get__Type_Dataset_Process()].c_str());
          return nullptr;
      }
      break;
    case ENV::VALID:
    case ENV::TESTG:
      break;
    default:
      ERR(L"DatasetV1 type (%d | %ls) is not managed in the switch.", env_type,
          ENV_NAME[env_type].c_str());
      return false;
  }
#else
  ERR(L"`CUDA` functionality was not built. "
      L"Pass `-DCOMPILE_CUDA` to the compiler.");
#endif

  return true;
}

DatasetV1 *Datasets::Allocate__Dataset(
    DATASET::TYPE const type_dataset_process_received,
    ENV::TYPE const type_data_received) {
  DatasetV1 *dataset(nullptr);

  switch (type_data_received) {
    case ENV::TRAIN:
      switch (type_dataset_process_received) {
        case DATASET::BATCH:  // Batch.
          dataset = new DatasetV1;
          break;
        case DATASET::MINIBATCH:  // Mini-batch stochastic.
          dataset = new MiniBatch;
          break;
        case DATASET::CROSS_VAL:  // Cross-validation
          dataset = new CrossVal;
          break;
        case DATASET::CROSS_VAL_OPT:  // Cross-validation,
                                      // random
                                      // search
          dataset = new CrossValOpt;
          break;
        default:
          ERR(L"DatasetV1 process type (%d | %ls) is not managed in "
              "the switch.",
              type_dataset_process_received,
              DATASET_NAME[type_dataset_process_received].c_str());
          return nullptr;
      }
      break;
    case ENV::VALID:
    case ENV::TESTG:
      dataset = new DatasetV1;
      break;
    default:
      ERR(L"DatasetV1 type (%d | %ls) is not managed in the switch.",
          type_data_received, ENV_NAME[type_data_received].c_str());
      return nullptr;
  }

  return (dataset);
}

bool Datasets::Initialize_Dataset(
    ENV::TYPE const env_type, DATASET::TYPE const type_dataset_process_received,
    struct DatasetParams const *const ptr_Dataset_Parameters_received) {
  DatasetV1 *const dataset(this->get_dataset(env_type));

  if (dataset == nullptr) {
    ERR(L"An error has been triggered from the `get_dataset(%d)` function.",
        env_type);
    return false;
  }

  switch (env_type) {
    case ENV::TRAIN:
      switch (type_dataset_process_received) {
        case DATASET::BATCH:
          break;
        case DATASET::MINIBATCH: {
          MiniBatch *const tmp_ptr_Dataset_Mini_Batch_Stochastic(
              dynamic_cast<MiniBatch *>(dataset));

          bool tmp_use_shuffle;

          size_t const n_data(dataset->DatasetV1::get_n_data());
          size_t tmp_number_desired_data_per_batch, tmp_number_maximum_batch;

          if (ptr_Dataset_Parameters_received == nullptr ||
              (ptr_Dataset_Parameters_received != nullptr &&
               ptr_Dataset_Parameters_received->value_0 == -1)) {
            // Shuffle.
            INFO(L"");
            INFO(L"Shuffle:");
            INFO(L"default=Yes.");
            tmp_use_shuffle = accept(L"Use shuffle: ");
            // |END| Shuffle. |END|
          } else {
            tmp_use_shuffle = ptr_Dataset_Parameters_received->value_0 != 0;
          }

          if (ptr_Dataset_Parameters_received == nullptr ||
              (ptr_Dataset_Parameters_received != nullptr &&
               ptr_Dataset_Parameters_received->value_1 == -1)) {
            // Desired-examples per batch.
            INFO(L"");
            INFO(L"Desired-examples per batch:");
            INFO(L"Range[1, %zu].", n_data);
            tmp_number_desired_data_per_batch = parse_discrete(
                1_UZ, n_data, L"Desired-examples per batch: ");
            // |END| Desired-examples per batch. |END|
          } else {
            tmp_number_desired_data_per_batch =
                static_cast<size_t>(ptr_Dataset_Parameters_received->value_1);
          }

          if (ptr_Dataset_Parameters_received == nullptr ||
              (ptr_Dataset_Parameters_received != nullptr &&
               ptr_Dataset_Parameters_received->value_2 == -1)) {
            // Maximum sub-sample.
            INFO(L"");
            INFO(L"Maximum sub-sample:");
            INFO(L"Range[0, %zu]. Off = 0.", n_data);
            tmp_number_maximum_batch =
                parse_discrete(0_UZ, n_data, L"Maximum sub-sample: ");
            // |END| Maximum sub-sample. |END|
          } else {
            tmp_number_maximum_batch =
                static_cast<size_t>(ptr_Dataset_Parameters_received->value_2);
          }

          if (tmp_ptr_Dataset_Mini_Batch_Stochastic->Initialize(
                  tmp_use_shuffle, tmp_number_desired_data_per_batch,
                  tmp_number_maximum_batch) == false) {
            ERR(L"An error has been triggered from the "
                L"`Initialize(%ls, %zu, %zu)` function.",
                to_wstring(tmp_use_shuffle).c_str(),
                tmp_number_desired_data_per_batch, tmp_number_maximum_batch);

            return false;
          }

          INFO(L"");
          INFO(L"The number of mini-batch is set to %zu.",
               tmp_ptr_Dataset_Mini_Batch_Stochastic->get_n_batch());
        } break;
        case DATASET::CROSS_VAL:
        case DATASET::CROSS_VAL_OPT: {
          CrossVal *const tmp_ptr_Dataset_Cross_Validation(
              dynamic_cast<CrossVal *>(dataset));

          bool tmp_use_shuffle;

          size_t const n_data(
              tmp_ptr_Dataset_Cross_Validation->DatasetV1::get_n_data());
          size_t tmp_number_k_folds, tmp_number_k_sub_folds;

          if (n_data < 2_UZ) {
            ERR(L"The number of example(s) (%zu) is less than 3.", n_data);

            return false;
          }

          if (ptr_Dataset_Parameters_received == nullptr ||
              (ptr_Dataset_Parameters_received != nullptr &&
               ptr_Dataset_Parameters_received->value_0 == -1)) {
            // Shuffle.
            INFO(L"");
            INFO(L"Shuffle:");
            INFO(L"default=Yes.");
            tmp_use_shuffle = accept(L"Use shuffle: ");
            // |END| Shuffle. |END|
          } else {
            tmp_use_shuffle = ptr_Dataset_Parameters_received->value_0 != 0;
          }

          if (ptr_Dataset_Parameters_received == nullptr ||
              (ptr_Dataset_Parameters_received != nullptr &&
               ptr_Dataset_Parameters_received->value_1 == -1)) {
            // K-fold.
            INFO(L"");
            INFO(L"K-fold:");
            INFO(L"Range[2, %zu].", n_data);
            tmp_number_k_folds =
                parse_discrete(2_UZ, n_data, L"K-fold: ");
            // |END| K-fold. |END|
          } else {
            tmp_number_k_folds =
                static_cast<size_t>(ptr_Dataset_Parameters_received->value_1);
          }

          if (ptr_Dataset_Parameters_received == nullptr ||
              (ptr_Dataset_Parameters_received != nullptr &&
               ptr_Dataset_Parameters_received->value_2 == -1)) {
            // K-sub-fold.
            size_t const tmp_number_examples_training(
                (tmp_number_k_folds - 1_UZ) * (n_data / tmp_number_k_folds));

            INFO(L"");
            INFO(L"K-sub-fold:");
            INFO(L"Range[0, %zu].", tmp_number_examples_training);
            INFO(L"default=%zu.", tmp_number_k_folds - 1_UZ);
            tmp_number_k_sub_folds = parse_discrete(
                0_UZ, tmp_number_examples_training, L"K-sub-fold: ");
            // |END| K-sub-fold. |END|
          } else {
            tmp_number_k_sub_folds =
                static_cast<size_t>(ptr_Dataset_Parameters_received->value_2);
          }

          if (tmp_ptr_Dataset_Cross_Validation->Initialize__Fold(
                  tmp_use_shuffle, tmp_number_k_folds,
                  tmp_number_k_sub_folds) == false) {
            ERR(L"An error has been triggered from the "
                L"`Initialize__Fold(%ls, %zu, %zu)` function.",
                to_wstring(tmp_use_shuffle).c_str(), tmp_number_k_folds,
                tmp_number_k_sub_folds);

            return false;
          }
        }

          if (type_dataset_process_received == DATASET::CROSS_VAL_OPT) {
            CrossValOpt *const
                tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization(
                    dynamic_cast<CrossValOpt *>(dataset));

            size_t tmp_number_hyper_optimization_iterations,
                tmp_number_hyper_optimization_iterations_delay;

            if (ptr_Dataset_Parameters_received == nullptr ||
                (ptr_Dataset_Parameters_received != nullptr &&
                 ptr_Dataset_Parameters_received->value_3 == -1)) {
              INFO(L"");
              INFO(L"Number hyperparameter optimization iteration(s):");
              INFO(L"Range[1, 8].");
              INFO(L"default=10.");
              tmp_number_hyper_optimization_iterations =
                  parse_discrete(0_UZ, L"Iteration(s): ");
            } else {
              tmp_number_hyper_optimization_iterations =
                  ptr_Dataset_Parameters_received->value_3 != 0;
            }

            if (tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
                    ->Set__Number_Hyperparameter_Optimization_Iterations(
                        tmp_number_hyper_optimization_iterations) == false) {
              ERR(L"An error has been triggered from the "
                  L"`Set__Number_Hyperparameter_Optimization_Iterations(%zu)` "
                  L"function.",
                  tmp_number_hyper_optimization_iterations);

              return false;
            }

            if (ptr_Dataset_Parameters_received == nullptr ||
                (ptr_Dataset_Parameters_received != nullptr &&
                 ptr_Dataset_Parameters_received->value_4 == -1)) {
              INFO(L"");
              INFO(L"Number hyperparameter optimization iteration(s) delay:");
              INFO(L"Range[1, 8].");
              INFO(L"default=25.");
              tmp_number_hyper_optimization_iterations_delay =
                  parse_discrete(0_UZ, L"Iteration(s) delay: ");
            } else {
              tmp_number_hyper_optimization_iterations_delay =
                  ptr_Dataset_Parameters_received->value_4 != 0;
            }

            if (tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
                    ->Set__Number_Hyperparameter_Optimization_Iterations_Delay(
                        tmp_number_hyper_optimization_iterations_delay) ==
                false) {
              ERR(L"An error has been triggered from the "
                  L"`Set__Number_Hyperparameter_Optimization_Iterations_Delay"
                  L"(%zu)` function.",
                  tmp_number_hyper_optimization_iterations_delay);
              return false;
            } else if (
                tmp_ptr_Dataset_Cross_Validation_Hyperparameter_Optimization
                    ->user_controls() == false) {
              ERR(L"An error has been triggered from the "
                  L"`user_controls()` function.");
              return false;
            }
          }
          break;
        default:
          ERR(L"DatasetV1 process type (%d | %ls) is not managed "
              L"in the switch.",
              type_dataset_process_received,
              DATASET_NAME[type_dataset_process_received].c_str());
          return false;
      }
      break;
    default:
      ERR(L"DatasetV1 type (%d | %ls) is not managed in the switch.", env_type,
          ENV_NAME[env_type].c_str());
      return false;
  }

  return true;
}

bool Datasets::Preparing_Dataset_Manager(
    struct DatasetsParams const
        *const ptr_Dataset_Manager_Parameters_received) {
  double tmp_percent_training_size, tmp_percent_validation_size,
      tmp_percent_testing_size;

  DatasetV1 *train_set(nullptr), *tmp_ptr_ValidatingSet(nullptr),
      *tmp_ptr_TestingSet(nullptr);

  // Type storage.
  int tmp_type_storage_choose;

  if (ptr_Dataset_Manager_Parameters_received == nullptr ||
      (ptr_Dataset_Manager_Parameters_received != nullptr &&
       ptr_Dataset_Manager_Parameters_received->type_storage == -1)) {
    INFO(L"");
    INFO(L"Type storage: ");
    INFO(L"[0]: Training.");
    INFO(L"[1]: Training and testing.");
    INFO(L"[2]: Training, validation and testing.");

    tmp_type_storage_choose = parse_discrete(0, 2, L"Choose: ");
  } else {
    tmp_type_storage_choose = static_cast<int>(
        ptr_Dataset_Manager_Parameters_received->type_storage);
  }

  // Type training.
  int tmp_type_training_choose;

  if (ptr_Dataset_Manager_Parameters_received == nullptr ||
      (ptr_Dataset_Manager_Parameters_received != nullptr &&
       ptr_Dataset_Manager_Parameters_received->type_train == -1)) {
    INFO(L"");
    INFO(L"Type training: ");
    INFO(L"[0]: Batch.");
    INFO(L"[1]: Mini-batch.");
    INFO(L"[2]: Cross-validation.");
    INFO(L"[3]: Cross-validation, random search.");

    tmp_type_training_choose = parse_discrete(0, 3, L"Choose: ");
  } else {
    tmp_type_training_choose = static_cast<int>(
        ptr_Dataset_Manager_Parameters_received->type_train);
  }

  switch (static_cast<enum ENUM_TYPE_DATASET_MANAGER_STORAGE>(
      tmp_type_storage_choose + 1)) {
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
      if ((train_set = this->Allocate__Dataset(
               static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
               ENV::TRAIN)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
            ENV::TRAIN);
        return false;
      }

      if (this->Prepare_Storage(train_set) == false) {
        ERR(L"An error has been triggered from the "
            L"`Prepare_Storage(ptr)` function.");
        return false;
      }

      if (this->Initialize_Dataset(
              ENV::TRAIN,
              static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
              ptr_Dataset_Manager_Parameters_received == nullptr
                  ? nullptr
                  : &ptr_Dataset_Manager_Parameters_received->train_params) ==
          false) {
        ERR(L"An error has been triggered from the "
            L"`Initialize_Dataset(%d, %d, ptr)` function.",
            ENV::TRAIN,
            static_cast<DATASET::TYPE>(tmp_type_training_choose + 1));
        return false;
      }
      break;
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
      if (ptr_Dataset_Manager_Parameters_received == nullptr ||
          (ptr_Dataset_Manager_Parameters_received != nullptr &&
           ptr_Dataset_Manager_Parameters_received->pct_train_size == 0.0)) {
        INFO(L"");
        tmp_percent_training_size = parse_real<double>(
            0.0001, 99.9999, L"Training size [0.0001%, 99.9999%]: ");
      } else {
        tmp_percent_training_size =
            ptr_Dataset_Manager_Parameters_received->pct_train_size;
      }

      tmp_percent_testing_size = 100.0 - tmp_percent_training_size;

      if ((train_set = this->Allocate__Dataset(
               static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
               ENV::TRAIN)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
            ENV::TRAIN);

        return false;
      }

      if ((tmp_ptr_TestingSet = this->Allocate__Dataset(
               static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
               ENV::TESTG)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
            ENV::VALID);

        return false;
      }

      if (this->Prepare_Storage(tmp_percent_training_size,
                                tmp_percent_testing_size, train_set,
                                tmp_ptr_TestingSet) == false) {
        ERR(L"An error has been triggered from the "
            L"`Prepare_Storage(%f, %f, ptr, ptr)` function.",
            tmp_percent_training_size, tmp_percent_testing_size);

        return false;
      }

      if (this->Initialize_Dataset(
              ENV::TRAIN,
              static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
              ptr_Dataset_Manager_Parameters_received == nullptr
                  ? nullptr
                  : &ptr_Dataset_Manager_Parameters_received->train_params) ==
          false) {
        ERR(L"An error has been triggered from the "
            L"`Initialize_Dataset(%d, %d, ptr)` function.",
            ENV::TRAIN,
            static_cast<DATASET::TYPE>(tmp_type_training_choose + 1));

        return false;
      }
      break;
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::
        TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
      if (ptr_Dataset_Manager_Parameters_received == nullptr ||
          (ptr_Dataset_Manager_Parameters_received != nullptr &&
           ptr_Dataset_Manager_Parameters_received->pct_train_size == 0.0)) {
        INFO(L"");
        tmp_percent_training_size = parse_real<double>(
            1e-4, 100.0 - 1e-4, L"Training size [0.0001%, 99.9999%]: ");
      } else {
        tmp_percent_training_size =
            ptr_Dataset_Manager_Parameters_received->pct_train_size;
      }

      if (ptr_Dataset_Manager_Parameters_received == nullptr ||
          (ptr_Dataset_Manager_Parameters_received != nullptr &&
           ptr_Dataset_Manager_Parameters_received->pct_valid_size == 0.0)) {
        INFO(L"");
        tmp_percent_validation_size = parse_real<double>(
            1e-5, 100.0 - 1e-5 - tmp_percent_training_size,
            (L"Validation size [1e-5%, " +
             to_wstring(100.0 - 1e-5 - tmp_percent_training_size) + L"%]: ")
                .c_str());
      } else {
        tmp_percent_validation_size =
            ptr_Dataset_Manager_Parameters_received->pct_valid_size;
      }

      tmp_percent_testing_size =
          100.0 - tmp_percent_training_size - tmp_percent_validation_size;

      if ((train_set = this->Allocate__Dataset(
               static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
               ENV::TRAIN)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
            ENV::TRAIN);

        return false;
      }

      if ((tmp_ptr_ValidatingSet = this->Allocate__Dataset(
               static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
               ENV::VALID)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
            ENV::VALID);

        return false;
      }

      if ((tmp_ptr_TestingSet = this->Allocate__Dataset(
               static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
               ENV::TESTG)) == nullptr) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%d, %d)` function.",
            static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
            ENV::TESTG);

        return false;
      }

      if (this->Prepare_Storage(
              tmp_percent_training_size, tmp_percent_validation_size,
              tmp_percent_testing_size, train_set, tmp_ptr_ValidatingSet,
              tmp_ptr_TestingSet) == false) {
        ERR(L"An error has been triggered from the "
            L"`Prepare_Storage(%f, %f, %f, ptr, ptr, ptr)` function.",
            tmp_percent_training_size, tmp_percent_validation_size,
            tmp_percent_testing_size);

        return false;
      }

      if (this->Initialize_Dataset(
              ENV::TRAIN,
              static_cast<DATASET::TYPE>(tmp_type_training_choose + 1),
              ptr_Dataset_Manager_Parameters_received == nullptr
                  ? nullptr
                  : &ptr_Dataset_Manager_Parameters_received->train_params) ==
          false) {
        ERR(L"An error has been triggered from the "
            L"`Initialize_Dataset(%d, %d, ptr)` function.",
            ENV::TRAIN,
            static_cast<DATASET::TYPE>(tmp_type_training_choose + 1));

        return false;
      }
      break;
    default:
      ERR(L"DatasetV1 storage type (%d | %ls) is not managed "
          L"in the switch.",
          tmp_type_storage_choose + 1,
          ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES
              [static_cast<enum ENUM_TYPE_DATASET_MANAGER_STORAGE>(
                   tmp_type_storage_choose + 1)]
                  .c_str());
      return false;
  }

  return true;
}

bool Datasets::Copy__Storage(
    Datasets const *const ptr_source_Dataset_Manager_received) {
      if (ptr_source_Dataset_Manager_received->_type_storage_data ==
          ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE) {
    ERR(L"Undefined dataset storage type.");
    return false;
  }

  DatasetV1 const *const train_set(
      ptr_source_Dataset_Manager_received->get_dataset(ENV::TRAIN));

  if (train_set == nullptr) {
    ERR(L"`train_set` is a nullptr.");
    return false;
  } else if (train_set->Get__Type_Dataset_Process() == DATASET::NONE) {
    ERR(L"Undefined dataset process type.");
    return false;
  }

  struct DatasetsParams datasets_params;

  datasets_params.type_storage =
      static_cast<int>(
          ptr_source_Dataset_Manager_received->_type_storage_data) -
      1;
  datasets_params.type_train =
      static_cast<int>(train_set->Get__Type_Dataset_Process()) - 1;

  datasets_params.pct_train_size =
      ptr_source_Dataset_Manager_received->_size_dataset_training__percent;
  datasets_params.pct_valid_size =
      ptr_source_Dataset_Manager_received->_size_dataset_validation__percent;

  switch (train_set->Get__Type_Dataset_Process()) {
    case DATASET::BATCH:
      break;
    case DATASET::MINIBATCH: {
      MiniBatch const *const tmp_ptr_Dataset_Mini_Batch(
          dynamic_cast<MiniBatch const *const>(train_set));

      datasets_params.train_params.value_0 =
          static_cast<int>(tmp_ptr_Dataset_Mini_Batch->Get__Use__Shuffle());
      datasets_params.train_params.value_1 = static_cast<int>(
          tmp_ptr_Dataset_Mini_Batch->Get__Number_Examples_Per_Batch());
      datasets_params.train_params.value_2 =
          static_cast<int>(tmp_ptr_Dataset_Mini_Batch->get_n_batch());
    } break;
    case DATASET::CROSS_VAL: {
      CrossVal const *const tmp_ptr_Dataset_Cross_Validation(
          dynamic_cast<CrossVal const *const>(train_set));

      datasets_params.train_params.value_0 = static_cast<int>(
          tmp_ptr_Dataset_Cross_Validation->Get__Use__Shuffle());
      datasets_params.train_params.value_1 =
          static_cast<int>(tmp_ptr_Dataset_Cross_Validation->get_n_batch());
      datasets_params.train_params.value_2 = static_cast<int>(
          tmp_ptr_Dataset_Cross_Validation->Get__Number_Sub_Batch());
    } break;
    default:
      ERR(L"DatasetV1 process type (%d | %ls) is not managed "
          "in the switch.",
          train_set->Get__Type_Dataset_Process(),
          DATASET_NAME[train_set->Get__Type_Dataset_Process()].c_str());
      return false;
  }

  if (this->Preparing_Dataset_Manager(&datasets_params) == false) {
    ERR(L"An error has been triggered from the "
        "`Preparing_Dataset_Manager(ptr)` function.");
    return false;
  }

  return true;
}

bool Datasets::reference(Datasets *const ptr_source_Dataset_Manager_received) {
  this->Deallocate();

  this->DatasetV1::reference(ptr_source_Dataset_Manager_received->p_n_data -
                                 ptr_source_Dataset_Manager_received->p_str_i,
                             ptr_source_Dataset_Manager_received->Xm +
                                 ptr_source_Dataset_Manager_received->p_str_i,
                             ptr_source_Dataset_Manager_received->Ym +
                                 ptr_source_Dataset_Manager_received->p_str_i,
                             *ptr_source_Dataset_Manager_received);

  this->_reference = true;

  // Private.
  this->_dataset_in_equal_less_holdout_accepted =
      ptr_source_Dataset_Manager_received
          ->_dataset_in_equal_less_holdout_accepted;
  this->_use_metric_loss =
      ptr_source_Dataset_Manager_received->_use_metric_loss;

  this->_maximum_examples =
      ptr_source_Dataset_Manager_received->_maximum_examples;

  this->_minimum_loss_holdout_accepted =
      ptr_source_Dataset_Manager_received->_minimum_loss_holdout_accepted;

  if (this->Copy__Storage(ptr_source_Dataset_Manager_received) == false) {
    ERR(L"An error has been triggered from the "
        L"`Copy__Storage(ptr)` function.");
    return false;
  }

  this->_desired_optimization_time_between_reports =
      ptr_source_Dataset_Manager_received
          ->_desired_optimization_time_between_reports;

  this->_type_evaluation =
      ptr_source_Dataset_Manager_received->_type_evaluation;

#ifdef COMPILE_CUDA
  this->_ptr_CUDA_Dataset_Manager =
      ptr_source_Dataset_Manager_received->_ptr_CUDA_Dataset_Manager;
#endif
  // |END| Private. |END|

  return true;
}

bool Datasets::Deallocate(void) {
  this->Deallocate__Storage();

  if (this->_reference) {
    // Protected.
    this->X = nullptr;
    this->Xm = nullptr;

    this->Y = nullptr;
    this->Ym = nullptr;
    // |END| Protected. |END|

    // Private.
#ifdef COMPILE_CUDA
    this->_ptr_CUDA_Dataset_Manager = nullptr;
#endif
    // |END| Private. |END|
  } else {
#ifdef COMPILE_CUDA
    this->Deallocate_CUDA();
#endif

    if (this->DatasetV1::Deallocate() == false) {
      ERR(L"An error has been triggered from the "
          "`DatasetV1::Deallocate()` function.");
      return false;
    }

    if (this->HyperOpt::Deallocate() == false) {
      ERR(L"An error has been triggered from the "
          "`HyperOpt::Deallocate()` function.");
      return false;
    }
  }

  this->_reference = false;

  return true;
}

double Datasets::train(Model *const model) {
  DatasetV1 *dataset(this->get_dataset(ENV::TRAIN));

  if (dataset == nullptr) {
    ERR(L"An error has been triggered from the "
        L"`get_dataset(%ls)` function. "
        L"Pointer returned is a nullptr.",
        ENV_NAME[ENV::TRAIN].c_str());
    return HUGE_VAL;
  }

  return dataset->train(model);
}

double Datasets::Optimize(Model *const model) {
  if (this->Get__Hyperparameter_Optimization() != HYPEROPT::NONE &&
      ++this->p_optimization_iterations_since_hyper_optimization >=
          this->p_number_hyper_optimization_iterations_delay) {
    this->p_optimization_iterations_since_hyper_optimization = 0_UZ;

    if (this->HyperOpt::Optimize(this, model) == false) {
      ERR(L"An error has been triggered from the "
          L"`Optimize(ptr, ptr)` function.");
      return (std::numeric_limits<real>::max)();
    }

    return model->get_loss(ENV::TRAIN);
  } else {
#ifdef COMPILE_CUDA
    if (model->Use__CUDA()) {
      return this->Get__CUDA()->train(model);
    } else
#endif
    {
      return this->train(model);
    }
  }
}

double Datasets::Type_Testing(ENV::TYPE const env_type, Model *const model) {
  DatasetV1 *dataset(this->get_dataset(env_type));

  if (dataset == nullptr) {
    ERR(L"An error has been triggered from the "
        L"`get_dataset(%ls)` function. "
        L"Pointer return is a nullptr.",
        ENV_NAME[env_type].c_str());
    return HUGE_VAL;
  }

  double const tmp_previous_loss(model->get_loss(ENV::TESTG)),
      tmp_previous_accuracy(model->get_accu(ENV::TESTG)),
      tmp_loss(dataset->evaluate(model));  // By default: loss_testg = tmp_loss;

  switch (env_type) {
    case ENV::TRAIN:
      model->set_loss(ENV::TRAIN, tmp_loss);
      model->set_accu(ENV::TRAIN, model->get_accu(ENV::TESTG));
      break;
    case ENV::VALID:
      model->set_loss(ENV::VALID, tmp_loss);
      model->set_accu(ENV::VALID, model->get_accu(ENV::TESTG));
      break;
    case ENV::TESTG:
      break;
    default:
      ERR(L"DatasetV1 type (%d | %ls) is not managed in the switch.", env_type,
          ENV_NAME[env_type].c_str());
      return (std::numeric_limits<real>::max)();
  }

  // reset testing loss/accuracy.
  if (env_type != ENV::TESTG) {
    model->set_loss(ENV::TESTG, tmp_previous_loss);
    model->set_accu(ENV::TESTG, tmp_previous_accuracy);
  }
  // |END| reset testing loss/accuracy. |END|

  return tmp_loss;
}

std::pair<double, double> Datasets::Type_Update_Batch_Normalization(
    ENV::TYPE const env_type, Model *const model) {
  model->type_state_propagation = PROPAGATION::UPDATE_BATCH_NORM;

  double const tmp_previous_loss(model->get_loss(env_type)),
      tmp_previous_accuracy(model->get_accu(env_type)),
      tmp_loss(this->Type_Testing(env_type, model));
  double const tmp_accuracy(model->get_accu(env_type));

  model->type_state_propagation = PROPAGATION::INFERENCE;

  // reset past loss/accuracy.
  if (env_type != ENV::TESTG) {
    model->set_loss(env_type, tmp_previous_loss);
    model->set_accu(env_type, tmp_previous_accuracy);
  }
  // |END| reset past loss/accuracy. |END|

  return (std::make_pair(tmp_loss, tmp_accuracy));
}

double Datasets::evaluate(Model *const model) {
  if (this->Get__Evaluation_Require()) {
    if (this->Evaluation(this) == false) {
      ERR(L"An error has been triggered from the "
          L"`evaluate(ptr)` function.");
      return (std::numeric_limits<real>::max)();
    }

    return model->get_loss(ENV::VALID);
  } else {
#ifdef COMPILE_CUDA
    if (model->Use__CUDA()) {
      return this->Get__CUDA()->Type_Testing(ENV::VALID, model);
    } else
#endif
    {
      return this->Type_Testing(ENV::VALID, model);
    }
  }
}

ENUM_TYPE_DATASET_MANAGER_STORAGE Datasets::get_storage_type(void) const {
  return this->_type_storage_data;
}

DatasetV1 *Datasets::get_dataset(ENV::TYPE const env_type) const {
  switch (this->_type_storage_data) {
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
      return (this->_ptr_array_ptr_Dataset[0]);
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
      switch (env_type) {
        case ENV::TRAIN:
          return this->_ptr_array_ptr_Dataset[0];
        case ENV::VALID:
          return this->_ptr_array_ptr_Dataset[0];
        case ENV::TESTG:
          return this->_ptr_array_ptr_Dataset[1];
        default:
          ERR(L"DatasetV1 type (%d | %ls) is not managed "
              L"in the switch.",
              env_type, ENV_NAME[env_type].c_str());
          return nullptr;
      }
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::
        TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
      switch (env_type) {
        case ENV::TRAIN:
          return this->_ptr_array_ptr_Dataset[0];
        case ENV::VALID:
          return this->_ptr_array_ptr_Dataset[1];
        case ENV::TESTG:
          return this->_ptr_array_ptr_Dataset[2];
        default:
          ERR(L"DatasetV1 type (%d | %ls) is not managed "
              L"in the switch.",
              env_type, ENV_NAME[env_type].c_str());
          return nullptr;
      }
    default:
      ERR(L"DatasetV1 storage type (%d | %ls) is not managed "
          L"in the switch.",
          this->_type_storage_data,
          ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[this->_type_storage_data]
              .c_str());
      return nullptr;
  }

  return nullptr;
}

#ifdef COMPILE_CUDA
cuDatasets *Datasets::Get__CUDA(void) {
  return this->_ptr_CUDA_Dataset_Manager;
}
#endif

Datasets::~Datasets(void) { this->Deallocate(); }
}  // namespace DL::v1
