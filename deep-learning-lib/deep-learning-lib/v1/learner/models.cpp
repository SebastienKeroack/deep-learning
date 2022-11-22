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

#include "deep-learning-lib/v1/learner/models.hpp"
#include "deep-learning-lib/v1/data/datasets.hpp"
#include "deep-learning-lib/data/string.hpp"
#include "deep-learning-lib/data/time.hpp"
#include "deep-learning-lib/v1/data/enum/activation.hpp"
#include "deep-learning-lib/v1/data/enum/dataset.hpp"
#include "deep-learning-lib/v1/data/enum/hierarchy.hpp"
#include "deep-learning-lib/v1/data/enum/loss_fn.hpp"
#include "deep-learning-lib/device/system/info.hpp"
#include "deep-learning-lib/device/system/shutdown_block.hpp"
#include "deep-learning-lib/io/file.hpp"
#include "deep-learning-lib/io/logger.hpp"
#include "deep-learning-lib/io/term/input.hpp"
#include "deep-learning-lib/io/term/spinner.hpp"

#include <fstream>
#include <iostream>
#include <array>

using namespace DL::File;
using namespace DL::Str;
using namespace DL::Term;

namespace DL::v1 {
Models::Models(void) {}

bool Models::initialize_dirs(std::wstring const &name) {
  std::wstring const home(home_directory() + L"deep-learning");
  
  if (path_exist(home) == false) {
    ERR(L"Directory `%ls` not found.", home);
    return false;
  }
  
  this->_path_root = home + OS_SEP + L"deep-learning" + OS_SEP + name;
  this->_path_trainer_dir = this->_path_root + OS_SEP + L"trainer";
  this->_path_trained_dir = this->_path_root + OS_SEP + L"trained";
  this->_path_dataset_dir = this->_path_root + OS_SEP + L"dataset";

  // ...//DL//name//model
  if (create_directories(this->_path_root) == false) {
    ERR(L"An error has been triggered from the `create_directories(%ls)` function.",
        this->_path_root.c_str());
    return false;
  }

  // ...//DL//name//model//trainer
  if (create_directory(this->_path_trainer_dir) == false) {
    ERR(L"An error has been triggered from the `create_directory(%ls)` function.",
        this->_path_trainer_dir
            .c_str());
    return false;
  }

  // ...//DL//name//model//trained
  if (create_directory(this->_path_trained_dir) == false) {
    ERR(L"An error has been triggered from the `create_directory(%ls)` function.",
        this->_path_trained_dir.c_str());
    return false;
  }

  // ...//DL//name//dataset
  if (create_directory(this->_path_dataset_dir) == false) {
    ERR(L"An error has been triggered from the `create_directory(%ls)` function.",
        this->_path_dataset_dir.c_str());
    return false;
  }

  return true;
}

bool Models::initialize_datasets(DatasetsParams const *const config) {
  if (this->_ptr_Dataset_Manager != nullptr) {
    ERR(L"...");

    return false;
  }

  DATASET_FORMAT::TYPE tmp_type_dataset_file;

  if (scan_datasets(tmp_type_dataset_file, this->_path_dataset_dir) == false) {
    ERR(L"An error has been triggered from the `scan_datasets(%ls)` function.",
        this->_path_dataset_dir.c_str());

    return false;
  }

  INFO(L"");
  Spinner spinner;
  spinner.start((L"Loading from " + this->_path_dataset_dir + L"... ").c_str());

  this->_ptr_Dataset_Manager =
      new Datasets(tmp_type_dataset_file, this->_path_dataset_dir);

  spinner.join();
  INFO(L"");

  if (this->_ptr_Dataset_Manager->Set__Maximum_Data(
          this->_ptr_Dataset_Manager->get_n_data()) == false) {
    ERR(L"An error has been triggered from the `Set__Maximum_Data(%d)` "
        "function.",
        this->_ptr_Dataset_Manager->get_n_data());

    SAFE_DELETE(this->_ptr_Dataset_Manager);

    return false;
  } else if (this->_ptr_Dataset_Manager->Preparing_Dataset_Manager(config) ==
             false) {
    ERR(L"An error has been triggered from the `Preparing_Dataset_Manager()` "
        "function.",
        this->_ptr_Dataset_Manager->get_n_data());

    SAFE_DELETE(this->_ptr_Dataset_Manager);

    return false;
  }

  if (this->_use_cuda &&
      this->_ptr_Dataset_Manager->Initialize__CUDA() == false) {
    ERR(L"An error has been triggered from the `Initialize__CUDA()` "
        "function.",
        this->_ptr_Dataset_Manager->get_n_data());

    SAFE_DELETE(this->_ptr_Dataset_Manager);

    return false;
  }

  return true;
}

bool Models::create(
    size_t const maximum_allowable_host_memory_bytes_received) {
  if (this->_ptr_trainer_Neural_Network != nullptr)
    this->Deallocate__Neural_Network(HIERARCHY::TRAINER);

  INFO(L"");
  if (accept(L"Do you want to use template?")) {
    // Neural network initializer.
    Neural_Network_Initializer tmp_Neural_Network_Initializer;

    if (tmp_Neural_Network_Initializer.Template_Initialize() == false) {
      ERR(L"Function \"Template_Initialize()\" return false.");
      return false;
    }

    if ((this->_ptr_trainer_Neural_Network =
             tmp_Neural_Network_Initializer.Output_Initialize(
                 maximum_allowable_host_memory_bytes_received)) == nullptr) {
      ERR(L"Function \"Output_Initialize()\" return false.");
      return false;
    }
    // |END| Neural network initializer. |END|

    size_t tmp_layer_index, tmp_residual_index(0_UZ),
        tmp_total_residual_blocks(0_UZ);

    INFO(L"");
    INFO(L"ShakeDrop, dropout probability.");
    INFO(L"Range[0, %f].",
           1_r - 1e-7_r);
    real const tmp_shakedrop_dropout_probability(
        parse_real<real>(
        0_r, 1_r - 1e-7_r, L"Dropout probability: "));

    INFO(L"");
    INFO(L"Activation functions:");
    for (unsigned int tmp_activation_function_index(1u);
         tmp_activation_function_index != ACTIVATION::LENGTH;
         ++tmp_activation_function_index) {
      INFO(L"[%d]: %ls.",
             tmp_activation_function_index,
             ACTIVATION_NAME[static_cast<ACTIVATION::TYPE>(
                                     tmp_activation_function_index)]
                 .c_str());
    }
    INFO(L"default=%ls.",
           ACTIVATION_NAME[ACTIVATION::LEAKY_RELU].c_str());

    ACTIVATION::TYPE tmp_type_activation_function_hidden,
        tmp_type_activation_function_output;

    if ((tmp_type_activation_function_hidden =
             static_cast<ACTIVATION::TYPE>(
                 parse_discrete<int>(
                     1, ACTIVATION::LENGTH - 1, L"Hidden layer, activation function: "))) >=
        ACTIVATION::LENGTH) {
      ERR(
          L"An error has been triggered from the "
          "\"parse_discrete<int>(%u, %u)\" function.", 1,
          ACTIVATION::LENGTH - 1u);

      return false;
    }

    if ((tmp_type_activation_function_output =
             static_cast<ACTIVATION::TYPE>(
                 parse_discrete<int>(
                     1, ACTIVATION::LENGTH - 1, L"Output layer, activation function: "))) >=
        ACTIVATION::LENGTH) {
      ERR(
          L"An error has been triggered from the "
          "\"parse_discrete<int>(%u, %u)\" function.", 1,
          ACTIVATION::LENGTH - 1u);

      return false;
    }

    for (tmp_layer_index = 1_UZ;
         tmp_layer_index !=
         this->_ptr_trainer_Neural_Network->total_layers - 1_UZ;
         ++tmp_layer_index) {
      if (this->_ptr_trainer_Neural_Network->ptr_array_layers[tmp_layer_index]
              .type_layer == LAYER::RESIDUAL) {
        ++tmp_total_residual_blocks;
      }
    }

    for (tmp_layer_index = 1_UZ;
         tmp_layer_index !=
         this->_ptr_trainer_Neural_Network->total_layers - 1_UZ;
         ++tmp_layer_index) {
      switch (
          this->_ptr_trainer_Neural_Network->ptr_array_layers[tmp_layer_index]
              .type_layer) {
        case LAYER::FULLY_CONNECTED:
        case LAYER::FULLY_CONNECTED_RECURRENT:
        case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
          if (this->_ptr_trainer_Neural_Network->set_layer_activation_function(
                  tmp_layer_index, tmp_type_activation_function_hidden) ==
              false) {
            ERR(
                L"An error has been triggered from the "
                "\"set_layer_activation_function(%zu, %u)\" function.", tmp_layer_index,
                tmp_type_activation_function_hidden);

            return false;
          } else if (this->_ptr_trainer_Neural_Network
                         ->Set__Layer_Normalization(
                             tmp_layer_index,
                             LAYER_NORM::BATCH_RENORMALIZATION) == false) {
            ERR(
                L"An error has been triggered from the "
                "\"Set__Layer_Normalization(%zu, %u)\" function.", tmp_layer_index,
                LAYER_NORM::BATCH_RENORMALIZATION);

            return false;
          }
          break;
        case LAYER::LSTM:
          if (this->_ptr_trainer_Neural_Network->Set__Layer_Normalization(
                  tmp_layer_index, LAYER_NORM::BATCH_RENORMALIZATION) ==
              false) {
            ERR(
                L"An error has been triggered from the "
                "\"Set__Layer_Normalization(%zu, %u)\" function.", tmp_layer_index,
                LAYER_NORM::BATCH_RENORMALIZATION);

            return false;
          }
          break;
        case LAYER::RESIDUAL:
          if (this->_ptr_trainer_Neural_Network->Set__Layer_Normalization(
                  tmp_layer_index, LAYER_NORM::BATCH_RENORMALIZATION) ==
              false) {
            ERR(
                L"An error has been triggered from the "
                "\"Set__Layer_Normalization(%zu, %u)\" function.", tmp_layer_index,
                LAYER_NORM::BATCH_RENORMALIZATION);

            return false;
          } else if (tmp_shakedrop_dropout_probability != 0_r &&
                     this->_ptr_trainer_Neural_Network->set_dropout(
                         tmp_layer_index, LAYER_DROPOUT::SHAKEDROP,
                         std::array<real, 1_UZ>{
                             1_r -
                             ((static_cast<real>(++tmp_residual_index) /
                               static_cast<real>(tmp_total_residual_blocks)) *
                              (1_r - tmp_shakedrop_dropout_probability))}
                             .data()) == false) {
            ERR(
                L"An error has been triggered from the "
                "\"set_dropout(%zu, %u, %f)\" function.", tmp_layer_index,
                LAYER_DROPOUT::SHAKEDROP,
                1_r - ((static_cast<real>(tmp_residual_index) /
                             static_cast<real>(tmp_total_residual_blocks)) *
                              (1_r - tmp_shakedrop_dropout_probability)));

            return false;
          }
          break;
        default:
          ERR(
              L"Layer type (%d | %ls) is not managed in the "
              "switch.",
              this->_ptr_trainer_Neural_Network
                  ->ptr_array_layers[tmp_layer_index]
                  .type_layer,
              LAYER_NAME[this->_ptr_trainer_Neural_Network
                                 ->ptr_array_layers[tmp_layer_index]
                                 .type_layer]
                  .c_str());
          return false;
      }
    }

    // Output layer.
    if (this->_ptr_trainer_Neural_Network->set_layer_activation_function(
            tmp_layer_index, tmp_type_activation_function_output) == false) {
      ERR(
          L"An error has been triggered from the "
          "\"set_layer_activation_function(%zu, %u)\" function.", tmp_layer_index,
          tmp_type_activation_function_output);

      return false;
    }
    // |END| Output layer. |END|
  } else {
    // Neural network initializer.
    Neural_Network_Initializer tmp_Neural_Network_Initializer;

    if (tmp_Neural_Network_Initializer.Input_Initialize() == false) {
      ERR(L"Function \"Input_Initialize()\" return false.");
      return false;
    }

    if ((this->_ptr_trainer_Neural_Network =
             tmp_Neural_Network_Initializer.Output_Initialize(
                 maximum_allowable_host_memory_bytes_received)) == nullptr) {
      ERR(L"Function \"Output_Initialize()\" return false.");
      return false;
    }
    // |END| Neural network initializer. |END|

    // Activation functions.
    Activation_Function_Initializer tmp_Activation_Function_Initializer;

    if (tmp_Activation_Function_Initializer.Input_Initialize(
            this->_ptr_trainer_Neural_Network->total_layers,
            this->_ptr_trainer_Neural_Network->type) == false) {
      ERR(L"Function \"Input_Initialize()\" return false.");
      return false;
    }

    if (tmp_Activation_Function_Initializer.Output_Initialize(
            this->_ptr_trainer_Neural_Network) == false) {
      ERR(L"Function \"Output_Initialize()\" return false.");

      return false;
    }
    // |END| Activation functions. |END|

    // Dropout.
    INFO(L"");
    if (accept(L"Do you want to use dropout?")) {
      Dropout_Initializer tmp_Dropout_Initializer;

      if (tmp_Dropout_Initializer.Input_Initialize(
              this->_ptr_trainer_Neural_Network->total_layers,
              this->_ptr_trainer_Neural_Network->type) == false) {
        ERR(L"Function \"Input_Initialize()\" return false.");
        return false;
      }

      if (tmp_Dropout_Initializer.Output_Initialize(
              this->_ptr_trainer_Neural_Network) == false) {
        ERR(L"Function \"Output_Initialize()\" return false.");
        return false;
      }
    }
    // |END| Dropout. |END|

    // Normalization.
    INFO(L"");
    if (accept(L"Do you want to use normalization?")) {
      Normalization_Initializer tmp_Normalization_Initializer;

      if (tmp_Normalization_Initializer.Input_Initialize(
              this->_ptr_trainer_Neural_Network->total_layers,
              this->_ptr_Dataset_Manager->get_dataset(ENV::TRAIN) ==
                      nullptr
                  ? 1_UZ
                  : this->_ptr_Dataset_Manager->get_dataset(ENV::TRAIN)
                        ->get_n_batch(),
              this->_ptr_trainer_Neural_Network->type) == false) {
        ERR(L"Function \"Input_Initialize()\" return false.");
        return false;
      }

      if (tmp_Normalization_Initializer.Output_Initialize(
              this->_ptr_trainer_Neural_Network) == false) {
        ERR(L"Function \"Output_Initialize()\" return false.");
        return false;
      }
    }
    // |END| Normalization. |END|

    // Tied parameter.
    INFO(L"");
    if (accept(L"Do you want to use tied parameter?") &&
        this->_ptr_trainer_Neural_Network->User_Controls__Tied__Parameter() ==
            false) {
      ERR(L"Function \"User_Controls__Tied__Parameter()\" return false.");
      return false;
    }
    // |END| Tied parameter. |END|

    // k-Sparse.
    INFO(L"");
    if (accept(L"Do you want to use k-Sparse?") &&
        this->_ptr_trainer_Neural_Network->User_Controls__K_Sparse() == false) {
      ERR(L"Function \"User_Controls__K_Sparse()\" return false.");
      return false;
    }
    // |END| k-Sparse. |END|
  }

  // Loss function.
  Loss_Function_Initializer tmp_Loss_Function_Initializer;

  if (tmp_Loss_Function_Initializer.Input_Initialize() == false) {
    ERR(L"Function \"Input_Initialize()\" return false.");
    return false;
  }

  tmp_Loss_Function_Initializer.Output_Initialize(
      this->_ptr_trainer_Neural_Network);
  // |END| Loss function. |END|

  // Accuracy function.
  Accuracy_Function_Initializer tmp_Accuracy_Function_Initializer;

  if (tmp_Accuracy_Function_Initializer.Input_Initialize() == false) {
    ERR(L"Function \"Input_Initialize()\" return false.");
    return false;
  }

  tmp_Accuracy_Function_Initializer.Output_Initialize(
      this->_ptr_trainer_Neural_Network);
  // |END| Accuracy function. |END|

  // Accuracy variance.
  if (this->_ptr_trainer_Neural_Network->type_accuracy_function ==
          ACCU_FN::DISTANCE &&
      this->_ptr_trainer_Neural_Network->User_Controls__Accuracy_Variance() ==
          false) {
    ERR(L"Function \"User_Controls__Accuracy_Variance()\" return false.");
    return false;
  }
  // |END| Accuracy variance. |END|

  // Optimizer function.
  Optimizer_Function_Initializer tmp_Optimizer_Function_Initializer;

  if (tmp_Optimizer_Function_Initializer.Input_Initialize() == false) {
    ERR(L"Function \"Input_Initialize()\" return false.");
    return false;
  }

  if (tmp_Optimizer_Function_Initializer.Output_Initialize(
          this->_ptr_trainer_Neural_Network) == false) {
    ERR(L"Function \"Output_Initialize()\" return false.");
    return false;
  }
  // |END| Optimizer function. |END|

  // Warm restarts.
  Warm_Restarts_Initializer tmp_Warm_Restarts_Initializer;

  tmp_Warm_Restarts_Initializer.Input_Initialize();

  if (this->_ptr_trainer_Neural_Network->Usable_Warm_Restarts() &&
      tmp_Warm_Restarts_Initializer.Output_Initialize(
          this->_ptr_trainer_Neural_Network) == false) {
    ERR(L"Function \"Output_Initialize()\" return false.");
    return false;
  }
  // |END| Warm restarts. |END|

  // clip gradient.
  if (this->_ptr_trainer_Neural_Network->User_Controls__Clip_Gradient() ==
      false) {
    ERR(L"Function \"User_Controls__Clip_Gradient()\" return false.");
    return false;
  }
  // |END| clip gradient. |END|

  // Regularization Max-norm.
  if (this->_ptr_trainer_Neural_Network->User_Controls__Max_Norm_Constaints() ==
      false) {
    ERR(L"Function \"User_Controls__Max_Norm_Constaints()\" return false.");
    return false;
  }
  // |END| Regularization Max-norm. |END|

  // Regularization L1.
  if (this->_ptr_trainer_Neural_Network->User_Controls__L1_Regularization() ==
      false) {
    ERR(L"Function \"User_Controls__L1_Regularization()\" return false.");
    return false;
  }
  // |END| Regularization L1. |END|

  // Regularization L2.
  if (this->_ptr_trainer_Neural_Network->User_Controls__L2_Regularization() ==
      false) {
    ERR(L"Function \"User_Controls__L2_Regularization()\" return false.");
    return false;
  }
  // |END| Regularization L2. |END|

  // Regularization SRIP.
  if (this->_ptr_trainer_Neural_Network->User_Controls__SRIP_Regularization() ==
      false) {
    ERR(L"Function \"User_Controls__SRIP_Regularization()\" return false.");
    return false;
  }
  // |END| Regularization SRIP. |END|

  // Weights initializer.
  Weights_Initializer tmp_Weights_Initializer;

  if (tmp_Weights_Initializer.Input_Initialize() == false) {
    ERR(L"Function \"Input_Initialize()\" return false.");
    return false;
  }

  if (tmp_Weights_Initializer.Output_Initialize(
          this->_ptr_trainer_Neural_Network) == false) {
    ERR(L"Function \"Output_Initialize()\" return false.");
    return false;
  }
  // |END| Weights initializer. |END|

  // Batch size.
  if (this->_ptr_trainer_Neural_Network->User_Controls__Maximum__Batch_Size() ==
      false) {
    ERR(L"Function \"User_Controls__Maximum__Batch_Size()\" return false.");
    return false;
  }
  // |END| Batch size. |END|

  // OpenMP.
  INFO(L"");
  if (accept(L"Do you want to use OpenMP?")) {
    INFO(L"");
    INFO(L"Maximum threads:");
    INFO(L"Range[0.0%%, 100.0%%].");
    this->_ptr_trainer_Neural_Network->pct_threads =
        parse_real<real>(0_r, 100_r, L"Maximum threads (percent): ");

    INFO(L"");
    INFO(L"Initialize OpenMP.");
    if (this->_ptr_trainer_Neural_Network->set_mp(true) == false) {
      ERR(
          L"An error has been triggered from the "
          "\"set_mp(true)\" function.",);

      return false;
    }
  }
  // |END| OpenMP. |END|

  // CUDA.
#ifdef COMPILE_CUDA
  INFO(L"");
  if (accept(L"Do you want to use CUDA?")) {
    int device_id(-1);

    size_t allowable_devc_mem(0_UZ);

    INFO(L"");
    this->set_use_cuda(CUDA__Input__Use__CUDA(
        device_id, allowable_devc_mem));

    INFO(L"");
    INFO(L"Initialize CUDA.");
    if (this->_ptr_trainer_Neural_Network->set_cu(
            this->_use_cuda, allowable_devc_mem) ==
        false) {
      ERR(
          L"An error has been triggered from the \"set_cu(%ls, "
          "%zu)\" function.",
          this->_use_cuda ? "true" : "false",
          allowable_devc_mem);

      return false;
    }
  }
#endif
  // |END| CUDA. |END|

  // copy trainer neural networks parameters to competitor neural network
  // general parameters.
  SAFE_DELETE(this->_ptr_competitor_Neural_Network);
  this->_ptr_competitor_Neural_Network = new Model;

  INFO(L"");
  INFO(
      L"copy dimension and hyper-parameters into a neural network called "
      "competitor.");
  if (this->_ptr_competitor_Neural_Network->copy(
          *this->_ptr_trainer_Neural_Network, true) == false) {
    ERR(L"Function \"copy()\" return false.");
    return false;
  }
  // |END| copy trainer neural networks parameters to competitor neural network
  // general parameters. |END|

  // copy trainer neural networks parameters to trained neural network general
  // parameters.
  SAFE_DELETE(this->_ptr_trained_Neural_Network);
  this->_ptr_trained_Neural_Network = new Model;

  INFO(
      L"copy dimension and hyper-parameters into a neural network called "
      "trained.");
  if (this->_ptr_trained_Neural_Network->copy(
          *this->_ptr_trainer_Neural_Network, true) == false) {
    ERR(L"Function \"copy()\" return false.");
    return false;
  }
  // |END| copy trainer neural networks parameters to trained neural network
  // general parameters. |END|

  return true;
}

void Models::set_use_cuda(bool const use_cu) {
  this->_use_cuda = use_cu;
}

void Models::Set__Auto_Save_Dataset(bool const auto_save_received) {
  this->_auto_save_dataset = auto_save_received;
}

void Models::auto_save_trainer(
    bool const auto_save_received) {
  this->_optimization_auto_save_trainer = auto_save_received;
}

void Models::auto_save_competitor(
    bool const auto_save_received) {
  this->_optimization_auto_save_competitor = auto_save_received;
}

void Models::auto_save_trained(
    bool const auto_save_received) {
  this->_optimization_auto_save_trained = auto_save_received;
}

void Models::Set__Comparison_Expiration(
    size_t const expiration_seconds_received) {
  this->_expiration_seconds = expiration_seconds_received;
}

bool Models::Set__Output_Mode(bool const use_last_layer_as_output_received,
                              HIERARCHY::TYPE const hierarchy) {
  Model *model;

  switch (hierarchy) {
    case HIERARCHY::TRAINER:
      model = this->_ptr_trainer_Neural_Network;
      break;
    case HIERARCHY::TRAINED:
      model = this->_ptr_trained_Neural_Network;
      break;
    default:
      ERR(L"Undefined `%ls` in the switch.", HIERARCHY_NAME[hierarchy].c_str());
      return false;
  }

  return (model->Set__Output_Mode(
      use_last_layer_as_output_received));
}

bool Models::set_while_cond(
    WhileCond &while_cond) {
  if (while_cond.type == WHILE_MODE::LENGTH) {
    ERR(L"Type `%ls` is not valid.",
            WHILE_MODE_NAME[while_cond.type]);
    return false;
  }

  this->_while_cond_opt = while_cond;

  return true;
}

bool Models::Set__Number_Inputs(size_t const number_inputs_received) {
  if (number_inputs_received == 0_UZ) {
    ERR(L"Arg `%zu` should be greater than 0.", number_inputs_received);
    return false;
  }

  this->_n_inp = number_inputs_received;

  return true;
}

bool Models::Set__Number_Outputs(size_t const number_outputs_received) {
  if (number_outputs_received == 0_UZ) {
    ERR(L"Arg `%zu` should be greater than 0.", number_outputs_received);
    return false;
  }

  this->_n_out = number_outputs_received;

  return true;
}

bool Models::Set__Number_Recurrent_Depth(
    size_t const number_recurrent_depth_received) {
  if (number_recurrent_depth_received == 0_UZ) {
    ERR(L"Arg `%zu` should be greater than 0.",
            number_recurrent_depth_received);
    return false;
  }

  this->_seq_w = number_recurrent_depth_received;

  return true;
}

bool Models::set_desired_loss(double const desired_loss_received) {
  if (desired_loss_received < 0.0) {
    ERR(L"Arg `%f` should be greater or equal to 0.", desired_loss_received);
    return false;
  }

  this->_desired_loss = desired_loss_received;

  return true;
}

bool Models::require_evaluate_envs(void) const {
  return this->_require_evaluate_envs;
}

// TODO: Merge to DatasetV1 class.
bool Models::Get__Is_Output_Symmetric(void) const { return false; }

bool Models::Get__Path_Neural_Network_Exist(
    HIERARCHY::TYPE const hierarchy) const {
  return (path_exist(this->get_model_path(hierarchy, L"net")) && path_exist(this->get_model_path(hierarchy, L"nn"))); }

size_t Models::get_n_inp(HIERARCHY::TYPE const hierarchy) const {
  Model *model;

  switch (hierarchy) {
    case HIERARCHY::TRAINER:
      model = this->_ptr_trainer_Neural_Network;
      break;
    case HIERARCHY::TRAINED:
      model = this->_ptr_trained_Neural_Network;
      break;
    default:
      ERR(L"Undefined `%ls` in the switch.", HIERARCHY_NAME[hierarchy].c_str());
      return (0_UZ);
  }

  if (model != nullptr) {
    return (model->n_inp);
  } else {
    return (this->_n_inp);
  }
}

size_t Models::get_n_out(HIERARCHY::TYPE const hierarchy) const {
  Model *model;

  switch (hierarchy) {
    case HIERARCHY::TRAINER:
      model = this->_ptr_trainer_Neural_Network;
      break;
    case HIERARCHY::TRAINED:
      model = this->_ptr_trained_Neural_Network;
      break;
    default:
      ERR(L"Undefined `%ls` in the switch.", HIERARCHY_NAME[hierarchy].c_str());
      return (0_UZ);
  }

  if (model != nullptr) {
    return (model->get_n_out());
  } else {
    return (this->_n_out);
  }
}

size_t Models::get_seq_w(
    HIERARCHY::TYPE const hierarchy) const {
  Model *model;

  switch (hierarchy) {
    case HIERARCHY::TRAINER:
      model = this->_ptr_trainer_Neural_Network;
      break;
    case HIERARCHY::TRAINED:
      model = this->_ptr_trained_Neural_Network;
      break;
    default:
      ERR(L"Undefined `%ls` in the switch.", HIERARCHY_NAME[hierarchy].c_str());
      return (0_UZ);
  }

  if (model != nullptr) {
    return (model->seq_w);
  } else {
    return (this->_seq_w);
  }
}

double Models::get_loss(HIERARCHY::TYPE const hierarchy,
                     ENV::TYPE const type) const {
  Model *model;

  switch (hierarchy) {
    case HIERARCHY::TRAINER:
      model = this->_ptr_trainer_Neural_Network;
      break;
    case HIERARCHY::TRAINED:
      model = this->_ptr_trained_Neural_Network;
      break;
    default:
      ERR(L"Undefined `%ls` in the switch.", HIERARCHY_NAME[hierarchy].c_str());
      return HUGE_VAL;
  }

  if (model == nullptr) {
    ERR(L"Invalid pointer.");
    return HUGE_VAL;
  } else {
    return (model->get_loss(type));
  }
}

double Models::get_accu(HIERARCHY::TYPE const hierarchy,
                         ENV::TYPE const type) const {
  Model *model;

  switch (hierarchy) {
    case HIERARCHY::TRAINER:
      model = this->_ptr_trainer_Neural_Network;
      break;
    case HIERARCHY::TRAINED:
      model = this->_ptr_trained_Neural_Network;
      break;
    default:
      ERR(L"Undefined `%ls` in the switch.", HIERARCHY_NAME[hierarchy].c_str());
      return HUGE_VAL;
  }

  if (model == nullptr) {
    ERR(L"Invalid pointer.");
    return HUGE_VAL;
  } else {
    return (model->get_accu(type));
  }
}

var const *const Models::Get__Output(size_t const time_step_index_received,
                                    HIERARCHY::TYPE const hierarchy) const {
  Model *model;

  switch (hierarchy) {
    case HIERARCHY::TRAINER:
      model = this->_ptr_trainer_Neural_Network;
      break;
    case HIERARCHY::TRAINED:
      model = this->_ptr_trained_Neural_Network;
      break;
    default:
      ERR(L"Undefined `%ls` in the switch.", HIERARCHY_NAME[hierarchy].c_str());
      return nullptr;
  }

  if (model == nullptr) {
    ERR(L"Invalid pointer.");
    return nullptr;
  } else if (time_step_index_received >=
             model->seq_w) {
    ERR(L"Arg out of range.");
    return nullptr;
  } else {
    return (
        model->get_out(0_UZ, time_step_index_received));
  }
}

std::wstring Models::get_model_path(
    HIERARCHY::TYPE const hierarchy,
    std::wstring const path_postfix_received) const {
  std::wstring path_name;

  switch (hierarchy) {
    case HIERARCHY::TRAINER:
      path_name = this->_path_trainer_dir + L'.' + path_postfix_received;
      break;
    case HIERARCHY::TRAINED:
      path_name = this->_path_trained_dir + L'.' + path_postfix_received;
      break;
    default:
      path_name = L"";
      ERR(L"Undefined `%ls` in the switch.", HIERARCHY_NAME[hierarchy].c_str());
      break;
  }

  return (path_name);
}

std::wstring Models::Get__Path_Dataset_Manager(void) const {
  return (this->_path_dataset_dir);
}

Datasets *Models::get_datasets(void) {
  return (this->_ptr_Dataset_Manager);
}

Model *
Models::get_model(HIERARCHY::TYPE const hierarchy) {
  Model *model;

  switch (hierarchy) {
    case HIERARCHY::TRAINER:
      model = this->_ptr_trainer_Neural_Network;
      break;
    case HIERARCHY::TRAINED:
      model = this->_ptr_trained_Neural_Network;
      break;
    case HIERARCHY::COMPETITOR:
      model = this->_ptr_competitor_Neural_Network;
      break;
    default:
      ERR(L"Undefined `%ls` in the switch.", HIERARCHY_NAME[hierarchy].c_str());
      return nullptr;
  }

  return (model);
}

bool Models::append_to_dataset(real const *const ptr_array_inputs_received,
                               real const *const ptr_array_outputs_received) {
  if (ptr_array_inputs_received == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  } else if (ptr_array_outputs_received == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  } else if (this->_ptr_Dataset_Manager == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  } else if (Sys::shutdownblock->preparing_for_shutdown()) {
    INFO(L"Preparing for shutdown.");
    return false;
  }

  if (this->_ptr_Dataset_Manager->push_back(
          ptr_array_inputs_received, ptr_array_outputs_received) == false) {
    ERR(L"Function \"push_back()\" return false.");
    return false;
  }

  if (this->_auto_save_dataset &&
      this->_ptr_Dataset_Manager->save(this->_path_dataset_dir, true) == false) {
    ERR(L"Function \"save()\" return false.");
    return false;
  }

  if (this->_ptr_trainer_Neural_Network != nullptr) {
    this->_ptr_trainer_Neural_Network->clear_training_arrays();
  }

  this->_require_evaluate_envs = true;

  return true;
}

bool Models::Check_Expiration(void) {
  if (std::chrono::system_clock::now() >= this->_competitor_expiration) {
    // Weights_Initializer.
    // TODO: Genetic algorithm.
    Weights_Initializer tmp_Weights_Initializer;

    // TODO: Hyperparameter weights initializer.
    tmp_Weights_Initializer.type_weights_initializer =
        INITIALIZER::ORTHOGONAL;

    if (tmp_Weights_Initializer.Output_Initialize(
            this->_ptr_trainer_Neural_Network) == false) {
      ERR(L"Function \"Output_Initialize()\" return false.");
      return false;
    }
    // |END| Weights_Initializer. |END|

    this->_ptr_trainer_Neural_Network->set_loss(
        ENV::TRAIN, (std::numeric_limits<real>::max)());
    this->_ptr_trainer_Neural_Network->set_loss(
        ENV::VALID, (std::numeric_limits<real>::max)());
    this->_ptr_trainer_Neural_Network->set_loss(
        ENV::TESTG, (std::numeric_limits<real>::max)());

    this->_ptr_trainer_Neural_Network->set_accu(ENV::TRAIN, 0_r);
    this->_ptr_trainer_Neural_Network->set_accu(ENV::VALID, 0_r);
    this->_ptr_trainer_Neural_Network->set_accu(ENV::TESTG, 0_r);

    this->_ptr_competitor_Neural_Network->set_loss(
        ENV::TRAIN, (std::numeric_limits<real>::max)());
    this->_ptr_competitor_Neural_Network->set_loss(
        ENV::VALID, (std::numeric_limits<real>::max)());
    this->_ptr_competitor_Neural_Network->set_loss(
        ENV::TESTG, (std::numeric_limits<real>::max)());

    this->_ptr_competitor_Neural_Network->set_accu(ENV::TRAIN, 0_r);
    this->_ptr_competitor_Neural_Network->set_accu(ENV::VALID, 0_r);
    this->_ptr_competitor_Neural_Network->set_accu(ENV::TESTG, 0_r);

    this->_competitor_expiration =
        std::chrono::system_clock::now() +
        std::chrono::seconds(this->_expiration_seconds);

    return true;
  }

  return false;
}

bool Models::evaluate_envs(void) {
  if (this->_ptr_Dataset_Manager == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  }

  if (this->_ptr_trainer_Neural_Network != nullptr &&
      this->evaluate_envs(this->_ptr_trainer_Neural_Network) == false) {
    ERR(L"Function \"evaluate_envs()\" return false.");
    return false;
  }

  if (this->_ptr_competitor_Neural_Network != nullptr &&
      this->evaluate_envs(this->_ptr_competitor_Neural_Network) == false) {
    ERR(L"Function \"evaluate_envs()\" return false.");
    return false;
  }

  if (this->_ptr_trained_Neural_Network != nullptr &&
      this->evaluate_envs(this->_ptr_trained_Neural_Network) == false) {
    ERR(L"Function \"evaluate_envs()\" return false.");
    return false;
  }

  this->_require_evaluate_envs = false;

  return true;
}

bool Models::evaluate_envs_pre_train(void) {
  if (this->_ptr_Dataset_Manager == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  }

  if (this->_ptr_trainer_Neural_Network != nullptr &&
      this->evaluate_envs_pre_train(this->_ptr_trainer_Neural_Network) ==
          false) {
    ERR(L"Function \"evaluate_envs_pre_train()\" return false.");
    return false;
  }

  if (this->_ptr_competitor_Neural_Network != nullptr &&
      this->evaluate_envs_pre_train(this->_ptr_competitor_Neural_Network) ==
          false) {
    ERR(L"Function \"evaluate_envs_pre_train()\" return false.");
    return false;
  }

  if (this->_ptr_trained_Neural_Network != nullptr &&
      this->evaluate_envs_pre_train(this->_ptr_trained_Neural_Network) ==
          false) {
    ERR(L"Function \"evaluate_envs_pre_train()\" return false.");
    return false;
  }

  this->_require_evaluate_envs = false;

  return true;
}

bool Models::evaluate_envs(Model *const ptr_neural_network_received) {
  if (this->_ptr_Dataset_Manager == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  } else if (ptr_neural_network_received == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  }

  this->_ptr_Dataset_Manager->evaluate_envs(ptr_neural_network_received);

  return true;
}

bool Models::evaluate_envs_pre_train(
    Model *const ptr_neural_network_received) {
  if (this->_ptr_Dataset_Manager == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  } else if (ptr_neural_network_received == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  }

  if (ptr_neural_network_received->Set__Pre_Training_Level(1_UZ) == false) {
    ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
    return false;
  }

  this->_ptr_Dataset_Manager->evaluate_envs(ptr_neural_network_received);

  if (ptr_neural_network_received->Set__Pre_Training_Level(0_UZ) == false) {
    ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
    return false;
  }

  return true;
}

bool Models::if_require_evaluate_envs(void) {
  if (this->require_evaluate_envs() && this->evaluate_envs() == false) {
    ERR(L"Function \"evaluate_envs()\" return false.");
    return false;
  }

  return true;
}

bool Models::if_require_evaluate_envs_pre_train(void) {
  if (this->require_evaluate_envs()) {
    if (this->evaluate_envs_pre_train() == false) {
      ERR(L"Function \"evaluate_envs_pre_train()\" return false.");
      return false;
    }

    this->_require_evaluate_envs = false;
  }

  return true;
}

bool Models::compare_trained(void) {
  bool const tmp_updated(this->_ptr_trained_Neural_Network->Compare(
      this->_ptr_Dataset_Manager->Use__Metric_Loss(),
      this->_ptr_Dataset_Manager->Get__Dataset_In_Equal_Less_Holdout_Accepted(),
      this->_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(),
      this->_ptr_Dataset_Manager->Get__Minimum_Loss_Holdout_Accepted(),
      this->_ptr_competitor_Neural_Network));

  if (tmp_updated) {
    if (this->_ptr_trained_Neural_Network->Update(
            *this->_ptr_competitor_Neural_Network, true) == false) {
      ERR(L"Function \"Update()\" return false.");
      return false;
    }

    if (this->_optimization_auto_save_trained &&
        this->save_model(HIERARCHY::TRAINED) == false) {
      ERR(L"Function \"save_model()\" return false.");
      return false;
    }

    this->_competitor_expiration =
        std::chrono::system_clock::now() +
        std::chrono::seconds(this->_expiration_seconds);
  }

  return (tmp_updated);
}

bool Models::compare_trained_pre_train(void) {
  // Enable last pre-training mode.
  if (this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level(
          (this->_ptr_competitor_Neural_Network->total_layers - 3_UZ) / 2_UZ +
          1_UZ) == false) {
    ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
    return false;
  } else if (this->_ptr_trained_Neural_Network->Set__Pre_Training_Level(
                 (this->_ptr_trained_Neural_Network->total_layers - 3_UZ) /
                     2_UZ +
                 1_UZ) == false) {
    ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
    return false;
  }

  // Evaluation.
  if (this->evaluate_envs(this->_ptr_competitor_Neural_Network) == false) {
    ERR(L"Function \"evaluate_envs()\" return false.");
    return false;
  } else if (this->evaluate_envs(this->_ptr_trained_Neural_Network) == false) {
    ERR(L"Function \"evaluate_envs()\" return false.");
    return false;
  }

  // Compare.
  bool const tmp_updated(this->_ptr_trained_Neural_Network->Compare(
      this->_ptr_Dataset_Manager->Use__Metric_Loss(),
      this->_ptr_Dataset_Manager->Get__Dataset_In_Equal_Less_Holdout_Accepted(),
      this->_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(),
      this->_ptr_Dataset_Manager->Get__Minimum_Loss_Holdout_Accepted(),
      this->_ptr_competitor_Neural_Network));

  // Disable pre-training mode.
  if (this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level(0_UZ) ==
      false) {
    ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
    return false;
  } else if (this->_ptr_trained_Neural_Network->Set__Pre_Training_Level(0_UZ) ==
             false) {
    ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
    return false;
  }

  if (tmp_updated) {
    if (this->_ptr_trained_Neural_Network->Update(
            *this->_ptr_competitor_Neural_Network, true) == false) {
      ERR(L"Function \"Update()\" return false.");
      return false;
    }

    if (this->_optimization_auto_save_trained &&
        this->save_model(HIERARCHY::TRAINED) == false) {
      ERR(L"Function \"save_model()\" return false.");
      return false;
    }

    this->_competitor_expiration =
        std::chrono::system_clock::now() +
        std::chrono::seconds(this->_expiration_seconds);
  }

  return (tmp_updated);
}

bool Models::pre_training(void) {
  if (this->_ptr_trainer_Neural_Network == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  } else if (this->_ptr_competitor_Neural_Network == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  } else if (this->_ptr_Dataset_Manager == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  } else if (Sys::shutdownblock->preparing_for_shutdown()) {
    INFO(L"Preparing for shutdown.");
    return false;
  }

  this->Check_Expiration();

  std::chrono::system_clock::time_point const tmp_expiration(
      this->_while_cond_opt.expiration);

  // Loop through each pre-training level.
  for (size_t tmp_optimization_time_level,
       tmp_pre_training_end(
           (this->_ptr_trainer_Neural_Network->total_layers - 3_UZ) / 2_UZ +
           2_UZ),
       tmp_pre_training_level(1_UZ);
       tmp_pre_training_level != tmp_pre_training_end &&
       Sys::shutdownblock->preparing_for_shutdown() == false;
       ++tmp_pre_training_level) {
    tmp_optimization_time_level =
        static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                tmp_expiration - std::chrono::system_clock::now())
                .count()) /
        (tmp_pre_training_end - tmp_pre_training_level);
    this->_while_cond_opt.expiration =
        std::chrono::system_clock::now() +
        std::chrono::seconds(tmp_optimization_time_level);

    if (this->_ptr_trainer_Neural_Network->Set__Pre_Training_Level(
            tmp_pre_training_level) == false) {
      ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
      return false;
    } else if (this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level(
                   tmp_pre_training_level) == false) {
      ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
      return false;
    }

    if (this->evaluate_envs(this->_ptr_trainer_Neural_Network) == false) {
      ERR(L"Function \"evaluate_envs()\" return false.");
      return false;
    } else if (this->evaluate_envs(this->_ptr_competitor_Neural_Network) ==
               false) {
      ERR(L"Function \"evaluate_envs()\" return false.");
      return false;
    }

    // If is not the first pre-training level. Train the neural network based on
    // the previously best parameters found.
    if (tmp_pre_training_level != 1_UZ &&
        this->_ptr_trainer_Neural_Network->Update(
            *this->_ptr_competitor_Neural_Network, true) == false) {
      ERR(L"Function \"Update()\" return false.");
      return false;
    }

    // reset training array(s).
    this->_ptr_trainer_Neural_Network->clear_training_arrays();

    // If use hyperparameter optimization. reset...
    this->_ptr_Dataset_Manager->reset();

    this->_ptr_Dataset_Manager->optimize(
        this->_while_cond_opt,
        this->_optimization_auto_save_trainer, false, this->_desired_loss,
        this->get_model_path(HIERARCHY::TRAINER, L"net"),
        this->get_model_path(HIERARCHY::TRAINER, L"nn"), L"", L"",
        this->_ptr_trainer_Neural_Network,
        this->_ptr_competitor_Neural_Network);
  }

  // Disable pre-training mode.
  if (this->_ptr_trainer_Neural_Network->Set__Pre_Training_Level(0_UZ) ==
      false) {
    ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
    return false;
  } else if (this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level(
                 0_UZ) == false) {
    ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
    return false;
  }

  return true;
}

bool Models::pre_training(
    std::vector<size_t> const
        &ref_vector_epochs_per_pre_training_level_received) {
  if (ref_vector_epochs_per_pre_training_level_received.empty()) {
    ERR(L"Vector empty.");
    return false;
  } else if (this->_ptr_trainer_Neural_Network == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  } else if (this->_ptr_competitor_Neural_Network == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  } else if (this->_ptr_Dataset_Manager == nullptr) {
    ERR(L"Invalid pointer.");
    return false;
  } else if (Sys::shutdownblock->preparing_for_shutdown()) {
    INFO(L"Preparing for shutdown.");
    return false;
  }

  this->Check_Expiration();

  WhileCond tmp_while_condition;

  tmp_while_condition.type = WHILE_MODE::ITERATION;

  // Loop through each pre-training level.
  for (size_t tmp_pre_training_end(
           (this->_ptr_trainer_Neural_Network->total_layers - 3_UZ) / 2_UZ +
           2_UZ),
       tmp_pre_training_level(1_UZ);
       tmp_pre_training_level != tmp_pre_training_end &&
       Sys::shutdownblock->preparing_for_shutdown() == false;
       ++tmp_pre_training_level) {
    if (this->_ptr_trainer_Neural_Network->Set__Pre_Training_Level(
            tmp_pre_training_level) == false) {
      ERR(L"Function \"Set__Pre_Training_Level()\" return false.");

      return false;
    } else if (this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level(
                   tmp_pre_training_level) == false) {
      ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
      return false;
    }

    if (this->evaluate_envs(this->_ptr_trainer_Neural_Network) == false) {
      ERR(L"Function \"evaluate_envs()\" return false.");
      return false;
    } else if (this->evaluate_envs(this->_ptr_competitor_Neural_Network) ==
               false) {
      ERR(L"Function \"evaluate_envs()\" return false.");
      return false;
    }

    // reset training array(s).
    this->_ptr_trainer_Neural_Network->clear_training_arrays();

    // If is not the first pre-training level. Train the neural network based on
    // the previously best parameters found.
    if (tmp_pre_training_level != 1_UZ &&
        this->_ptr_trainer_Neural_Network->Update(
            *this->_ptr_competitor_Neural_Network, true) == false) {
      ERR(L"Function \"Update()\" return false.");
      return false;
    }

    tmp_while_condition.maximum_iterations =
        this->_ptr_trainer_Neural_Network->pre_training_level <=
                ref_vector_epochs_per_pre_training_level_received.size()
            ? ref_vector_epochs_per_pre_training_level_received.at(
                  this->_ptr_trainer_Neural_Network->pre_training_level - 1_UZ)
            : ref_vector_epochs_per_pre_training_level_received.at(
                  ref_vector_epochs_per_pre_training_level_received.size() -
                  1_UZ);

    // If use hyperparameter optimization. reset...
    this->_ptr_Dataset_Manager->reset();

    this->_ptr_Dataset_Manager->optimize(
        tmp_while_condition, this->_optimization_auto_save_trainer, false,
        this->_desired_loss,
        this->get_model_path(HIERARCHY::TRAINER, L"net"),
        this->get_model_path(HIERARCHY::TRAINER, L"nn"), L"", L"",
        this->_ptr_trainer_Neural_Network,
        this->_ptr_competitor_Neural_Network);
  }

  // Disable pre-training mode.
  if (this->_ptr_trainer_Neural_Network->Set__Pre_Training_Level(0_UZ) ==
      false) {
    ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
    return false;
  } else if (this->_ptr_competitor_Neural_Network->Set__Pre_Training_Level(
                 0_UZ) == false) {
    ERR(L"Function \"Set__Pre_Training_Level()\" return false.");
    return false;
  }

  return true;
}

double Models::optimize(void) {
  if (this->_ptr_trainer_Neural_Network == nullptr) {
    ERR(L"Invalid pointer.");
    return ((std::numeric_limits<real>::max)());
  } else if (this->_ptr_competitor_Neural_Network == nullptr) {
    ERR(L"Invalid pointer.");
    return ((std::numeric_limits<real>::max)());
  } else if (this->_ptr_Dataset_Manager == nullptr) {
    ERR(L"Invalid pointer.");
    return ((std::numeric_limits<real>::max)());
  } else if (Sys::shutdownblock->preparing_for_shutdown()) {
    INFO(L"Preparing for shutdown.");
    return ((std::numeric_limits<real>::max)());
  }

  this->Check_Expiration();

  // If use hyperparameter optimization. reset...
  this->_ptr_Dataset_Manager->reset();

  this->_ptr_Dataset_Manager->optimize(
      this->_while_cond_opt,
      this->_optimization_auto_save_trainer,
      this->_optimization_auto_save_competitor, this->_desired_loss,
      this->get_model_path(HIERARCHY::TRAINER, L"net"),
      this->get_model_path(HIERARCHY::TRAINER, L"nn"),
      this->get_model_path(HIERARCHY::TRAINED, L"net"),
      this->get_model_path(HIERARCHY::TRAINED, L"nn"),
      this->_ptr_trainer_Neural_Network, this->_ptr_competitor_Neural_Network);

  return (this->_ptr_trainer_Neural_Network->get_loss(ENV::TRAIN));
}

bool Models::load_model(
    HIERARCHY::TYPE const hierarchy,
    size_t const maximum_allowable_host_memory_bytes_received,
    size_t const maximum_allowable_device_memory_bytes_received,
    bool const copy_to_competitor_received)
{
  Model *model;

  switch (hierarchy) {
    case HIERARCHY::TRAINER:
      model = this->_ptr_trainer_Neural_Network;
      break;
    case HIERARCHY::TRAINED:
      model = this->_ptr_trained_Neural_Network;
      break;
    default:
      ERR(L"Undefined `%ls` in the switch.", HIERARCHY_NAME[hierarchy].c_str());
      return false;
  }

  std::wstring const tmp_path_net(
      this->get_model_path(hierarchy, L"net"));

  if (path_exist(tmp_path_net) == false) {
    ERR(L"No such directory/file: `%ls`.", tmp_path_net);
    return false;
  }

  std::wstring const tmp_path_nn(
      this->get_model_path(hierarchy, L"nn"));

  if (path_exist(tmp_path_nn) == false) {
    ERR(L"No such directory/file: `%ls`.", tmp_path_nn);
    return false;
  }

  SAFE_DELETE(model)
  model = new Model;

  if (model->load(
          tmp_path_net, tmp_path_nn,
          maximum_allowable_host_memory_bytes_received) == false) {
    ERR(L"Function \"load()\" return false.");
    SAFE_DELETE(model);
    return false;
  }

  switch (hierarchy) {
    case HIERARCHY::TRAINER:
      this->_ptr_trainer_Neural_Network = model;
      break;
    case HIERARCHY::TRAINED:
      this->_ptr_trained_Neural_Network = model;
      break;
    default:
      ERR(L"Undefined `%ls` in the switch.", HIERARCHY_NAME[hierarchy].c_str());
      SAFE_DELETE(model);
      return false;
  }

  this->_n_inp = model->n_inp;

  this->_n_out = model->n_out;

  this->_seq_w = model->seq_w;

  if (copy_to_competitor_received) {
    SAFE_DELETE(this->_ptr_competitor_Neural_Network);
    this->_ptr_competitor_Neural_Network = new Model;

    if (this->_ptr_competitor_Neural_Network->copy(*model,
                                                   true) == false) {
      ERR(L"Function \"copy()\" return false.");
      return false;
    }
  }

  if (model->set_mp(model->use_mp) ==
      false) {
    ERR(L"Function \"set_mp()\" return false.");
    return false;
  }

  if (model->set_cu(
          model->use_cu,
          maximum_allowable_device_memory_bytes_received) == false) {
    ERR(L"Function \"set_cu()\" return false.");
    return false;
  }

  return true;
}

bool Models::save_model(HIERARCHY::TYPE const hierarchy) {
  std::wstring tmp_path_net, tmp_path_nn;

  if (hierarchy == HIERARCHY::ALL || hierarchy == HIERARCHY::TRAINER) {
    tmp_path_net =
        this->get_model_path(HIERARCHY::TRAINER, L"net");

    if (this->_ptr_trainer_Neural_Network->save_params(
            tmp_path_net.c_str()) == false) {
      ERR(L"Function \"save_params()\" return false.");
      return false;
    }

    tmp_path_nn = this->get_model_path(HIERARCHY::TRAINER, L"nn");

    if (this->_ptr_trainer_Neural_Network->save_spec_params(
            tmp_path_nn.c_str()) == false) {
      ERR(L"Function \"save_spec_params()\" return false.");
      return false;
    }
  }

  if (hierarchy == HIERARCHY::ALL || hierarchy == HIERARCHY::TRAINED) {
    tmp_path_net =
        this->get_model_path(HIERARCHY::TRAINED, L"net");

    if (this->_ptr_trained_Neural_Network->save_params(
            tmp_path_net.c_str()) == false) {
      ERR(L"Function \"save_params()\" return false.");
      return false;
    }

    tmp_path_nn = this->get_model_path(HIERARCHY::TRAINED, L"nn");

    if (this->_ptr_trained_Neural_Network->save_spec_params(
            tmp_path_nn.c_str()) == false) {
      ERR(L"Function \"save_spec_params()\" return false.");
      return false;
    }
  }

  return true;
}

void Models::Deallocate__Neural_Network(HIERARCHY::TYPE const hierarchy) {
  // Trainer.
  if ((hierarchy == HIERARCHY::ALL ||
       hierarchy == HIERARCHY::TRAINER) &&
      this->_ptr_trainer_Neural_Network != nullptr) {
    SAFE_DELETE(this->_ptr_trainer_Neural_Network);
  }

  // Competitor.
  if ((hierarchy == HIERARCHY::ALL ||
       hierarchy == HIERARCHY::TRAINER) &&
      this->_ptr_competitor_Neural_Network != nullptr) {
    SAFE_DELETE(this->_ptr_competitor_Neural_Network);
  }

  // Trained.
  if ((hierarchy == HIERARCHY::ALL ||
       hierarchy == HIERARCHY::TRAINED) &&
      this->_ptr_trained_Neural_Network != nullptr) {
    SAFE_DELETE(this->_ptr_trained_Neural_Network);
  }
}

void Models::Deallocate__Dataset_Manager(void) {
  SAFE_DELETE(this->_ptr_Dataset_Manager);
}

Models::~Models(void) {
  this->Deallocate__Neural_Network(HIERARCHY::ALL);

  this->Deallocate__Dataset_Manager();
}
}  // namespace DL