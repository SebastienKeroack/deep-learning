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
#include "deep-learning/v1/learner/model.hpp"

// Deep learning:
#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/logger.hpp"

// Standard:
#include <fstream>
#include <iostream>

using namespace DL::Str;
using namespace DL::File;

namespace DL::v1 {
bool Model::load_spec_params(std::wstring const &path_name) {
  if (path_exist(path_name) == false) {
    ERR(L"File not found. path_name: \"%ls\".", path_name.c_str());
    return false;
  } else if (recover_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`recover_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  std::wifstream file(CP_STR(path_name), std::ios::in | std::ios::binary);

  if (file.is_open() == false) {
    ERR(L"Failed to open file correctly. path_name: \"%ls\".",
        path_name.c_str());
    return false;
  } else if (file.eof()) {
    ERR(L"Failed to correctly read the file because it is empty. "
        L"path_name: \"%ls\".",
        path_name.c_str());
    return false;
  }

  size_t size_t_input;

  real real_input;

  std::wstring line;

  getline(file, line);  // "|===| GRADIENT DESCENT PARAMETERS |===|"

  if ((file >> line) && line.find(L"learning_rate") == std::wstring::npos) {
    ERR(L"Can not find `learning_rate` inside \"%ls\".", line.c_str());
    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());
    return false;
  } else {
    file >> this->learning_rate >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());
      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"learning_rate_final") == std::wstring::npos) {
    ERR(L"Can not find `learning_rate_final` inside \"%ls\".", line.c_str());
    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());
    return false;
  } else {
    file >> this->learning_rate_final >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());
      return false;
    }
  }

  if ((file >> line) && line.find(L"learning_momentum") == std::wstring::npos) {
    ERR(L"Can not find `learning_momentum` inside \"%ls\".", line.c_str());
    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());
    return false;
  } else {
    file >> this->learning_momentum >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());
      return false;
    }
  }

  if ((file >> line) && line.find(L"learning_gamma") == std::wstring::npos) {
    ERR(L"Can not find `learning_gamma` inside \"%ls\".", line.c_str());
    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());
    return false;
  } else {
    file >> this->learning_gamma >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());
      return false;
    }
  }

  if ((file >> line) && line.find(L"use_nesterov") == std::wstring::npos) {
    ERR(L"Can not find `use_nesterov` inside \"%ls\".", line.c_str());
    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());
    return false;
  } else {
    file >> this->use_nesterov >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());
      return false;
    }
  }

  getline(file, line);  // "|END| GRADIENT DESCENT PARAMETERS |END|"
  getline(file, line);  // CRLF

  getline(file, line);  // "|===| QUICKPROP PARAMETERS |===|"

  if ((file >> line) && line.find(L"quickprop_decay") == std::wstring::npos) {
    ERR(L"Can not find `quickprop_decay` inside \"%ls\".", line.c_str());
    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());
    return false;
  } else {
    file >> this->quickprop_decay >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());
      return false;
    }
  }

  if ((file >> line) && line.find(L"quickprop_mu") == std::wstring::npos) {
    ERR(L"Can not find `quickprop_mu` inside \"%ls\".", line.c_str());
    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());
    return false;
  } else {
    file >> this->quickprop_mu >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());
      return false;
    }
  }

  getline(file, line);  // "|END| QUICKPROP PARAMETERS |END|"
  getline(file, line);  // CRLF

  getline(file, line);  // "|===| RESILLENT PROPAGATION PARAMETERS |===|"

  if ((file >> line) &&
      line.find(L"rprop_increase_factor") == std::wstring::npos) {
    ERR(L"Can not find `rprop_increase_factor` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->rprop_increase_factor >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"rprop_decrease_factor") == std::wstring::npos) {
    ERR(L"Can not find `rprop_decrease_factor` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->rprop_decrease_factor >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"rprop_delta_min") == std::wstring::npos) {
    ERR(L"Can not find `rprop_delta_min` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->rprop_delta_min >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"rprop_delta_max") == std::wstring::npos) {
    ERR(L"Can not find `rprop_delta_max` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->rprop_delta_max >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"rprop_delta_zero") == std::wstring::npos) {
    ERR(L"Can not find `rprop_delta_zero` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->rprop_delta_zero >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  getline(file, line);  // "|END| RESILLENT PROPAGATION PARAMETERS |END|"
  getline(file, line);  // CRLF

  getline(file, line);  // "|===| SARPROP PARAMETERS |===|"

  if ((file >> line) &&
      line.find(L"sarprop_weight_decay_shift") == std::wstring::npos) {
    ERR(L"Can not find `sarprop_weight_decay_shift` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->sarprop_weight_decay_shift >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"sarprop_step_error_threshold_factor") == std::wstring::npos) {
    ERR(L"Can not find `sarprop_step_error_threshold_factor` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->sarprop_step_error_threshold_factor >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"sarprop_step_error_shift") == std::wstring::npos) {
    ERR(L"Can not find `sarprop_step_error_shift` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->sarprop_step_error_shift >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"sarprop_temperature") == std::wstring::npos) {
    ERR(L"Can not find `sarprop_temperature` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->sarprop_temperature >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"sarprop_epoch") == std::wstring::npos) {
    ERR(L"Can not find `sarprop_epoch` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->sarprop_epoch >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  getline(file, line);  // "|END| SARPROP PARAMETERS |END|"
  getline(file, line);  // CRLF

  getline(file, line);  // "|===| ADAM PARAMETERS |===|"

  if ((file >> line) &&
      line.find(L"adam_learning_rate") == std::wstring::npos) {
    ERR(L"Can not find `adam_learning_rate` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->adam_learning_rate >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"adam_beta1") == std::wstring::npos) {
    ERR(L"Can not find `adam_beta1` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->adam_beta1 >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"adam_beta2") == std::wstring::npos) {
    ERR(L"Can not find `adam_beta2` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->adam_beta2 >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"adam_epsilon") == std::wstring::npos) {
    ERR(L"Can not find `adam_epsilon` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->adam_epsilon >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"adam_bias_correction") == std::wstring::npos) {
    ERR(L"Can not find `adam_bias_correction` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->use_adam_bias_correction >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"adam_gamma") == std::wstring::npos) {
    ERR(L"Can not find `adam_gamma` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->adam_gamma >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  getline(file, line);  // "|END| ADAM PARAMETERS |END|"
  getline(file, line);  // CRLF

  getline(file, line);  // "|===| WARM RESTARTS PARAMETERS |===|"

  if ((file >> line) && line.find(L"use_warm_restarts") == std::wstring::npos) {
    ERR(L"Can not find `use_warm_restarts` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->use_warm_restarts >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"warm_restarts_decay_learning_rate") == std::wstring::npos) {
    ERR(L"Can not find `warm_restarts_decay_learning_rate` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->warm_restarts_decay_learning_rate >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"warm_restarts_maximum_learning_rate") == std::wstring::npos) {
    ERR(L"Can not find `warm_restarts_maximum_learning_rate` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->warm_restarts_initial_maximum_learning_rate >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"warm_restarts_minimum_learning_rate") == std::wstring::npos) {
    ERR(L"Can not find `warm_restarts_minimum_learning_rate` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->warm_restarts_minimum_learning_rate >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"warm_restarts_initial_T_i") == std::wstring::npos) {
    ERR(L"Can not find `warm_restarts_initial_T_i` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->warm_restarts_initial_T_i >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"warm_restarts_multiplier") == std::wstring::npos) {
    ERR(L"Can not find `warm_restarts_multiplier` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->warm_restarts_multiplier >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  getline(file, line);  // "|END| WARM RESTARTS PARAMETERS |END|"
  getline(file, line);  // CRLF

  getline(file, line);  // "|===| TRAINING PARAMETERS |===|"

  if ((file >> line) &&
      line.find(L"type_optimizer_function") == std::wstring::npos) {
    ERR(L"Can not find `type_optimizer_function` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    if (size_t_input >= static_cast<size_t>(OPTIMIZER::LENGTH)) {
      ERR(L"Undefined optimization type %zu.", size_t_input);

      return false;
    }

    this->set_optimizer(static_cast<OPTIMIZER::TYPE>(size_t_input));
  }

  if ((file >> line) &&
      line.find(L"type_loss_function") == std::wstring::npos) {
    ERR(L"Can not find `type_loss_function` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    if (size_t_input >= static_cast<size_t>(LOSS_FN::LENGTH)) {
      ERR(L"Undefined loss function type %zu.", size_t_input);

      return false;
    }

    this->set_loss_fn(static_cast<LOSS_FN::TYPE>(size_t_input));
  }

  if ((file >> line) &&
      line.find(L"type_accuracy_function") == std::wstring::npos) {
    ERR(L"Can not find `type_accuracy_function` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    if (size_t_input >= static_cast<size_t>(ACCU_FN::LENGTH)) {
      ERR(L"Undefined loss function type %zu.", size_t_input);

      return false;
    }

    this->set_accu_fn(static_cast<ACCU_FN::TYPE>(size_t_input));
  }

  if ((file >> line) && line.find(L"bit_fail_limit") == std::wstring::npos) {
    ERR(L"Can not find `bit_fail_limit` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> real_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    this->set_bit_fail_limit(real_input);
  }

  if ((file >> line) &&
      line.find(L"pre_training_level") == std::wstring::npos) {
    ERR(L"Can not find `pre_training_level` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->pre_training_level >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"use_clip_gradient") == std::wstring::npos) {
    ERR(L"Can not find `use_clip_gradient` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->use_clip_gradient >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"clip_gradient") == std::wstring::npos) {
    ERR(L"Can not find `clip_gradient` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->clip_gradient >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  getline(file, line);  // "|END| TRAINING PARAMETERS |END|"
  getline(file, line);  // CRLF

  getline(file, line);  // "|===| REGULARIZATION PARAMETERS |===|"

  if ((file >> line) && line.find(L"regularization__max_norm_constraints") ==
                            std::wstring::npos) {
    ERR(L"Can not find `regularization__max_norm_constraints` inside "
        L"`%ls`.",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> real_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    this->Set__Regularization__Max_Norm_Constraints(real_input);
  }

  if ((file >> line) &&
      line.find(L"regularization__l1") == std::wstring::npos) {
    ERR(L"Can not find `regularization__l1` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> real_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    this->set_l1(real_input);
  }

  if ((file >> line) &&
      line.find(L"regularization__l2") == std::wstring::npos) {
    ERR(L"Can not find `regularization__l2` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> real_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    this->set_l2(real_input);
  }

  if ((file >> line) &&
      line.find(L"regularization__srip") == std::wstring::npos) {
    ERR(L"Can not find `regularization__srip` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> real_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    this->set_srip(real_input);
  }

  if ((file >> line) && line.find(L"weight_decay") == std::wstring::npos) {
    ERR(L"Can not find `weight_decay` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> real_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    this->set_weight_decay(real_input);
  }

  if ((file >> line) &&
      line.find(L"use_normalized_weight_decay") == std::wstring::npos) {
    ERR(L"Can not find `use_normalized_weight_decay` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->use_normalized_weight_decay >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  getline(file, line);  // "|END| REGULARIZATION PARAMETERS |END|"
  getline(file, line);  // CRLF

  getline(file, line);  // "|===| NORMALIZATION PARAMETERS |===|"

  if ((file >> line) &&
      line.find(L"normalization_momentum_average") == std::wstring::npos) {
    ERR(L"Can not find `normalization_momentum_average` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> real_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    this->Set__Normalization_Momentum_Average(real_input);
  }

  if ((file >> line) &&
      line.find(L"normalization_epsilon") == std::wstring::npos) {
    ERR(L"Can not find `normalization_epsilon` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> real_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    this->Set__Normalization_Epsilon(real_input);
  }

  if ((file >> line) &&
      line.find(L"batch_renormalization_r_correction_maximum") ==
          std::wstring::npos) {
    ERR(L"Can not find `batch_renormalization_r_correction_maximum` inside "
        L"`%ls`.",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> real_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    this->Set__Batch_Renormalization_r_Correction_Maximum(real_input);
  }

  if ((file >> line) &&
      line.find(L"batch_renormalization_d_correction_maximum") ==
          std::wstring::npos) {
    ERR(L"Can not find `batch_renormalization_d_correction_maximum` inside "
        L"`%ls`.",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> real_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    this->Set__Batch_Renormalization_d_Correction_Maximum(real_input);
  }

  getline(file, line);  // "|END| NORMALIZATION PARAMETERS |END|"
  getline(file, line);  // CRLF

  getline(file, line);  // "|===| LOSS PARAMETERS |===|"

  if ((file >> line) && line.find(L"loss_train") == std::wstring::npos) {
    ERR(L"Can not find `loss_train` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->loss_train >> std::ws;

    if (file.fail()) {
      file.clear();

      // Inf.
      file >> line >> std::ws;

      this->loss_train = (std::numeric_limits<real>::max)();
    }
  }

  if ((file >> line) && line.find(L"loss_valid") == std::wstring::npos) {
    ERR(L"Can not find `loss_valid` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->loss_valid >> std::ws;

    if (file.fail()) {
      file.clear();

      // Inf.
      file >> line >> std::ws;

      this->loss_valid = (std::numeric_limits<real>::max)();
    }
  }

  if ((file >> line) && line.find(L"loss_testg") == std::wstring::npos) {
    ERR(L"Can not find `loss_testg` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->loss_testg >> std::ws;

    if (file.fail()) {
      file.clear();

      // Inf.
      file >> line >> std::ws;

      this->loss_testg = (std::numeric_limits<real>::max)();
    }
  }

  getline(file, line);  // "|END| LOSS PARAMETERS |END|"
  getline(file, line);  // CRLF

  getline(file, line);  // "|===| ACCURANCY PARAMETERS |===|"

  if ((file >> line) && line.find(L"acc_var") == std::wstring::npos) {
    ERR(L"Can not find `acc_var` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> real_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    this->Set__Accurancy_Variance(real_input);
  }

  if ((file >> line) && line.find(L"acc_train") == std::wstring::npos) {
    ERR(L"Can not find `acc_train` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->acc_train >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"acc_valid") == std::wstring::npos) {
    ERR(L"Can not find `acc_valid` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->acc_valid >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"acc_testg") == std::wstring::npos) {
    ERR(L"Can not find `acc_testg` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->acc_testg >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  getline(file, line);  // "|END| ACCURANCY PARAMETERS |END|"
  getline(file, line);  // CRLF

  getline(file, line);  // "|===| COMPUTATION PARAMETERS |===|"

  if ((file >> line) && line.find(L"use_cu") == std::wstring::npos) {
    ERR(L"Can not find `use_cu` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->use_cu >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"use_mp") == std::wstring::npos) {
    ERR(L"Can not find `use_mp` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->use_mp >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"pct_threads") == std::wstring::npos) {
    ERR(L"Can not find `pct_threads` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->pct_threads >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"maximum_batch_size") == std::wstring::npos) {
    ERR(L"Can not find `maximum_batch_size` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->maximum_batch_size >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  getline(file, line);  // "|END| COMPUTATION PARAMETERS |END|"

  if (file.fail()) {
    ERR(L"Logical error on i/o operation `%ls`.", path_name.c_str());

    return false;
  }

  file.close();

  return true;
}

bool Model::load(std::wstring const &path_params,
                 std::wstring const &path_spec_params,
                 size_t const allowable_memory) {
  if (path_exist(path_spec_params) == false) {
    ERR(L"File not found. path_name: \"%ls\".", path_spec_params.c_str());
    return false;
  } else if (path_exist(path_params) == false) {
    ERR(L"File not found. path_name: \"%ls\".", path_params.c_str());
    return false;
  } else if (recover_temp_file(path_params) == false) {
    ERR(L"An error has been triggered from the "
        L"`recover_temp_file(%ls)` function.",
        path_params.c_str());
    return false;
  }

  std::wifstream file(CP_STR(path_params), std::ios::in | std::ios::binary);

  if (file.is_open() == false) {
    ERR(L"Failed to open file correctly. path_name: \"%ls\".",
        path_params.c_str());
    return false;
  } else if (file.eof()) {
    ERR(L"Failed to correctly read the file because it is empty. "
        L"path_name: \"%ls\".",
        path_params.c_str());
    return false;
  }

  this->clear();

  bool tmp_input_boolean;

  size_t tmp_state_layer_index(0_UZ), size_t_input;

  real tmp_input_T[2] = {0};

  std::wstring line;

  auto load_dropout_params_fn([self = this, &file = file](
                                  Layer *const layer_it,
                                  bool const is_hidden_layer_received =
                                      true) -> bool {
    size_t size_t_input;

    real tmp_dropout_values[3] = {0};

    LAYER_DROPOUT::TYPE tmp_type_layer_dropout;

    std::wstring line;

    if ((file >> line) && line.find(L"type_dropout") == std::wstring::npos) {
      ERR(L"Can not find `type_dropout` inside \"%ls\".", line.c_str());

      return false;
    } else if (file.fail()) {
      ERR(L"Can not read properly inside \"%ls\".", line.c_str());

      return false;
    } else {
      file >> size_t_input;

      if (file.fail()) {
        ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

        return false;
      }

      if (size_t_input >= static_cast<size_t>(LAYER_DROPOUT::LENGTH)) {
        ERR(L"Undefined layer dropout type %zu.", size_t_input);

        return false;
      }

      tmp_type_layer_dropout = static_cast<LAYER_DROPOUT::TYPE>(size_t_input);
    }

    if (is_hidden_layer_received) {
      if ((file >> line) &&
          line.find(L"use_coded_dropout") == std::wstring::npos) {
        ERR(L"Can not find `use_coded_dropout` inside \"%ls\".", line.c_str());

        return false;
      } else if (file.fail()) {
        ERR(L"Can not read properly inside \"%ls\".", line.c_str());

        return false;
      } else {
        file >> layer_it->use_coded_dropout >> std::ws;

        if (file.fail()) {
          ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

          return false;
        }
      }
    }

    if ((file >> line) &&
        line.find(L"dropout_values[0]") == std::wstring::npos) {
      ERR(L"Can not find `dropout_values[0]` inside \"%ls\".", line.c_str());

      return false;
    } else if (file.fail()) {
      ERR(L"Can not read properly inside \"%ls\".", line.c_str());

      return false;
    } else {
      file >> tmp_dropout_values[0] >> std::ws;

      if (file.fail()) {
        ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

        return false;
      }
    }

    if ((file >> line) &&
        line.find(L"dropout_values[1]") == std::wstring::npos) {
      ERR(L"Can not find `dropout_values[1]` inside \"%ls\".", line.c_str());

      return false;
    } else if (file.fail()) {
      ERR(L"Can not read properly inside \"%ls\".", line.c_str());

      return false;
    } else {
      file >> tmp_dropout_values[1] >> std::ws;

      if (file.fail()) {
        ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

        return false;
      }
    }

    if ((file >> line) &&
        line.find(L"dropout_values[2]") == std::wstring::npos) {
      ERR(L"Can not find `dropout_values[2]` inside \"%ls\".", line.c_str());

      return false;
    } else if (file.fail()) {
      ERR(L"Can not read properly inside \"%ls\".", line.c_str());

      return false;
    } else {
      file >> tmp_dropout_values[2] >> std::ws;

      if (file.fail()) {
        ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

        return false;
      }
    }

    if (self->type == MODEL::AUTOENCODER &&
        (layer_it ==
             self->ptr_last_layer - (self->total_layers - 3_UZ) / 2_UZ + 2_UZ ||
         layer_it >= self->ptr_last_layer - (self->total_layers - 3_UZ) / 2_UZ +
                         1_UZ)) {
      return true;
    }

    if (self->set_dropout(layer_it, tmp_type_layer_dropout, tmp_dropout_values,
                          false) == false) {
      ERR(L"An error has been triggered from the "
          L"`set_dropout(ptr, %u, %f, %f)` function.",
          tmp_type_layer_dropout, tmp_dropout_values[0], tmp_dropout_values[1]);
      return false;
    }

    return true;
  });

  auto tmp_Valid__Layer__Normalization(
      [self = this](Layer *const layer_it) -> bool {
        if (self->type == MODEL::AUTOENCODER &&
            layer_it >= self->ptr_last_layer -
                            (self->total_layers - 3_UZ) / 2_UZ + 1_UZ) {
          return false;
        }

        return true;
      });

  INFO(L"");
  INFO(L"Load params `%ls`.", path_params.c_str());

  getline(file, line);  // "|===| DIMENSION |===|"

  if ((file >> line) && line.find(L"type") == std::wstring::npos) {
    ERR(L"Can not find `type` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    if (size_t_input >= static_cast<size_t>(MODEL::LENGTH)) {
      ERR(L"Undefined network type %zu.", size_t_input);

      return false;
    }

    this->type = static_cast<MODEL::TYPE>(size_t_input);
  }

  if ((file >> line) && line.find(L"number_layers") == std::wstring::npos) {
    ERR(L"Can not find `number_layers` inside \"%ls\".", line.c_str());
    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());
    return false;
  } else {
    file >> size_t_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());
      return false;
    } else if (size_t_input < 2) {
      ERR(L"The number of layers is set too small.");
      return false;
    }
  }

  // allocate structure.
  INFO(L"allocate %zu layer(s).", size_t_input);
  if (this->Allocate__Structure(size_t_input, allowable_memory) == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Structure(%zu, %zu)` function.",
        size_t_input, allowable_memory);
    return false;
  }

  if ((file >> line) && line.find(L"seq_w") == std::wstring::npos) {
    ERR(L"Can not find `seq_w` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->seq_w >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"n_time_delay") == std::wstring::npos) {
    ERR(L"Can not find `n_time_delay` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->n_time_delay >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"use_first_layer_as_input") == std::wstring::npos) {
    ERR(L"Can not find `use_first_layer_as_input` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> tmp_input_boolean >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    if (this->Set__Input_Mode(tmp_input_boolean) == false) {
      ERR(L"An error has been triggered from the "
          L"`Set__Input_Mode(%ls)` function.",
          to_wstring(tmp_input_boolean).c_str());
      return false;
    }
  }

  if ((file >> line) &&
      line.find(L"use_last_layer_as_output") == std::wstring::npos) {
    ERR(L"Can not find `use_last_layer_as_output` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> tmp_input_boolean >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    if (this->Set__Output_Mode(tmp_input_boolean) == false) {
      ERR(L"An error has been triggered from the "
          L"`Set__Output_Mode(%ls)` function.",
          to_wstring(tmp_input_boolean).c_str());
      return false;
    }
  }

  Layer const *const tmp_ptr_first_layer(this->ptr_array_layers),
      *const last_layer(this->ptr_last_layer - 1),  // Subtract output layer.
      *tmp_ptr_previous_layer, *tmp_ptr_layer_state(nullptr);
  Layer *layer_it(this->ptr_array_layers);
  // |END| allocate structure. |END|

  // allocate basic unit(s).
  if ((file >> line) && line.find(L"total_basic_units") == std::wstring::npos) {
    ERR(L"Can not find `total_basic_units` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->total_basic_units >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  layer_it->ptr_array_basic_units = nullptr;
  layer_it->ptr_last_basic_unit =
      layer_it->ptr_array_basic_units + this->total_basic_units;

  INFO(L"allocate %zu basic unit(s).", this->total_basic_units);
  if (this->Allocate__Basic_Units() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Basic_Units()` function.");
    return false;
  }
  // |END| allocate basic unit(s). |END|

  // allocate basic indice unit(s).
  if ((file >> line) &&
      line.find(L"total_basic_indice_units") == std::wstring::npos) {
    ERR(L"Can not find `total_basic_indice_units` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->total_basic_indice_units >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  layer_it->ptr_array_basic_indice_units = nullptr;
  layer_it->ptr_last_basic_indice_unit =
      layer_it->ptr_array_basic_indice_units + this->total_basic_indice_units;

  INFO(L"allocate %zu basic indice unit(s).", this->total_basic_indice_units);
  if (this->Allocate__Basic_Indice_Units() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Basic_Indice_Units()` function.");
    return false;
  }
  // |END| allocate basic indice unit(s). |END|

  // allocate neuron unit(s).
  if ((file >> line) &&
      line.find(L"total_neuron_units") == std::wstring::npos) {
    ERR(L"Can not find `total_neuron_units` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->total_neuron_units >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  layer_it->ptr_array_neuron_units = nullptr;
  layer_it->ptr_last_neuron_unit =
      layer_it->ptr_array_neuron_units + this->total_neuron_units;

  INFO(L"allocate %zu neuron unit(s).", this->total_neuron_units);
  if (this->Allocate__Neuron_Units() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Neuron_Units()` function.");
    return false;
  }
  // |END| allocate neuron unit(s). |END|

  // allocate AF unit(s).
  if ((file >> line) && line.find(L"total_AF_units") == std::wstring::npos) {
    ERR(L"Can not find `total_AF_units` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->total_AF_units >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  layer_it->ptr_array_AF_units = nullptr;
  layer_it->ptr_last_AF_unit =
      layer_it->ptr_array_AF_units + this->total_AF_units;

  INFO(L"allocate %zu AF unit(s).", this->total_AF_units);
  if (this->Allocate__AF_Units() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__AF_Units()` function.");
    return false;
  }
  // |END| allocate AF unit(s). |END|

  // allocate af_ind unit(s).
  if ((file >> line) &&
      line.find(L"total_AF_Ind_recurrent_units") == std::wstring::npos) {
    ERR(L"Can not find `total_AF_Ind_recurrent_units` inside \"%ls\".",
        line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->total_AF_Ind_recurrent_units >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  layer_it->ptr_array_AF_Ind_recurrent_units = nullptr;
  layer_it->ptr_last_AF_Ind_recurrent_unit =
      layer_it->ptr_array_AF_Ind_recurrent_units +
      this->total_AF_Ind_recurrent_units;

  INFO(L"allocate %zu AF Ind recurrent unit(s).",
       this->total_AF_Ind_recurrent_units);
  if (this->Allocate__AF_Ind_Recurrent_Units() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__AF_Ind_Recurrent_Units()` function.");
    return false;
  }
  // |END| allocate af_ind unit(s). |END|

  // allocate block/cell unit(s).
  if ((file >> line) && line.find(L"total_block_units") == std::wstring::npos) {
    ERR(L"Can not find `total_block_units` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->total_block_units >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  layer_it->ptr_array_block_units = nullptr;
  layer_it->ptr_last_block_unit =
      layer_it->ptr_array_block_units + this->total_block_units;

  if ((file >> line) && line.find(L"total_cell_units") == std::wstring::npos) {
    ERR(L"Can not find `total_cell_units` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->total_cell_units >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  layer_it->ptr_array_cell_units = nullptr;
  layer_it->ptr_last_cell_unit =
      layer_it->ptr_array_cell_units + this->total_cell_units;

  if (this->total_block_units != 0_UZ && this->total_cell_units != 0_UZ) {
    INFO(L"allocate %zu block unit(s).", this->total_block_units);
    INFO(L"allocate %zu cell unit(s).", this->total_cell_units);
    if (this->Allocate__LSTM_Layers() == false) {
      ERR(L"An error has been triggered from the "
          L"`Allocate__LSTM_Layers()` function.");
      return false;
    }
  }
  // |END| allocate block/cell unit(s). |END|

  // allocate normalized unit(s).
  if ((file >> line) &&
      line.find(L"total_normalized_units") == std::wstring::npos) {
    ERR(L"Can not find `total_normalized_units` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->total_normalized_units >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  INFO(L"allocate %zu normalized unit(s).", this->total_normalized_units);
  if (this->Allocate__Normalized_Unit(false) == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Normalized_Unit(false)` function.");
    return false;
  }
  // |END| allocate normalized unit(s). |END|

  // allocate parameter(s).
  if ((file >> line) && line.find(L"total_parameters") == std::wstring::npos) {
    ERR(L"Can not find `total_parameters` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->total_parameters >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"total_weights") == std::wstring::npos) {
    ERR(L"Can not find `total_weights` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->total_weights >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  if ((file >> line) && line.find(L"total_bias") == std::wstring::npos) {
    ERR(L"Can not find `total_bias` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> this->total_bias >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  INFO(L"allocate %zu parameter(s).", this->total_parameters);
  if (this->Allocate__Parameter() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Parameter()` function.");
    return false;
  }
  // |END| allocate parameter(s). |END|

  // Initialize layer(s).
  // reset number of weights to zero. Increment the variable inside the loading
  // layer.
  this->total_weights = 0_UZ;
  this->total_bias = 0_UZ;

  Basic_unit *tmp_ptr_array_basic_units(this->ptr_array_basic_units);

  Basic_indice_unit *tmp_ptr_array_basic_indice_units(
      this->ptr_array_basic_indice_units);

  Neuron_unit *tmp_ptr_array_neuron_units(this->ptr_array_neuron_units);

  AF_unit *tmp_ptr_array_AF_units(this->ptr_array_AF_units);

  AF_Ind_recurrent_unit *tmp_ptr_array_AF_Ind_recurrent_units(
      this->ptr_array_AF_Ind_recurrent_units);

  BlockUnit *tmp_ptr_array_block_units(this->ptr_array_block_units);

  CellUnit *tmp_ptr_array_cell_units(this->ptr_array_cell_units);

  union Normalized_unit *tmp_ptr_array_normalized_units(
      this->ptr_array_normalized_units);

  // Input layer.
  //  Type layer.
  INFO(L"load input layer.");
  getline(file, line);  // "Input layer:"

  if ((file >> line) && line.find(L"type_layer") == std::wstring::npos) {
    ERR(L"Can not find `type_layer` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    if (size_t_input >= static_cast<size_t>(LAYER::LENGTH)) {
      ERR(L"Undefined layer type %zu.", size_t_input);

      return false;
    }

    layer_it->type_layer = static_cast<LAYER::TYPE>(size_t_input);
  }
  //  |END| Type layer. |END|

  //  Type activation.
  if ((file >> line) && line.find(L"type_activation") == std::wstring::npos) {
    ERR(L"Can not find `type_activation` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    if (size_t_input >= static_cast<size_t>(LAYER_ACTIVATION::LENGTH)) {
      ERR(L"Undefined layer activation type %zu.", size_t_input);

      return false;
    }

    layer_it->type_activation =
        static_cast<LAYER_ACTIVATION::TYPE>(size_t_input);
  }
  //  |END| Type activation. |END|

  //  Dropout.
  if (load_dropout_params_fn(layer_it, false) == false) {
    ERR(L"An error has been triggered from the "
        L"`load_dropout_params_fn(false)` function.");
    return false;
  }
  //  |END| Dropout. |END|

  //  Initialize input(s).
  if ((file >> line) && line.find(L"n_inp") == std::wstring::npos) {
    ERR(L"Can not find `n_inp` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    *layer_it->ptr_number_outputs = this->n_inp = size_t_input;

    layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
    tmp_ptr_array_neuron_units += size_t_input;
    layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;

    this->Order__Layer__Neuron(layer_it);
  }
  //  |END| Initialize input(s). |END|

  //  Initialize normalized unit(s).
  layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
  if (this->total_normalized_units_allocated != 0_UZ) {
    tmp_ptr_array_normalized_units += *layer_it->ptr_number_outputs;
  }  // If use normalization.
  layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
  //  |END| Initialize normalized unit(s). |END|

  // Initialize AF unit(s).
  layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
  tmp_ptr_array_AF_units += *layer_it->ptr_number_outputs;
  layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
  // |END| Initialize AF unit(s). |END|

  // Initialize AF Ind recurrent unit(s).
  layer_it->ptr_array_AF_Ind_recurrent_units =
      tmp_ptr_array_AF_Ind_recurrent_units;
  layer_it->ptr_last_AF_Ind_recurrent_unit =
      tmp_ptr_array_AF_Ind_recurrent_units;
  // |END| Initialize AF Ind recurrent unit(s). |END|

  //  Initialize basic unit(s).
  layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
  layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
  //  |END| Initialize basic unit(s). |END|

  //  Initialize basic indice unit(s).
  layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
  layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
  //  |END| Initialize basic indice unit(s). |END|

  //  Initialize block/cell unit(s).
  layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
  layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

  layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
  layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
  //  |END| Initialize block/cell unit(s). |END|
  // |END| Input layer. |END|

  // Hidden layer.
  for (++layer_it; layer_it != last_layer; ++layer_it) {
    // Type layer.
    getline(file, line);  // "Hidden layer %u:"

    if ((file >> line) && line.find(L"type_layer") == std::wstring::npos) {
      ERR(L"Can not find `type_layer` inside \"%ls\".", line.c_str());

      return false;
    } else if (file.fail()) {
      ERR(L"Can not read properly inside \"%ls\".", line.c_str());

      return false;
    } else {
      file >> size_t_input;

      if (file.fail()) {
        ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

        return false;
      }

      if (size_t_input >= static_cast<size_t>(LAYER::LENGTH)) {
        ERR(L"Undefined layer type %zu.", size_t_input);

        return false;
      }

      layer_it->type_layer = static_cast<LAYER::TYPE>(size_t_input);
    }

    INFO(L"load hidden layer %zu (%ls | %u).",
         static_cast<size_t>(layer_it - tmp_ptr_first_layer),
         LAYER_NAME[layer_it->type_layer].c_str(), layer_it->type_layer);
    // |END| Type layer. |END|

    this->Organize__Previous_Layers_Connected(tmp_state_layer_index, layer_it,
                                              tmp_ptr_layer_state);

    tmp_ptr_previous_layer = layer_it->previous_connected_layers[0];

    // Use bidirectional.
    if ((file >> line) &&
        line.find(L"use_bidirectional") == std::wstring::npos) {
      ERR(L"Can not find `use_bidirectional` inside \"%ls\".", line.c_str());

      return false;
    } else if (file.fail()) {
      ERR(L"Can not read properly inside \"%ls\".", line.c_str());

      return false;
    } else {
      file >> layer_it->use_bidirectional;

      if (file.fail()) {
        ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

        return false;
      }
    }
    // |END| Use bidirectional. |END|

    switch (layer_it->type_layer) {
      case LAYER::AVERAGE_POOLING:
        // Pooling.
        if ((file >> line) && line.find(L"kernel_size") == std::wstring::npos) {
          ERR(L"Can not find `kernel_size` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->pooling_values[0] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) && line.find(L"stride") == std::wstring::npos) {
          ERR(L"Can not find `stride` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->pooling_values[1] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) && line.find(L"padding") == std::wstring::npos) {
          ERR(L"Can not find `padding` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->pooling_values[2] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) && line.find(L"dilation") == std::wstring::npos) {
          ERR(L"Can not find `dilation` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->pooling_values[3] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }
        // |END| Pooling. |END|

        //  Initialize normalized unit(s).
        layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
        layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
        //  |END| Initialize normalized unit(s). |END|

        // Initialize basic unit(s).
        if ((file >> line) &&
            line.find(L"number_basic_units") == std::wstring::npos) {
          ERR(L"Can not find `number_basic_units` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> *layer_it->ptr_number_outputs >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
          tmp_ptr_array_basic_units += *layer_it->ptr_number_outputs;
          layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;

          this->Order__Layer__Basic(layer_it);
        }
        // |END| Initialize basic unit(s). |END|

        //  Initialize basic indice unit(s).
        layer_it->ptr_array_basic_indice_units =
            tmp_ptr_array_basic_indice_units;
        layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
        //  |END| Initialize basic indice unit(s). |END|

        // Initialize neuron unit(s).
        layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
        layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
        // |END| Initialize neuron unit(s). |END|

        // Initialize AF unit(s).
        layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
        layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
        // |END| Initialize AF unit(s). |END|

        // Initialize AF Ind recurrent unit(s).
        layer_it->ptr_array_AF_Ind_recurrent_units =
            tmp_ptr_array_AF_Ind_recurrent_units;
        layer_it->ptr_last_AF_Ind_recurrent_unit =
            tmp_ptr_array_AF_Ind_recurrent_units;
        // |END| Initialize AF Ind recurrent unit(s). |END|

        //  Initialize block/cell unit(s).
        layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
        layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

        layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
        layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
        //  |END| Initialize block/cell unit(s). |END|
        break;
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_RECURRENT:
        // Type activation.
        if ((file >> line) &&
            line.find(L"type_activation") == std::wstring::npos) {
          ERR(L"Can not find `type_activation` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (size_t_input >= static_cast<size_t>(LAYER_ACTIVATION::LENGTH)) {
            ERR(L"Undefined layer activation type %zu.", size_t_input);

            return false;
          }

          layer_it->type_activation =
              static_cast<LAYER_ACTIVATION::TYPE>(size_t_input);
        }
        // |END| Type activation. |END|

        // Dropout.
        if (load_dropout_params_fn(layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`load_dropout_params_fn()` function.");
          return false;
        }
        // |END| Dropout. |END|

        // Normalization.
        if ((file >> line) &&
            line.find(L"type_normalization") == std::wstring::npos) {
          ERR(L"Can not find `type_normalization` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (size_t_input >= static_cast<size_t>(LAYER_NORM::LENGTH)) {
            ERR(L"Undefined layer normalization type %zu.", size_t_input);

            return false;
          }

          if (tmp_Valid__Layer__Normalization(layer_it) &&
              this->Set__Layer_Normalization(
                  layer_it, static_cast<LAYER_NORM::TYPE>(size_t_input), false,
                  false) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Layer_Normalization(%zu)` function.",
                size_t_input);
            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"use_layer_normalization_before_activation") ==
                std::wstring::npos) {
          ERR(L"Can not find `use_layer_normalization_before_activation` "
              L"inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->use_layer_normalization_before_activation >>
              std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"number_normalized_units") == std::wstring::npos) {
          ERR(L"Can not find `number_normalized_units` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
          tmp_ptr_array_normalized_units += size_t_input;
          layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;

          this->Order__Layer__Normalization(layer_it);

          if (this->Load_Dimension__Normalization(layer_it, file) == false) {
            ERR(L"An error has been triggered from the "
                L"`Load_Dimension__Normalization()` function.");
            return false;
          }
        }
        // |END| Normalization. |END|

        if ((file >> line) &&
            line.find(L"use_tied_parameter") == std::wstring::npos) {
          ERR(L"Can not find `use_tied_parameter` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_boolean >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__Tied_Parameter(layer_it, tmp_input_boolean, false) ==
              false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Tied_Parameter(ptr, %ls, false)` function.",
                to_wstring(tmp_input_boolean).c_str());
            return false;
          }
        }

        // k-Sparse filters.
        if ((file >> line) && line.find(L"k_sparsity") == std::wstring::npos) {
          ERR(L"Can not find `k_sparsity` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__K_Sparsity(layer_it, size_t_input) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__K_Sparsity(ptr, %zu)` function.",
                size_t_input);
            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"alpha_sparsity") == std::wstring::npos) {
          ERR(L"Can not find `alpha_sparsity` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_T[0] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__Alpha_Sparsity(layer_it, tmp_input_T[0]) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Alpha_Sparsity(ptr, %f)` function.",
                tmp_input_T[0]);
            return false;
          }
        }
        // |END| k-Sparse filters. |END|

        // Constraint.
        if ((file >> line) &&
            line.find(L"constraint_recurrent_weight_lower_bound") ==
                std::wstring::npos) {
          ERR(L"Can not find `constraint_recurrent_weight_lower_bound` "
              L"inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_T[0] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"constraint_recurrent_weight_upper_bound") ==
                std::wstring::npos) {
          ERR(L"Can not find `constraint_recurrent_weight_upper_bound` "
              L"inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_T[1] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__Regularization__Constraint_Recurrent_Weight(
                  layer_it, tmp_input_T[0], tmp_input_T[1]) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Regularization__Constraint_Recurrent_Weight"
                L"(ptr, %f, %f)` function.",
                tmp_input_T[0], tmp_input_T[1]);
            return false;
          }
        }
        // |END| Constraint. |END|

        // Initialize basic unit(s).
        layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
        layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
        // |END| Initialize basic unit(s). |END|

        // Initialize basic indice unit(s).
        layer_it->ptr_array_basic_indice_units =
            tmp_ptr_array_basic_indice_units;
        layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
        // |END| Initialize basic indice unit(s). |END|

        // Initialize neuron unit(s).
        if ((file >> line) &&
            line.find(L"number_neuron_units") == std::wstring::npos) {
          ERR(L"Can not find `number_neuron_units` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> *layer_it->ptr_number_outputs >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
          tmp_ptr_array_neuron_units += *layer_it->ptr_number_outputs;
          layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;

          this->Order__Layer__Neuron(layer_it);

          switch (tmp_ptr_previous_layer->type_layer) {
            case LAYER::AVERAGE_POOLING:
            case LAYER::RESIDUAL:
              if (this->Load_Dimension__FC<Basic_unit, LAYER::AVERAGE_POOLING>(
                      layer_it, this->ptr_array_basic_units, file) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__FC()` function.");
                return false;
              }
              break;
            case LAYER::FULLY_CONNECTED:
            case LAYER::FULLY_CONNECTED_RECURRENT:
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
              if (this->Load_Dimension__FC<Neuron_unit, LAYER::FULLY_CONNECTED>(
                      layer_it, this->ptr_array_neuron_units, file) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__FC()` function.");
                return false;
              }
              break;
            case LAYER::LSTM:
              if (this->Load_Dimension__FC<CellUnit, LAYER::LSTM>(
                      layer_it, this->ptr_array_cell_units, file) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__FC()` function.");
                return false;
              }
              break;
            case LAYER::MAX_POOLING:
              if (this->Load_Dimension__FC<Basic_indice_unit,
                                           LAYER::MAX_POOLING>(
                      layer_it, this->ptr_array_basic_indice_units, file) ==
                  false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__FC()` function.");
                return false;
              }
              break;
            default:
              ERR(L"Layer type (%d | %ls) is not managed in the switch.",
                  tmp_ptr_previous_layer->type_layer,
                  LAYER_NAME[tmp_ptr_previous_layer->type_layer].c_str());
              return false;
          }
        }
        // |END| Initialize neuron unit(s). |END|

        // Initialize AF unit(s).
        if ((file >> line) &&
            line.find(L"number_AF_units") == std::wstring::npos) {
          ERR(L"Can not find `number_AF_units` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
          tmp_ptr_array_AF_units += size_t_input;
          layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;

          this->Order__Layer__AF(layer_it);

          if (this->Load_Dimension__AF(layer_it, file) == false) {
            ERR(L"An error has been triggered from the "
                L"`Load_Dimension__AF()` function.");
            return false;
          }
        }
        // |END| Initialize AF unit(s). |END|

        // Initialize AF Ind recurrent unit(s).
        layer_it->ptr_array_AF_Ind_recurrent_units =
            tmp_ptr_array_AF_Ind_recurrent_units;
        layer_it->ptr_last_AF_Ind_recurrent_unit =
            tmp_ptr_array_AF_Ind_recurrent_units;
        // |END| Initialize AF Ind recurrent unit(s). |END|

        //  Initialize block/cell unit(s).
        layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
        layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

        layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
        layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
        //  |END| Initialize block/cell unit(s). |END|

        // Initialize bias parameter(s).
        if (this->Load_Dimension__Bias(layer_it, file) == false) {
          ERR(L"An error has been triggered from the "
              L"`Load_Dimension__Bias()` function.");
          return false;
        }
        // |END| Initialize bias parameter(s). |END|
        break;
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        // Type activation.
        if ((file >> line) &&
            line.find(L"type_activation") == std::wstring::npos) {
          ERR(L"Can not find `type_activation` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (size_t_input >= static_cast<size_t>(LAYER_ACTIVATION::LENGTH)) {
            ERR(L"Undefined layer activation type %zu.", size_t_input);

            return false;
          }

          layer_it->type_activation =
              static_cast<LAYER_ACTIVATION::TYPE>(size_t_input);
        }
        // |END| Type activation. |END|

        // Dropout.
        if (load_dropout_params_fn(layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`load_dropout_params_fn()` function.");
          return false;
        }
        // |END| Dropout. |END|

        // Normalization.
        if ((file >> line) &&
            line.find(L"type_normalization") == std::wstring::npos) {
          ERR(L"Can not find `type_normalization` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (size_t_input >= static_cast<size_t>(LAYER_NORM::LENGTH)) {
            ERR(L"Undefined layer normalization type %zu.", size_t_input);

            return false;
          }

          if (this->Set__Layer_Normalization(
                  layer_it, static_cast<LAYER_NORM::TYPE>(size_t_input), false,
                  false) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Layer_Normalization(%zu)` function.",
                size_t_input);
            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"use_layer_normalization_before_activation") ==
                std::wstring::npos) {
          ERR(L"Can not find `use_layer_normalization_before_activation` "
              L"inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->use_layer_normalization_before_activation >>
              std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"number_normalized_units") == std::wstring::npos) {
          ERR(L"Can not find `number_normalized_units` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
          tmp_ptr_array_normalized_units += size_t_input;
          layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;

          this->Order__Layer__Normalization(layer_it);

          if (this->Load_Dimension__Normalization(layer_it, file) == false) {
            ERR(L"An error has been triggered from the "
                L"`Load_Dimension__Normalization()` function.");
            return false;
          }
        }
        // |END| Normalization. |END|

        if ((file >> line) &&
            line.find(L"use_tied_parameter") == std::wstring::npos) {
          ERR(L"Can not find `use_tied_parameter` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_boolean >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__Tied_Parameter(layer_it, tmp_input_boolean, false) ==
              false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Tied_Parameter(ptr, %ls, false)` function.",
                to_wstring(tmp_input_boolean).c_str());
            return false;
          }
        }

        // k-Sparse filters.
        if ((file >> line) && line.find(L"k_sparsity") == std::wstring::npos) {
          ERR(L"Can not find `k_sparsity` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__K_Sparsity(layer_it, size_t_input) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__K_Sparsity(ptr, %zu)` function.",
                size_t_input);
            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"alpha_sparsity") == std::wstring::npos) {
          ERR(L"Can not find `alpha_sparsity` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_T[0] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__Alpha_Sparsity(layer_it, tmp_input_T[0]) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Alpha_Sparsity(ptr, %f)` function.",
                tmp_input_T[0]);
            return false;
          }
        }
        // |END| k-Sparse filters. |END|

        // Constraint.
        if ((file >> line) &&
            line.find(L"constraint_recurrent_weight_lower_bound") ==
                std::wstring::npos) {
          ERR(L"Can not find `constraint_recurrent_weight_lower_bound` "
              L"inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_T[0] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"constraint_recurrent_weight_upper_bound") ==
                std::wstring::npos) {
          ERR(L"Can not find `constraint_recurrent_weight_upper_bound` "
              L"inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_T[1] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__Regularization__Constraint_Recurrent_Weight(
                  layer_it, tmp_input_T[0], tmp_input_T[1]) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Regularization__Constraint_Recurrent_Weight"
                L"(ptr, %f, %f)` function.",
                tmp_input_T[0], tmp_input_T[1]);
            return false;
          }
        }
        // |END| Constraint. |END|

        // Initialize basic unit(s).
        layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
        layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
        // |END| Initialize basic unit(s). |END|

        // Initialize basic indice unit(s).
        layer_it->ptr_array_basic_indice_units =
            tmp_ptr_array_basic_indice_units;
        layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
        // |END| Initialize basic indice unit(s). |END|

        // Initialize neuron unit(s).
        if ((file >> line) &&
            line.find(L"number_neuron_units") == std::wstring::npos) {
          ERR(L"Can not find `number_neuron_units` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> *layer_it->ptr_number_outputs >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
          tmp_ptr_array_neuron_units += *layer_it->ptr_number_outputs;
          layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;

          this->Order__Layer__Neuron(layer_it);

          switch (tmp_ptr_previous_layer->type_layer) {
            case LAYER::AVERAGE_POOLING:
            case LAYER::RESIDUAL:
              if (this->Load_Dimension__FC<Basic_unit, LAYER::AVERAGE_POOLING>(
                      layer_it, this->ptr_array_basic_units, file) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__FC()` function.");
                return false;
              }
              break;
            case LAYER::FULLY_CONNECTED:
            case LAYER::FULLY_CONNECTED_RECURRENT:
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
              if (this->Load_Dimension__FC<Neuron_unit, LAYER::FULLY_CONNECTED>(
                      layer_it, this->ptr_array_neuron_units, file) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__FC()` function.");
                return false;
              }
              break;
            case LAYER::LSTM:
              if (this->Load_Dimension__FC<CellUnit, LAYER::LSTM>(
                      layer_it, this->ptr_array_cell_units, file) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__FC()` function.");
                return false;
              }
              break;
            case LAYER::MAX_POOLING:
              if (this->Load_Dimension__FC<Basic_indice_unit,
                                           LAYER::MAX_POOLING>(
                      layer_it, this->ptr_array_basic_indice_units, file) ==
                  false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__FC()` function.");
                return false;
              }
              break;
            default:
              ERR(L"Layer type (%d | %ls) is not managed in the switch.",
                  tmp_ptr_previous_layer->type_layer,
                  LAYER_NAME[tmp_ptr_previous_layer->type_layer].c_str());
              return false;
          }
        }
        // |END| Initialize neuron unit(s). |END|

        // Initialize AF unit(s).
        layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
        layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
        // |END| Initialize AF unit(s). |END|

        // Initialize AF Ind recurrent unit(s).
        if ((file >> line) &&
            line.find(L"number_AF_Ind_recurrent_units") == std::wstring::npos) {
          ERR(L"Can not find `number_AF_Ind_recurrent_units` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_AF_Ind_recurrent_units =
              tmp_ptr_array_AF_Ind_recurrent_units;
          tmp_ptr_array_AF_Ind_recurrent_units += size_t_input;
          layer_it->ptr_last_AF_Ind_recurrent_unit =
              tmp_ptr_array_AF_Ind_recurrent_units;

          this->Order__Layer__AF_Ind_Recurrent(layer_it);

          if (this->Load_Dimension__AF_Ind_Recurrent(layer_it, file) == false) {
            ERR(L"An error has been triggered from the "
                L"`Load_Dimension__AF_Ind_Recurrent()` function.");
            return false;
          }
        }
        // |END| Initialize AF Ind recurrent unit(s). |END|

        // Initialize block/cell unit(s).
        layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
        layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

        layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
        layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
        // |END| Initialize block/cell unit(s). |END|

        // Initialize bias parameter(s).
        if (this->Load_Dimension__Bias(layer_it, file) == false) {
          ERR(L"An error has been triggered from the "
              L"`Load_Dimension__Bias()` function.");
          return false;
        }
        // |END| Initialize bias parameter(s). |END|
        break;
      case LAYER::LSTM:
        // Type activation.
        if ((file >> line) &&
            line.find(L"type_activation") == std::wstring::npos) {
          ERR(L"Can not find `type_activation` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (size_t_input >= static_cast<size_t>(LAYER_ACTIVATION::LENGTH)) {
            ERR(L"Undefined layer activation type %zu.", size_t_input);

            return false;
          }

          layer_it->type_activation =
              static_cast<LAYER_ACTIVATION::TYPE>(size_t_input);
        }
        // |END| Type activation. |END|

        // Dropout.
        if (load_dropout_params_fn(layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`load_dropout_params_fn()` function.");
          return false;
        }
        // |END| Dropout. |END|

        // Normalization.
        if ((file >> line) &&
            line.find(L"type_normalization") == std::wstring::npos) {
          ERR(L"Can not find `type_normalization` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (size_t_input >= static_cast<size_t>(LAYER_NORM::LENGTH)) {
            ERR(L"Undefined layer normalization type %zu.", size_t_input);

            return false;
          }

          if (this->Set__Layer_Normalization(
                  layer_it, static_cast<LAYER_NORM::TYPE>(size_t_input), false,
                  false) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Layer_Normalization(%zu)` function.",
                size_t_input);
            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"use_layer_normalization_before_activation") ==
                std::wstring::npos) {
          ERR(L"Can not find `use_layer_normalization_before_activation` "
              L"inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->use_layer_normalization_before_activation >>
              std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"number_normalized_units") == std::wstring::npos) {
          ERR(L"Can not find `number_normalized_units` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
          tmp_ptr_array_normalized_units += size_t_input;
          layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;

          this->Order__Layer__Normalization(layer_it);

          if (this->Load_Dimension__Normalization(layer_it, file) == false) {
            ERR(L"An error has been triggered from the "
                L"`Load_Dimension__Normalization()` function.");
            return false;
          }
        }
        // |END| Normalization. |END|

        if ((file >> line) &&
            line.find(L"use_tied_parameter") == std::wstring::npos) {
          ERR(L"Can not find `use_tied_parameter` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_boolean >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__Tied_Parameter(layer_it, tmp_input_boolean, false) ==
              false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Tied_Parameter(ptr, %ls, false)` function.",
                to_wstring(tmp_input_boolean).c_str());
            return false;
          }
        }

        // k-Sparse filters.
        if ((file >> line) && line.find(L"k_sparsity") == std::wstring::npos) {
          ERR(L"Can not find `k_sparsity` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__K_Sparsity(layer_it, size_t_input) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__K_Sparsity(ptr, %zu)` function.",
                size_t_input);
            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"alpha_sparsity") == std::wstring::npos) {
          ERR(L"Can not find `alpha_sparsity` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_T[0] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__Alpha_Sparsity(layer_it, tmp_input_T[0]) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Alpha_Sparsity(ptr, %f)` function.",
                tmp_input_T[0]);
            return false;
          }
        }
        // |END| k-Sparse filters. |END|

        // Constraint.
        if ((file >> line) &&
            line.find(L"constraint_recurrent_weight_lower_bound") ==
                std::wstring::npos) {
          ERR(L"Can not find `constraint_recurrent_weight_lower_bound` "
              L"inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_T[0] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"constraint_recurrent_weight_upper_bound") ==
                std::wstring::npos) {
          ERR(L"Can not find `constraint_recurrent_weight_upper_bound` "
              L"inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> tmp_input_T[1] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (this->Set__Regularization__Constraint_Recurrent_Weight(
                  layer_it, tmp_input_T[0], tmp_input_T[1]) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Regularization__Constraint_Recurrent_Weight"
                L"(ptr, %f, %f)` function.",
                tmp_input_T[0], tmp_input_T[1]);
            return false;
          }
        }
        // |END| Constraint. |END|

        // Initialize basic unit(s).
        layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
        layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
        // |END| Initialize basic unit(s). |END|

        // Initialize basic indice unit(s).
        layer_it->ptr_array_basic_indice_units =
            tmp_ptr_array_basic_indice_units;
        layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
        // |END| Initialize basic indice unit(s). |END|

        // Initialize neuron unit(s).
        layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
        layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
        // |END| Initialize neuron unit(s). |END|

        // Initialize AF unit(s).
        layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
        layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
        // |END| Initialize AF unit(s). |END|

        // Initialize AF Ind recurrent unit(s).
        layer_it->ptr_array_AF_Ind_recurrent_units =
            tmp_ptr_array_AF_Ind_recurrent_units;
        layer_it->ptr_last_AF_Ind_recurrent_unit =
            tmp_ptr_array_AF_Ind_recurrent_units;
        // |END| Initialize AF Ind recurrent unit(s). |END|

        // Initialize block/cell unit(s).
        if ((file >> line) &&
            line.find(L"number_block_units") == std::wstring::npos) {
          ERR(L"Can not find `number_block_units` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
          tmp_ptr_array_block_units += size_t_input;
          layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

          if (this->Load_Dimension__Cell_Units(
                  layer_it, tmp_ptr_array_cell_units, file) == false) {
            ERR(L"An error has been triggered from the "
                L"`Load_Dimension__Cell_Units()` function.");
            return false;
          }

          *layer_it->ptr_number_outputs = static_cast<size_t>(
              layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units);

          this->Order__Layer__LSTM(layer_it);

          switch (tmp_ptr_previous_layer->type_layer) {
            case LAYER::AVERAGE_POOLING:
            case LAYER::RESIDUAL:
              if (this->Load_Dimension__LSTM<Basic_unit,
                                             LAYER::AVERAGE_POOLING>(
                      layer_it, this->ptr_array_basic_units, file) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__LSTM()` function.");
                return false;
              }
              break;
            case LAYER::FULLY_CONNECTED:
            case LAYER::FULLY_CONNECTED_RECURRENT:
            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
              if (this->Load_Dimension__LSTM<Neuron_unit,
                                             LAYER::FULLY_CONNECTED>(
                      layer_it, this->ptr_array_neuron_units, file) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__LSTM()` function.");
                return false;
              }
              break;
            case LAYER::LSTM:
              if (this->Load_Dimension__LSTM<CellUnit, LAYER::LSTM>(
                      layer_it, this->ptr_array_cell_units, file) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__LSTM()` function.");
                return false;
              }
              break;
            case LAYER::MAX_POOLING:
              if (this->Load_Dimension__LSTM<Basic_indice_unit,
                                             LAYER::MAX_POOLING>(
                      layer_it, this->ptr_array_basic_indice_units, file) ==
                  false) {
                ERR(L"An error has been triggered from the "
                    L"`Load_Dimension__LSTM()` function.");
                return false;
              }
              break;
            default:
              ERR(L"Layer type (%d | %ls) is not managed in the switch.",
                  tmp_ptr_previous_layer->type_layer,
                  LAYER_NAME[tmp_ptr_previous_layer->type_layer].c_str());
              return false;
          }
        }
        // |END| Initialize block/cell unit(s). |END|

        // Initialize bias parameter(s).
        if (this->Load_Dimension__Bias(layer_it, file) == false) {
          ERR(L"An error has been triggered from the "
              L"`Load_Dimension__Bias()` function.");
          return false;
        }
        // |END| Initialize bias parameter(s). |END|
        break;
      case LAYER::MAX_POOLING:
        // Pooling.
        if ((file >> line) && line.find(L"kernel_size") == std::wstring::npos) {
          ERR(L"Can not find `kernel_size` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->pooling_values[0] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) && line.find(L"stride") == std::wstring::npos) {
          ERR(L"Can not find `stride` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->pooling_values[1] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) && line.find(L"padding") == std::wstring::npos) {
          ERR(L"Can not find `padding` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->pooling_values[2] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }

        if ((file >> line) && line.find(L"dilation") == std::wstring::npos) {
          ERR(L"Can not find `dilation` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->pooling_values[3] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }
        // |END| Pooling. |END|

        //  Initialize normalized unit(s).
        layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
        layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
        //  |END| Initialize normalized unit(s). |END|

        //  Initialize basic unit(s).
        layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
        layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
        //  |END| Initialize basic unit(s). |END|

        // Initialize basic indice unit(s).
        if ((file >> line) &&
            line.find(L"number_basic_indice_units") == std::wstring::npos) {
          ERR(L"Can not find `number_basic_indice_units` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> *layer_it->ptr_number_outputs >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_basic_indice_units =
              tmp_ptr_array_basic_indice_units;
          tmp_ptr_array_basic_indice_units += *layer_it->ptr_number_outputs;
          layer_it->ptr_last_basic_indice_unit =
              tmp_ptr_array_basic_indice_units;

          this->Order__Layer__Basic_indice(layer_it);
        }
        // |END| Initialize basic indice unit(s). |END|

        // Initialize neuron unit(s).
        layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
        layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
        // |END| Initialize neuron unit(s). |END|

        // Initialize AF unit(s).
        layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
        layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
        // |END| Initialize AF unit(s). |END|

        // Initialize AF Ind recurrent unit(s).
        layer_it->ptr_array_AF_Ind_recurrent_units =
            tmp_ptr_array_AF_Ind_recurrent_units;
        layer_it->ptr_last_AF_Ind_recurrent_unit =
            tmp_ptr_array_AF_Ind_recurrent_units;
        // |END| Initialize AF Ind recurrent unit(s). |END|

        //  Initialize block/cell unit(s).
        layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
        layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

        layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
        layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
        //  |END| Initialize block/cell unit(s). |END|
        break;
      case LAYER::RESIDUAL:
        // Initialize block depth.
        if ((file >> line) && line.find(L"block_depth") == std::wstring::npos) {
          ERR(L"Can not find `block_depth` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->block_depth >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }
        // |END| Initialize block depth. |END|

        // Initialize padding.
        if ((file >> line) && line.find(L"padding") == std::wstring::npos) {
          ERR(L"Can not find `padding` inside \"%ls\".", line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> layer_it->pooling_values[2] >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }
        // |END| Initialize padding. |END|

        // Dropout.
        if (load_dropout_params_fn(layer_it) == false) {
          ERR(L"An error has been triggered from the "
              L"`load_dropout_params_fn()` function.");
          return false;
        }
        // |END| Dropout. |END|

        // Normalization.
        if ((file >> line) &&
            line.find(L"type_normalization") == std::wstring::npos) {
          ERR(L"Can not find `type_normalization` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          if (size_t_input >= static_cast<size_t>(LAYER_NORM::LENGTH)) {
            ERR(L"Undefined layer normalization type %zu.", size_t_input);

            return false;
          }

          if (this->Set__Layer_Normalization(
                  layer_it, static_cast<LAYER_NORM::TYPE>(size_t_input), false,
                  false) == false) {
            ERR(L"An error has been triggered from the "
                L"`Set__Layer_Normalization(%zu)` function.",
                size_t_input);
            return false;
          }
        }

        if ((file >> line) &&
            line.find(L"number_normalized_units") == std::wstring::npos) {
          ERR(L"Can not find `number_normalized_units` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> size_t_input >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
          tmp_ptr_array_normalized_units += size_t_input;
          layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;

          this->Order__Layer__Normalization(layer_it);

          if (this->Load_Dimension__Normalization(layer_it, file) == false) {
            ERR(L"An error has been triggered from the "
                L"`Load_Dimension__Normalization()` function.");
            return false;
          }
        }
        // |END| Normalization. |END|

        // Initialize basic unit(s).
        if ((file >> line) &&
            line.find(L"number_basic_units") == std::wstring::npos) {
          ERR(L"Can not find `number_basic_units` inside \"%ls\".",
              line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> *layer_it->ptr_number_outputs >> std::ws;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }

          layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
          tmp_ptr_array_basic_units += *layer_it->ptr_number_outputs;
          layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;

          this->Order__Layer__Basic(layer_it);
        }
        // |END| Initialize basic unit(s). |END|

        //  Initialize basic indice unit(s).
        layer_it->ptr_array_basic_indice_units =
            tmp_ptr_array_basic_indice_units;
        layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
        //  |END| Initialize basic indice unit(s). |END|

        // Initialize neuron unit(s).
        layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
        layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
        // |END| Initialize neuron unit(s). |END|

        // Initialize AF unit(s).
        layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
        layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
        // |END| Initialize AF unit(s). |END|

        // Initialize AF Ind recurrent unit(s).
        layer_it->ptr_array_AF_Ind_recurrent_units =
            tmp_ptr_array_AF_Ind_recurrent_units;
        layer_it->ptr_last_AF_Ind_recurrent_unit =
            tmp_ptr_array_AF_Ind_recurrent_units;
        // |END| Initialize AF Ind recurrent unit(s). |END|

        //  Initialize block/cell unit(s).
        layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
        layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

        layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
        layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
        //  |END| Initialize block/cell unit(s). |END|
        break;
      default:
        ERR(L"Layer type (%d | %ls) is not managed in the switch.",
            layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
        return false;
    }
  }
  // |END| Hidden layer. |END|

  // allocate bidirectional layer(s).
  if (this->Allocate__Bidirectional__Layers() == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Bidirectional__Layers()` function.");
    return false;
  }
  // |END| allocate bidirectional layer(s). |END|

  // Output layer.
  //  Type layer.
  INFO(L"load output layer.");
  getline(file, line);  // "Output layer:"

  if ((file >> line) && line.find(L"type_layer") == std::wstring::npos) {
    ERR(L"Can not find `type_layer` inside \"%ls\".", line.c_str());

    file.close();

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    file.close();

    return false;
  } else {
    file >> size_t_input;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      file.close();

      return false;
    }

    if (size_t_input >= static_cast<size_t>(LAYER::LENGTH)) {
      ERR(L"Undefined layer type %zu.", size_t_input);

      file.close();

      return false;
    }

    layer_it->type_layer = static_cast<LAYER::TYPE>(size_t_input);
  }
  //  |END| Type layer. |END|

  this->Organize__Previous_Layers_Connected(tmp_state_layer_index, layer_it,
                                            tmp_ptr_layer_state);

  tmp_ptr_previous_layer = layer_it->previous_connected_layers[0];

  //  Type activation.
  if ((file >> line) && line.find(L"type_activation") == std::wstring::npos) {
    ERR(L"Can not find `type_activation` inside \"%ls\".", line.c_str());

    file.close();

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    file.close();

    return false;
  } else {
    file >> size_t_input;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      file.close();

      return false;
    }

    if (size_t_input >= static_cast<size_t>(LAYER_ACTIVATION::LENGTH)) {
      ERR(L"Undefined layer activation type %zu.", size_t_input);

      file.close();

      return false;
    }

    layer_it->type_activation =
        static_cast<LAYER_ACTIVATION::TYPE>(size_t_input);
  }
  //  |END| Type activation. |END|

  //  Initialize output unit(s).
  if ((file >> line) && line.find(L"n_out") == std::wstring::npos) {
    ERR(L"Can not find `n_out` inside \"%ls\".", line.c_str());

    file.close();

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    file.close();

    return false;
  } else {
    file >> size_t_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      file.close();

      return false;
    }

    *layer_it->ptr_number_outputs = this->n_out = size_t_input;

    layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
    tmp_ptr_array_neuron_units += size_t_input;
    layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;

    this->Order__Layer__Neuron(layer_it);

    switch (tmp_ptr_previous_layer->type_layer) {
      case LAYER::AVERAGE_POOLING:
      case LAYER::RESIDUAL:
        if (this->Load_Dimension__FC<Basic_unit, LAYER::AVERAGE_POOLING>(
                layer_it, this->ptr_array_basic_units, file) == false) {
          ERR(L"An error has been triggered from the "
              L"`Load_Dimension__FC()` function.");
          return false;
        }
        break;
      case LAYER::FULLY_CONNECTED:
      case LAYER::FULLY_CONNECTED_RECURRENT:
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        if (this->Load_Dimension__FC<Neuron_unit, LAYER::FULLY_CONNECTED>(
                layer_it, this->ptr_array_neuron_units, file) == false) {
          ERR(L"An error has been triggered from the "
              L"`Load_Dimension__FC()` function.");
          return false;
        }
        break;
      case LAYER::LSTM:
        if (this->Load_Dimension__FC<CellUnit, LAYER::LSTM>(
                layer_it, this->ptr_array_cell_units, file) == false) {
          ERR(L"An error has been triggered from the "
              L"`Load_Dimension__FC()` function.");
          return false;
        }
        break;
      case LAYER::MAX_POOLING:
        if (this->Load_Dimension__FC<Basic_indice_unit, LAYER::MAX_POOLING>(
                layer_it, this->ptr_array_basic_indice_units, file) == false) {
          ERR(L"An error has been triggered from the "
              L"`Load_Dimension__FC()` function.");
          return false;
        }
        break;
      default:
        ERR(L"Layer type (%d | %ls) is not managed in the switch.",
            tmp_ptr_previous_layer->type_layer,
            LAYER_NAME[tmp_ptr_previous_layer->type_layer].c_str());
        return false;
    }
  }
  //  |END| Initialize output unit(s). |END|

  // Initialize AF unit(s).
  if ((file >> line) && line.find(L"number_AF_units") == std::wstring::npos) {
    ERR(L"Can not find `number_AF_units` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
    tmp_ptr_array_AF_units += size_t_input;
    layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;

    this->Order__Layer__AF(layer_it);

    if (this->Load_Dimension__AF(layer_it, file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__AF()` function.");
      return false;
    }
  }
  // |END| Initialize AF unit(s). |END|

  // Initialize AF Ind recurrent unit(s).
  layer_it->ptr_array_AF_Ind_recurrent_units =
      tmp_ptr_array_AF_Ind_recurrent_units;
  layer_it->ptr_last_AF_Ind_recurrent_unit =
      tmp_ptr_array_AF_Ind_recurrent_units;
  // |END| Initialize AF Ind recurrent unit(s). |END|

  //  Initialize basic unit(s).
  layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
  layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
  //  |END| Initialize basic unit(s). |END|

  //  Initialize basic indice unit(s).
  layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
  layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
  //  |END| Initialize basic indice unit(s). |END|

  //  Initialize block/cell unit(s).
  layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
  layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

  layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
  layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
  //  |END| Initialize block/cell unit(s). |END|

  //  Initialize bias parameter(s).
  if (this->Load_Dimension__Bias(layer_it, file) == false) {
    ERR(L"An error has been triggered from the "
        L"`Load_Dimension__Bias()` function.");
    return false;
  }
  //  |END| Initialize bias parameter(s). |END|
  // |END| Output layer. |END|

  if (file.fail()) {
    ERR(L"Logical error on i/o operation `%ls`.", path_params.c_str());
    return false;
  }

  file.close();

  if (this->total_weights != this->total_weights_allocated) {
    ERR(L"Total weights prepared (%zu) "
        L"differ from the total weights allocated (%zu).",
        this->total_weights, this->total_weights_allocated);
    return false;
  } else if (this->total_bias != this->total_bias_allocated) {
    ERR(L"Total bias prepared (%zu) "
        L"differ from the total bias allocated (%zu).",
        this->total_bias, this->total_bias_allocated);
    return false;
  }
  // |END| Initialize layer(s). |END|

  // Layers, connections.
  this->Order__Layers__Connection();

  // Layers, outputs pointers.
  this->Order__Layers__Output();

  if (this->load_spec_params(path_spec_params) == false) {
    ERR(L"An error has been triggered from the "
        L"`load_spec_params(%ls)` function.",
        path_spec_params.c_str());
    return false;
  }

  return true;
}

bool Model::Load_Dimension__Neuron(Neuron_unit *const ptr_neuron_received,
                                   std::wifstream &file) {
  std::wstring line;

  getline(file, line);  // "Neuron_unit[%zu]"

  // Number connection(s).
  if ((file >> line) &&
      line.find(L"number_connections") == std::wstring::npos) {
    ERR(L"Can not find `number_connections` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> *ptr_neuron_received->ptr_number_connections >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  *ptr_neuron_received->ptr_first_connection_index = this->total_weights;
  this->total_weights += *ptr_neuron_received->ptr_number_connections;
  *ptr_neuron_received->ptr_last_connection_index = this->total_weights;
  // |END| Number connection(s). |END|

  return true;
}

bool Model::Load_Dimension__AF(AF_unit *const ptr_AF_received,
                               std::wifstream &file) {
  size_t size_t_input;

  std::wstring line;

  getline(file, line);  // "AF[%zu]"

  // Activation function.
  if ((file >> line) &&
      line.find(L"activation_function") == std::wstring::npos) {
    ERR(L"Can not find `activation_function` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    if (size_t_input >= static_cast<size_t>(ACTIVATION::LENGTH)) {
      ERR(L"Undefined activation function type %zu.", size_t_input);

      return false;
    }

    *ptr_AF_received->ptr_type_activation_function =
        static_cast<ACTIVATION::TYPE>(size_t_input);
  }
  // |END| Activation function. |END|

  return true;
}

bool Model::Load_Dimension__AF_Ind_Recurrent(
    AF_Ind_recurrent_unit *const ptr_AF_Ind_received, std::wifstream &file) {
  size_t size_t_input;

  std::wstring line;

  getline(file, line);  // "AF_Ind_R[%zu]"

  // Activation function.
  if ((file >> line) &&
      line.find(L"activation_function") == std::wstring::npos) {
    ERR(L"Can not find `activation_function` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    if (size_t_input >= static_cast<size_t>(ACTIVATION::LENGTH)) {
      ERR(L"Undefined activation function type %zu.", size_t_input);

      return false;
    }

    *ptr_AF_Ind_received->ptr_type_activation_function =
        static_cast<ACTIVATION::TYPE>(size_t_input);
  }
  // |END| Activation function. |END|

  *ptr_AF_Ind_received->ptr_recurrent_connection_index = this->total_weights++;

  AF_Ind_recurrent_unit **tmp_ptr_array_U_ptr_connections(
      reinterpret_cast<AF_Ind_recurrent_unit **>(
          this->ptr_array_ptr_connections));

  if (this->Load_Dimension__Connection<
          AF_Ind_recurrent_unit,
          LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT>(
          *ptr_AF_Ind_received->ptr_recurrent_connection_index,
          this->ptr_array_parameters, this->ptr_array_AF_Ind_recurrent_units,
          tmp_ptr_array_U_ptr_connections, file) == false) {
    ERR(L"An error has been triggered from the "
        L"`Load_Dimension__Connection(%zu)` function.",
        *ptr_AF_Ind_received->ptr_recurrent_connection_index);
    return false;
  }

  return true;
}

bool Model::Load_Dimension__Normalized_Unit(
    size_t const number_units_received,
    LAYER_NORM::TYPE const type_normalization_received,
    union Normalized_unit *const ptr_normalized_unit_received,
    std::wifstream &file) {
  size_t tmp_time_step_index, tmp_unit_timed_index;

  std::wstring line;

  real out;

  getline(file, line);  // "NormU[%zu]"

  switch (type_normalization_received) {
    case LAYER_NORM::BATCH_NORMALIZATION:
    case LAYER_NORM::BATCH_RENORMALIZATION:
    case LAYER_NORM::GHOST_BATCH_NORMALIZATION:
      // Scale.
      if ((file >> line) && line.find(L"scale") == std::wstring::npos) {
        ERR(L"Can not find `scale` inside \"%ls\".", line.c_str());

        return false;
      } else if (file.fail()) {
        ERR(L"Can not read properly inside \"%ls\".", line.c_str());

        return false;
      } else {
        file >> out >> std::ws;
        *ptr_normalized_unit_received->normalized_batch_units.ptr_scale = out;

        if (file.fail()) {
          ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

          return false;
        }
      }
      // |END| Scale. |END|

      // shift.
      if ((file >> line) && line.find(L"shift") == std::wstring::npos) {
        ERR(L"Can not find `scale` inside \"%ls\".", line.c_str());

        return false;
      } else if (file.fail()) {
        ERR(L"Can not read properly inside \"%ls\".", line.c_str());

        return false;
      } else {
        file >> out >> std::ws;
        *ptr_normalized_unit_received->normalized_batch_units.ptr_shift = out;

        if (file.fail()) {
          ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

          return false;
        }
      }
      // |END| shift. |END|

      for (tmp_time_step_index = 0_UZ; tmp_time_step_index != this->seq_w;
           ++tmp_time_step_index) {
        tmp_unit_timed_index = number_units_received * tmp_time_step_index;

        // Mean average.
        if ((file >> line) &&
            line.find(L"mean_average[" + std::to_wstring(tmp_time_step_index) +
                      L"]") == std::wstring::npos) {
          ERR(L"Can not find `mean_average[%zu]` inside \"%ls\".",
              tmp_time_step_index, line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> out >> std::ws;
          ptr_normalized_unit_received->normalized_batch_units
              .ptr_mean_average[tmp_unit_timed_index] = out;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }
        // |END| Mean average. |END|

        // Variance average.
        if ((file >> line) && line.find(L"variance_average[" +
                                        std::to_wstring(tmp_time_step_index) +
                                        L"]") == std::wstring::npos) {
          ERR(L"Can not find `variance_average[%zu]` inside \"%ls\".",
              tmp_time_step_index, line.c_str());

          return false;
        } else if (file.fail()) {
          ERR(L"Can not read properly inside \"%ls\".", line.c_str());

          return false;
        } else {
          file >> out >> std::ws;
          ptr_normalized_unit_received->normalized_batch_units
              .ptr_variance_average[tmp_unit_timed_index] = out;

          if (file.fail()) {
            ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

            return false;
          }
        }
        // |END| Variance average. |END|
      }
      break;
    default:
      ERR(L"Layer normalization type (%d | %ls) is not managed in the switch.",
          type_normalization_received,
          LAYER_NORM_NAME[type_normalization_received].c_str());
      break;
  }

  return true;
}

bool Model::Load_Dimension__Bias(Layer *const layer_it, std::wifstream &file) {
  size_t size_t_input;

  std::wstring line;

  if ((file >> line) &&
      line.find(L"number_bias_parameters") == std::wstring::npos) {
    ERR(L"Can not find `number_bias_parameters` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  var *const tmp_ptr_array_parameters(this->ptr_array_parameters +
                                      this->total_weights_allocated +
                                      this->total_bias);

  real out;

  layer_it->first_bias_connection_index =
      this->total_weights_allocated + this->total_bias;

  for (size_t tmp_connection_index(0_UZ); tmp_connection_index != size_t_input;
       ++tmp_connection_index) {
    if ((file >> line) && line.find(L"weight") == std::wstring::npos) {
      ERR(L"Can not find `weight` inside \"%ls\".", line.c_str());

      return false;
    } else if (file.fail()) {
      ERR(L"Can not read properly inside \"%ls\".", line.c_str());

      return false;
    } else {
      file >> out >> std::ws;
      tmp_ptr_array_parameters[tmp_connection_index] = out;

      if (file.fail()) {
        ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

        return false;
      }
    }
  }

  this->total_bias += size_t_input;

  layer_it->last_bias_connection_index =
      this->total_weights_allocated + this->total_bias;

  return true;
}

bool Model::Load_Dimension__Block(
    size_t const layer_number_block_units_received,
    size_t const layer_number_cell_units_received,
    LAYER_NORM::TYPE const type_normalization_received,
    BlockUnit *const ptr_block_unit_it_received, std::wifstream &file) {
  CellUnit const *const tmp_ptr_block_ptr_cell_unit(
      ptr_block_unit_it_received->ptr_array_cell_units),
      *const tmp_ptr_block_ptr_last_cell_unit(
          ptr_block_unit_it_received->ptr_last_cell_unit);
  CellUnit *tmp_ptr_block_ptr_cell_unit_it;

  size_t const tmp_block_number_cell_units(static_cast<size_t>(
      tmp_ptr_block_ptr_last_cell_unit - tmp_ptr_block_ptr_cell_unit));
  size_t size_t_input;

  std::wstring line;

  getline(file, line);  // "Block[%zu]"

  // Activation function.
  if ((file >> line) &&
      line.find(L"activation_function") == std::wstring::npos) {
    ERR(L"Can not find `activation_function` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    if (size_t_input >= static_cast<size_t>(ACTIVATION::LENGTH)) {
      ERR(L"Undefined activation function type %zu.", size_t_input);

      return false;
    }

    ptr_block_unit_it_received->activation_function_io =
        static_cast<ACTIVATION::TYPE>(size_t_input);
  }
  // |END| Activation function. |END|

  // Number connection(s).
  if ((file >> line) &&
      line.find(L"number_connections") == std::wstring::npos) {
    ERR(L"Can not find `number_connections` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

#ifndef NO_PEEPHOLE
  size_t const tmp_number_inputs((size_t_input -
                                  layer_number_cell_units_received * 4_UZ -
                                  tmp_block_number_cell_units * 3_UZ) /
                                 4_UZ);
#else
  size_t const tmp_number_inputs(
      (size_t_input - layer_number_cell_units_received * 4_UZ) / 4_UZ);
#endif

  ptr_block_unit_it_received->first_index_connection = this->total_weights;

  // [0] Cell input.
  for (tmp_ptr_block_ptr_cell_unit_it =
           ptr_block_unit_it_received->ptr_array_cell_units;
       tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit;
       ++tmp_ptr_block_ptr_cell_unit_it) {
    //    [1] Input, cell.
    tmp_ptr_block_ptr_cell_unit_it
        ->first_index_feedforward_connection_cell_input = this->total_weights;
    this->total_weights += tmp_number_inputs;
    tmp_ptr_block_ptr_cell_unit_it
        ->last_index_feedforward_connection_cell_input = this->total_weights;
    //    [1] |END| Input, cell. |END|

    //    [1] Recurrent, cell.
    tmp_ptr_block_ptr_cell_unit_it
        ->first_index_recurrent_connection_cell_input = this->total_weights;
    this->total_weights += layer_number_cell_units_received;
    tmp_ptr_block_ptr_cell_unit_it->last_index_recurrent_connection_cell_input =
        this->total_weights;
    //    [1] |END| Recurrent, cell. |END|
  }
  // [0] |END| Cell input. |END|

  // [0] Input, gates.
  //    [1] Input gate.
  ptr_block_unit_it_received->first_index_feedforward_connection_input_gate =
      this->total_weights;
  this->total_weights += tmp_number_inputs;
  ptr_block_unit_it_received->last_index_feedforward_connection_input_gate =
      this->total_weights;
  //    [1] |END| Input gate. |END|

  //    [1] Forget gate.
  ptr_block_unit_it_received->first_index_feedforward_connection_forget_gate =
      this->total_weights;
  this->total_weights += tmp_number_inputs;
  ptr_block_unit_it_received->last_index_feedforward_connection_forget_gate =
      this->total_weights;
  //    [1] |END| Forget gate. |END|

  //    [1] Output gate.
  ptr_block_unit_it_received->first_index_feedforward_connection_output_gate =
      this->total_weights;
  this->total_weights += tmp_number_inputs;
  ptr_block_unit_it_received->last_index_feedforward_connection_output_gate =
      this->total_weights;
  //    [1] |END| Output gate. |END|
  // [0] |END| Input, gates. |END|

  // [0] Recurrent, gates.
  //    [1] Input gate.
  ptr_block_unit_it_received->first_index_recurrent_connection_input_gate =
      this->total_weights;
  this->total_weights += layer_number_cell_units_received;
  ptr_block_unit_it_received->last_index_recurrent_connection_input_gate =
      this->total_weights;
  //    [1] |END| Input gate. |END|

  //    [1] Forget gate.
  ptr_block_unit_it_received->first_index_recurrent_connection_forget_gate =
      this->total_weights;
  this->total_weights += layer_number_cell_units_received;
  ptr_block_unit_it_received->last_index_recurrent_connection_forget_gate =
      this->total_weights;
  //    [1] |END| Forget gate. |END|

  //    [1] Output gate.
  ptr_block_unit_it_received->first_index_recurrent_connection_output_gate =
      this->total_weights;
  this->total_weights += layer_number_cell_units_received;
  ptr_block_unit_it_received->last_index_recurrent_connection_output_gate =
      this->total_weights;
  //    [1] |END| Output gate. |END|
  // [0] |END| Recurrent, gates. |END|

#ifndef NO_PEEPHOLE
  // [0] Peepholes.
  //    [1] Input gate.
  ptr_block_unit_it_received->first_index_peephole_input_gate =
      this->total_weights;
  this->total_weights += tmp_block_number_cell_units;
  ptr_block_unit_it_received->last_index_peephole_input_gate =
      this->total_weights;
  //    [1] |END| Input gate. |END|

  //    [1] Forget gate.
  ptr_block_unit_it_received->first_index_peephole_forget_gate =
      this->total_weights;
  this->total_weights += tmp_block_number_cell_units;
  ptr_block_unit_it_received->last_index_peephole_forget_gate =
      this->total_weights;
  //    [1] |END| Forget gate. |END|

  //    [1] Output gate.
  ptr_block_unit_it_received->first_index_peephole_output_gate =
      this->total_weights;
  this->total_weights += tmp_block_number_cell_units;
  ptr_block_unit_it_received->last_index_peephole_output_gate =
      this->total_weights;
  //    [1] |END| Output gate. |END|

  for (tmp_ptr_block_ptr_cell_unit_it =
           ptr_block_unit_it_received->ptr_array_cell_units;
       tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit;
       ++tmp_ptr_block_ptr_cell_unit_it) {
    tmp_ptr_block_ptr_cell_unit_it->index_peephole_input_gate =
        ptr_block_unit_it_received->first_index_peephole_input_gate +
        static_cast<size_t>(tmp_ptr_block_ptr_cell_unit_it -
                            tmp_ptr_block_ptr_cell_unit);
    tmp_ptr_block_ptr_cell_unit_it->index_peephole_forget_gate =
        ptr_block_unit_it_received->first_index_peephole_forget_gate +
        static_cast<size_t>(tmp_ptr_block_ptr_cell_unit_it -
                            tmp_ptr_block_ptr_cell_unit);
    tmp_ptr_block_ptr_cell_unit_it->index_peephole_output_gate =
        ptr_block_unit_it_received->first_index_peephole_output_gate +
        static_cast<size_t>(tmp_ptr_block_ptr_cell_unit_it -
                            tmp_ptr_block_ptr_cell_unit);
  }
  // [0] |END| Peepholes. |END|
#endif

  ptr_block_unit_it_received->last_index_connection = this->total_weights;
  // |END| Number connection(s). |END|

  return true;
}

bool Model::Load_Dimension__Cell_Units(
    Layer *const layer_it, CellUnit *&ptr_reference_array_cells_received,
    std::wifstream &file) {
  size_t size_t_input;

  std::wstring line;

  BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
  BlockUnit *tmp_ptr_block_unit_it(layer_it->ptr_array_block_units);

  // Number cell unit(s).
  if ((file >> line) && line.find(L"number_cell_units") == std::wstring::npos) {
    ERR(L"Can not find `number_cell_units` inside \"%ls\".", line.c_str());

    return false;
  } else if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  } else {
    file >> size_t_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }
  }

  size_t const tmp_layer_number_block_units(
      static_cast<size_t>(tmp_ptr_last_block_unit - tmp_ptr_block_unit_it)),
      tmp_block_number_cell_units(size_t_input / tmp_layer_number_block_units);
  // |END| Number cell unit(s). |END|

  layer_it->ptr_array_cell_units = ptr_reference_array_cells_received;

  for (; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit;
       ++tmp_ptr_block_unit_it) {
    tmp_ptr_block_unit_it->ptr_array_cell_units =
        ptr_reference_array_cells_received;
    ptr_reference_array_cells_received += tmp_block_number_cell_units;
    tmp_ptr_block_unit_it->ptr_last_cell_unit =
        ptr_reference_array_cells_received;
  }

  layer_it->ptr_last_cell_unit = ptr_reference_array_cells_received;

  return true;
}

template <class U, LAYER::TYPE const E>
inline bool Model::Load_Dimension__Connection(
    size_t index_received, var *const ptr_array_parameters_received,
    U *const ptr_first_U_unit_received,
    U **ptr_array_ptr_U_unit_connection_received, std::wifstream &file) {
  size_t size_t_input;

  std::wstring line;

  real out;

  file >> line;  // "connected_to_%ls=%u"

  if (file.fail()) {
    ERR(L"Can not read properly inside \"%ls\".", line.c_str());

    return false;
  }

  if (line.find(LAYER_CONN_NAME[E]) !=
      std::wstring::npos)  // If is "connected_to_neuron".
  {
    file >> size_t_input >> std::ws;

    if (file.fail()) {
      ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

      return false;
    }

    ptr_array_ptr_U_unit_connection_received[index_received] =
        ptr_first_U_unit_received + size_t_input;

    if ((file >> line) && line.find(L"weight") == std::wstring::npos) {
      ERR(L"Can not find `weight` inside \"%ls\".", line.c_str());

      return false;
    } else if (file.fail()) {
      ERR(L"Can not read properly inside \"%ls\".", line.c_str());

      return false;
    } else {
      file >> out >> std::ws;
      ptr_array_parameters_received[index_received] = out;

      if (file.fail()) {
        ERR(L"Can not parse line \"%ls\" to the variable.", line.c_str());

        return false;
      }
    }
  } else {
    ERR(L"Can not find `connected_to_neuron` inside \"%ls\".", line.c_str());

    return false;
  }

  return true;
}

template <class U, LAYER::TYPE const E>
bool Model::Load_Dimension__Neuron__Forward__Connection(
    Neuron_unit *const ptr_neuron_received, U *const ptr_first_U_unit_received,
    std::wifstream &file) {
  size_t const tmp_number_connections(
      *ptr_neuron_received->ptr_number_connections);
  size_t tmp_connection_index;

  var *const tmp_ptr_array_parameters(
      this->ptr_array_parameters +
      *ptr_neuron_received->ptr_first_connection_index);

  U **tmp_ptr_array_U_ptr_connections(
      reinterpret_cast<U **>(this->ptr_array_ptr_connections +
                             *ptr_neuron_received->ptr_first_connection_index));

  for (tmp_connection_index = 0_UZ;
       tmp_connection_index != tmp_number_connections; ++tmp_connection_index) {
    if (this->Load_Dimension__Connection<U, E>(
            tmp_connection_index, tmp_ptr_array_parameters,
            ptr_first_U_unit_received, tmp_ptr_array_U_ptr_connections,
            file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Connection(%zu)` function.",
          tmp_connection_index);
      return false;
    }
  }

  return true;
}

template <class U, LAYER::TYPE const E>
bool Model::Load_Dimension__Block__Connection(
    BlockUnit *const ptr_block_unit_it_received,
    U *const ptr_first_U_unit_received, std::wifstream &file) {
  size_t const tmp_number_inputs_connections(
      ptr_block_unit_it_received->last_index_feedforward_connection_input_gate -
      ptr_block_unit_it_received
          ->first_index_feedforward_connection_input_gate),
      tmp_number_recurrents_connection(
          ptr_block_unit_it_received
              ->last_index_recurrent_connection_input_gate -
          ptr_block_unit_it_received
              ->first_index_recurrent_connection_input_gate);
  size_t tmp_connection_index;

  var *tmp_ptr_array_parameters;

  CellUnit const *const tmp_ptr_block_ptr_last_cell_unit(
      ptr_block_unit_it_received->ptr_last_cell_unit);
  CellUnit *tmp_ptr_block_ptr_cell_unit_it,
      **tmp_ptr_array_ptr_connections_layer_cell_units;

#ifndef NO_PEEPHOLE
  CellUnit **tmp_ptr_array_ptr_connections_peepholes_cell_units(
      reinterpret_cast<CellUnit **>(this->ptr_array_ptr_connections));
#endif

  U **tmp_ptr_array_U_ptr_connections;

  // [0] Cell input.
  for (tmp_ptr_block_ptr_cell_unit_it =
           ptr_block_unit_it_received->ptr_array_cell_units;
       tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit;
       ++tmp_ptr_block_ptr_cell_unit_it) {
    //    [1] Input, cell input.
    tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(
        this->ptr_array_ptr_connections +
        tmp_ptr_block_ptr_cell_unit_it
            ->first_index_feedforward_connection_cell_input);

    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_ptr_cell_unit_it
            ->first_index_feedforward_connection_cell_input;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_inputs_connections;
         ++tmp_connection_index) {
      if (this->Load_Dimension__Connection<U, E>(
              tmp_connection_index, tmp_ptr_array_parameters,
              ptr_first_U_unit_received, tmp_ptr_array_U_ptr_connections,
              file) == false) {
        ERR(L"An error has been triggered from the "
            L"`Load_Dimension__Connection(%zu)` function.",
            tmp_connection_index);
        return false;
      }
    }
    //    [1] |END| Input, cell input. |END|

    //    [1] Recurrent, cell input.
    tmp_ptr_array_ptr_connections_layer_cell_units =
        reinterpret_cast<CellUnit **>(
            this->ptr_array_ptr_connections +
            tmp_ptr_block_ptr_cell_unit_it
                ->first_index_recurrent_connection_cell_input);

    tmp_ptr_array_parameters =
        this->ptr_array_parameters +
        tmp_ptr_block_ptr_cell_unit_it
            ->first_index_recurrent_connection_cell_input;

    for (tmp_connection_index = 0_UZ;
         tmp_connection_index != tmp_number_recurrents_connection;
         ++tmp_connection_index) {
      if (this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(
              tmp_connection_index, tmp_ptr_array_parameters,
              this->ptr_array_cell_units,
              tmp_ptr_array_ptr_connections_layer_cell_units, file) == false) {
        ERR(L"An error has been triggered from the "
            L"`Load_Dimension__Connection(%zu)` function.",
            tmp_connection_index);
        return false;
      }
    }
    //    [1] |END| Recurrent, cell input. |END|
  }
  // [0] |END| Cell input. |END|

  // [0] Input, Gates.
  //  [1] Input gate.
  tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(
      this->ptr_array_ptr_connections +
      ptr_block_unit_it_received
          ->first_index_feedforward_connection_input_gate);

  tmp_ptr_array_parameters =
      this->ptr_array_parameters +
      ptr_block_unit_it_received->first_index_feedforward_connection_input_gate;

  for (tmp_connection_index = 0_UZ;
       tmp_connection_index != tmp_number_inputs_connections;
       ++tmp_connection_index) {
    if (this->Load_Dimension__Connection<U, E>(
            tmp_connection_index, tmp_ptr_array_parameters,
            ptr_first_U_unit_received, tmp_ptr_array_U_ptr_connections,
            file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Connection(%zu)` function.",
          tmp_connection_index);
      return false;
    }
  }
  //  [1] |END| Input gate. |END|

  //  [1] Forget gate.
  tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(
      this->ptr_array_ptr_connections +
      ptr_block_unit_it_received
          ->first_index_feedforward_connection_forget_gate);

  tmp_ptr_array_parameters =
      this->ptr_array_parameters +
      ptr_block_unit_it_received
          ->first_index_feedforward_connection_forget_gate;

  for (tmp_connection_index = 0_UZ;
       tmp_connection_index != tmp_number_inputs_connections;
       ++tmp_connection_index) {
    if (this->Load_Dimension__Connection<U, E>(
            tmp_connection_index, tmp_ptr_array_parameters,
            ptr_first_U_unit_received, tmp_ptr_array_U_ptr_connections,
            file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Connection(%zu)` function.",
          tmp_connection_index);
      return false;
    }
  }
  //  [1] |END| Forget gate. |END|

  //  [1] Output gate.
  tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(
      this->ptr_array_ptr_connections +
      ptr_block_unit_it_received
          ->first_index_feedforward_connection_output_gate);

  tmp_ptr_array_parameters =
      this->ptr_array_parameters +
      ptr_block_unit_it_received
          ->first_index_feedforward_connection_output_gate;

  for (tmp_connection_index = 0_UZ;
       tmp_connection_index != tmp_number_inputs_connections;
       ++tmp_connection_index) {
    if (this->Load_Dimension__Connection<U, E>(
            tmp_connection_index, tmp_ptr_array_parameters,
            ptr_first_U_unit_received, tmp_ptr_array_U_ptr_connections,
            file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Connection(%zu)` function.",
          tmp_connection_index);
      return false;
    }
  }
  //  [1] |END| Output gate. |END|
  // [0] |END| Input, Gates. |END|

  // [0] Recurrent, Gates.
  //  [1] Input gate.
  tmp_ptr_array_ptr_connections_layer_cell_units =
      reinterpret_cast<CellUnit **>(
          this->ptr_array_ptr_connections +
          ptr_block_unit_it_received
              ->first_index_recurrent_connection_input_gate);

  tmp_ptr_array_parameters =
      this->ptr_array_parameters +
      ptr_block_unit_it_received->first_index_recurrent_connection_input_gate;

  for (tmp_connection_index = 0_UZ;
       tmp_connection_index != tmp_number_recurrents_connection;
       ++tmp_connection_index) {
    if (this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(
            tmp_connection_index, tmp_ptr_array_parameters,
            this->ptr_array_cell_units,
            tmp_ptr_array_ptr_connections_layer_cell_units, file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Connection(%zu)` function.",
          tmp_connection_index);
      return false;
    }
  }
  //  [1] |END| Input gate. |END|

  //  [1] Forget gate.
  tmp_ptr_array_ptr_connections_layer_cell_units =
      reinterpret_cast<CellUnit **>(
          this->ptr_array_ptr_connections +
          ptr_block_unit_it_received
              ->first_index_recurrent_connection_forget_gate);

  tmp_ptr_array_parameters =
      this->ptr_array_parameters +
      ptr_block_unit_it_received->first_index_recurrent_connection_forget_gate;

  for (tmp_connection_index = 0_UZ;
       tmp_connection_index != tmp_number_recurrents_connection;
       ++tmp_connection_index) {
    if (this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(
            tmp_connection_index, tmp_ptr_array_parameters,
            this->ptr_array_cell_units,
            tmp_ptr_array_ptr_connections_layer_cell_units, file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Connection(%zu)` function.",
          tmp_connection_index);
      return false;
    }
  }
  //  [1] |END| Forget gate. |END|

  //  [1] Output gate.
  tmp_ptr_array_ptr_connections_layer_cell_units =
      reinterpret_cast<CellUnit **>(
          this->ptr_array_ptr_connections +
          ptr_block_unit_it_received
              ->first_index_recurrent_connection_output_gate);

  tmp_ptr_array_parameters =
      this->ptr_array_parameters +
      ptr_block_unit_it_received->first_index_recurrent_connection_output_gate;

  for (tmp_connection_index = 0_UZ;
       tmp_connection_index != tmp_number_recurrents_connection;
       ++tmp_connection_index) {
    if (this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(
            tmp_connection_index, tmp_ptr_array_parameters,
            this->ptr_array_cell_units,
            tmp_ptr_array_ptr_connections_layer_cell_units, file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Connection(%zu)` function.",
          tmp_connection_index);
      return false;
    }
  }
  //  [1] |END| Output gate. |END|
  // [0] |END| Recurrent, Gates. |END|

#ifndef NO_PEEPHOLE
  // [0] Peepholes.
  //  [1] Input gate.
  for (tmp_ptr_block_ptr_cell_unit_it =
           ptr_block_unit_it_received->ptr_array_cell_units;
       tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit;
       ++tmp_ptr_block_ptr_cell_unit_it) {
    if (this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(
            tmp_ptr_block_ptr_cell_unit_it->index_peephole_input_gate,
            this->ptr_array_parameters, this->ptr_array_cell_units,
            tmp_ptr_array_ptr_connections_peepholes_cell_units,
            file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Connection(%zu)` function.",
          tmp_ptr_block_ptr_cell_unit_it->index_peephole_input_gate);
      return false;
    }
  }

  //  [1] Forget gate.
  for (tmp_ptr_block_ptr_cell_unit_it =
           ptr_block_unit_it_received->ptr_array_cell_units;
       tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit;
       ++tmp_ptr_block_ptr_cell_unit_it) {
    if (this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(
            tmp_ptr_block_ptr_cell_unit_it->index_peephole_forget_gate,
            this->ptr_array_parameters, this->ptr_array_cell_units,
            tmp_ptr_array_ptr_connections_peepholes_cell_units,
            file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Connection(%zu)` function.",
          tmp_ptr_block_ptr_cell_unit_it->index_peephole_forget_gate);
      return false;
    }
  }

  //  [1] Output gate.
  for (tmp_ptr_block_ptr_cell_unit_it =
           ptr_block_unit_it_received->ptr_array_cell_units;
       tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit;
       ++tmp_ptr_block_ptr_cell_unit_it) {
    if (this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(
            tmp_ptr_block_ptr_cell_unit_it->index_peephole_output_gate,
            this->ptr_array_parameters, this->ptr_array_cell_units,
            tmp_ptr_array_ptr_connections_peepholes_cell_units,
            file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Connection(%zu)` function.",
          tmp_ptr_block_ptr_cell_unit_it->index_peephole_output_gate);
      return false;
    }
  }
  // [0] |END| Peepholes. |END|
#endif

  return true;
}

template <class U, LAYER::TYPE const E>
bool Model::Load_Dimension__FC(Layer *const layer_it,
                               U *const ptr_first_U_unit_received,
                               std::wifstream &file) {
  Neuron_unit const *const tmp_ptr_last_neuron_unit(
      layer_it->ptr_last_neuron_unit);
  Neuron_unit *tmp_ptr_neuron_unit_it;

  *layer_it->ptr_first_connection_index = this->total_weights;

  // Forward connection.
  for (tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units;
       tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit;
       ++tmp_ptr_neuron_unit_it) {
    if (this->Load_Dimension__Neuron(tmp_ptr_neuron_unit_it, file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Neuron()` function.");
      return false;
    } else if (this->Load_Dimension__Neuron__Forward__Connection<U, E>(
                   tmp_ptr_neuron_unit_it, ptr_first_U_unit_received, file) ==
               false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Neuron__Forward__Connection()` function.");
      return false;
    }
  }

  *layer_it->ptr_last_connection_index = this->total_weights;

  return true;
}

bool Model::Load_Dimension__AF(Layer *const layer_it, std::wifstream &file) {
  AF_unit const *const tmp_ptr_last_AF_unit(layer_it->ptr_last_AF_unit);
  AF_unit *tmp_ptr_AF_unit_it(layer_it->ptr_array_AF_units);

  for (; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it) {
    if (this->Load_Dimension__AF(tmp_ptr_AF_unit_it, file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__AF_Unit()` function.");
      return false;
    }
  }

  return true;
}

bool Model::Load_Dimension__AF_Ind_Recurrent(Layer *const layer_it,
                                             std::wifstream &file) {
  AF_Ind_recurrent_unit const
      *const tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit(
          layer_it->ptr_last_AF_Ind_recurrent_unit);
  AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it(
      layer_it->ptr_array_AF_Ind_recurrent_units);

  for (; tmp_ptr_AF_Ind_recurrent_unit_it !=
         tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit;
       ++tmp_ptr_AF_Ind_recurrent_unit_it) {
    if (this->Load_Dimension__AF_Ind_Recurrent(tmp_ptr_AF_Ind_recurrent_unit_it,
                                               file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__AF_Ind_Recurrent()` function.");
      return false;
    }
  }

  return true;
}

template <class U, LAYER::TYPE const E>
bool Model::Load_Dimension__LSTM(Layer *const layer_it,
                                 U *const ptr_first_U_unit_received,
                                 std::wifstream &file) {
  BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
  BlockUnit *tmp_ptr_block_unit_it(layer_it->ptr_array_block_units);

  size_t const tmp_layer_number_block_units(
      static_cast<size_t>(tmp_ptr_last_block_unit - tmp_ptr_block_unit_it)),
      tmp_layer_number_cell_units(static_cast<size_t>(
          layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units));

  LAYER_NORM::TYPE const tmp_type_layer_normalization(
      layer_it->type_normalization);

  *layer_it->ptr_first_connection_index = this->total_weights;

  for (; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit;
       ++tmp_ptr_block_unit_it) {
    if (this->Load_Dimension__Block(
            tmp_layer_number_block_units,
            tmp_layer_number_cell_units >>
                static_cast<size_t>(layer_it->Use__Bidirectional()),
            tmp_type_layer_normalization, tmp_ptr_block_unit_it,
            file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Block(%zu, %zu, %u, ptr, ref)` function.",
          tmp_layer_number_block_units,
          tmp_layer_number_cell_units >>
              static_cast<size_t>(layer_it->Use__Bidirectional()),
          tmp_type_layer_normalization);
      return false;
    } else if (this->Load_Dimension__Block__Connection<U, E>(
                   tmp_ptr_block_unit_it, ptr_first_U_unit_received, file) ==
               false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__LSTM__Connection()` function.");
      return false;
    }
  }

  *layer_it->ptr_last_connection_index = this->total_weights;

  return true;
}

bool Model::Load_Dimension__Normalization(Layer *const layer_it,
                                          std::wifstream &file) {
  LAYER_NORM::TYPE const tmp_type_layer_normalization(
      layer_it->type_normalization);

  union Normalized_unit const *const tmp_ptr_last_normalized_unit(
      layer_it->ptr_last_normalized_unit);
  union Normalized_unit *tmp_ptr_normalized_unit_it(
      layer_it->ptr_array_normalized_units);

  size_t const tmp_number_normalized_units(static_cast<size_t>(
      tmp_ptr_last_normalized_unit - tmp_ptr_normalized_unit_it));

  for (; tmp_ptr_normalized_unit_it != tmp_ptr_last_normalized_unit;
       ++tmp_ptr_normalized_unit_it) {
    if (this->Load_Dimension__Normalized_Unit(
            tmp_number_normalized_units, tmp_type_layer_normalization,
            tmp_ptr_normalized_unit_it, file) == false) {
      ERR(L"An error has been triggered from the "
          L"`Load_Dimension__Normalized_Unit(%zu, %u)` function.",
          tmp_number_normalized_units, tmp_type_layer_normalization);
      return false;
    }
  }

  return true;
}
}  // namespace DL::v1