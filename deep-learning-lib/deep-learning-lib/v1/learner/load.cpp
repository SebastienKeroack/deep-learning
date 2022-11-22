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

#include <fstream>
#include <iostream>

using namespace DL::File;

namespace DL::v1 {
bool Model::load_spec_params(std::wstring const &path_name)
{
  if(path_exist(path_name) == false)
  {
      ERR(L"Could not find the following path \"%ls\".",
                                path_name.c_str());

      return false;
  }
  else if(recover_temp_file(path_name) == false)
  {
      ERR(L"An error has been triggered from the \"recover_temp_file(%ls)\" function.",
                                path_name.c_str());

      return false;
  }

  std::wifstream file(path_name, std::ios::in | std::ios::binary);

  if(file.is_open() == false)
  {
    ERR(L"The file %ls can not be opened.",
                              path_name.c_str());

    return false;
  }

        if(file.eof())
        {
            ERR(L"File \"%ls\" is empty.",
                                     path_name.c_str());

            return false;
        }

        size_t tmp_input_integer;

        real tmp_input_real;

        std::wstring tmp_line;

        getline(file, tmp_line); // "|===| GRADIENT DESCENT PARAMETERS |===|"

        if((file >> tmp_line) && tmp_line.find(L"learning_rate") == std::wstring::npos)
        {
            ERR(L"Can not find \"learning_rate\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->learning_rate >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"learning_rate_final") == std::wstring::npos)
        {
            ERR(L"Can not find \"learning_rate_final\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->learning_rate_final >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"learning_momentum") == std::wstring::npos)
        {
            ERR(L"Can not find \"learning_momentum\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->learning_momentum >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"learning_gamma") == std::wstring::npos)
        {
            ERR(L"Can not find \"learning_gamma\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->learning_gamma >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"use_nesterov") == std::wstring::npos)
        {
            ERR(L"Can not find \"use_nesterov\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->use_nesterov >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        getline(file, tmp_line); // "|END| GRADIENT DESCENT PARAMETERS |END|"
        getline(file, tmp_line); // CRLF

        getline(file, tmp_line); // "|===| QUICKPROP PARAMETERS |===|"
        
        if((file >> tmp_line) && tmp_line.find(L"quickprop_decay") == std::wstring::npos)
        {
            ERR(L"Can not find \"quickprop_decay\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->quickprop_decay >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"quickprop_mu") == std::wstring::npos)
        {
            ERR(L"Can not find \"quickprop_mu\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->quickprop_mu >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        getline(file, tmp_line); // "|END| QUICKPROP PARAMETERS |END|"
        getline(file, tmp_line); // CRLF

        getline(file, tmp_line); // "|===| RESILLENT PROPAGATION PARAMETERS |===|"
        
        if((file >> tmp_line) && tmp_line.find(L"rprop_increase_factor") == std::wstring::npos)
        {
            ERR(L"Can not find \"rprop_increase_factor\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->rprop_increase_factor >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"rprop_decrease_factor") == std::wstring::npos)
        {
            ERR(L"Can not find \"rprop_decrease_factor\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->rprop_decrease_factor >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"rprop_delta_min") == std::wstring::npos)
        {
            ERR(L"Can not find \"rprop_delta_min\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->rprop_delta_min >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"rprop_delta_max") == std::wstring::npos)
        {
            ERR(L"Can not find \"rprop_delta_max\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->rprop_delta_max >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"rprop_delta_zero") == std::wstring::npos)
        {
            ERR(L"Can not find \"rprop_delta_zero\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->rprop_delta_zero >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        getline(file, tmp_line); // "|END| RESILLENT PROPAGATION PARAMETERS |END|"
        getline(file, tmp_line); // CRLF

        getline(file, tmp_line); // "|===| SARPROP PARAMETERS |===|"
        
        if((file >> tmp_line) && tmp_line.find(L"sarprop_weight_decay_shift") == std::wstring::npos)
        {
            ERR(L"Can not find \"sarprop_weight_decay_shift\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->sarprop_weight_decay_shift >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"sarprop_step_error_threshold_factor") == std::wstring::npos)
        {
            ERR(L"Can not find \"sarprop_step_error_threshold_factor\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->sarprop_step_error_threshold_factor >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"sarprop_step_error_shift") == std::wstring::npos)
        {
            ERR(L"Can not find \"sarprop_step_error_shift\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->sarprop_step_error_shift >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"sarprop_temperature") == std::wstring::npos)
        {
            ERR(L"Can not find \"sarprop_temperature\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->sarprop_temperature >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"sarprop_epoch") == std::wstring::npos)
        {
            ERR(L"Can not find \"sarprop_epoch\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->sarprop_epoch >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        getline(file, tmp_line); // "|END| SARPROP PARAMETERS |END|"
        getline(file, tmp_line); // CRLF

        getline(file, tmp_line); // "|===| ADAM PARAMETERS |===|"
        
        if((file >> tmp_line) && tmp_line.find(L"adam_learning_rate") == std::wstring::npos)
        {
            ERR(L"Can not find \"adam_learning_rate\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->adam_learning_rate >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"adam_beta1") == std::wstring::npos)
        {
            ERR(L"Can not find \"adam_beta1\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->adam_beta1 >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"adam_beta2") == std::wstring::npos)
        {
            ERR(L"Can not find \"adam_beta2\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->adam_beta2 >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"adam_epsilon") == std::wstring::npos)
        {
            ERR(L"Can not find \"adam_epsilon\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->adam_epsilon >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"adam_bias_correction") == std::wstring::npos)
        {
            ERR(L"Can not find \"adam_bias_correction\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->use_adam_bias_correction >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"adam_gamma") == std::wstring::npos)
        {
            ERR(L"Can not find \"adam_gamma\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->adam_gamma >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        getline(file, tmp_line); // "|END| ADAM PARAMETERS |END|"
        getline(file, tmp_line); // CRLF

        getline(file, tmp_line); // "|===| WARM RESTARTS PARAMETERS |===|"
        
        if((file >> tmp_line) && tmp_line.find(L"use_warm_restarts") == std::wstring::npos)
        {
            ERR(L"Can not find \"use_warm_restarts\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->use_warm_restarts >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"warm_restarts_decay_learning_rate") == std::wstring::npos)
        {
            ERR(L"Can not find \"warm_restarts_decay_learning_rate\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->warm_restarts_decay_learning_rate >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"warm_restarts_maximum_learning_rate") == std::wstring::npos)
        {
            ERR(L"Can not find \"warm_restarts_maximum_learning_rate\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->warm_restarts_initial_maximum_learning_rate >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"warm_restarts_minimum_learning_rate") == std::wstring::npos)
        {
            ERR(L"Can not find \"warm_restarts_minimum_learning_rate\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->warm_restarts_minimum_learning_rate >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"warm_restarts_initial_T_i") == std::wstring::npos)
        {
            ERR(L"Can not find \"warm_restarts_initial_T_i\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->warm_restarts_initial_T_i >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"warm_restarts_multiplier") == std::wstring::npos)
        {
            ERR(L"Can not find \"warm_restarts_multiplier\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->warm_restarts_multiplier >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        getline(file, tmp_line); // "|END| WARM RESTARTS PARAMETERS |END|"
        getline(file, tmp_line); // CRLF

        getline(file, tmp_line); // "|===| TRAINING PARAMETERS |===|"
        
        if((file >> tmp_line) && tmp_line.find(L"type_optimizer_function") == std::wstring::npos)
        {
            ERR(L"Can not find \"type_optimizer_function\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_integer;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            if(tmp_input_integer >= static_cast<size_t>(OPTIMIZER::LENGTH))
            {
                ERR(L"Undefined optimization type %zu.",
                                         tmp_input_integer);

                return false;
            }

            this->set_optimizer(static_cast<OPTIMIZER::TYPE>(tmp_input_integer));
        }
        
        if((file >> tmp_line) && tmp_line.find(L"type_loss_function") == std::wstring::npos)
        {
            ERR(L"Can not find \"type_loss_function\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_integer;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            if(tmp_input_integer >= static_cast<size_t>(LOSS_FN::LENGTH))
            {
                ERR(L"Undefined loss function type %zu.",
                                         tmp_input_integer);

                return false;
            }
            
            this->set_loss_fn(static_cast<LOSS_FN::TYPE>(tmp_input_integer));
        }
        
        if((file >> tmp_line) && tmp_line.find(L"type_accuracy_function") == std::wstring::npos)
        {
            ERR(L"Can not find \"type_accuracy_function\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_integer;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            if(tmp_input_integer >= static_cast<size_t>(ACCU_FN::LENGTH))
            {
                ERR(L"Undefined loss function type %zu.",
                                         tmp_input_integer);

                return false;
            }
            
            this->set_accu_fn(static_cast<ACCU_FN::TYPE>(tmp_input_integer));
        }
        
        if((file >> tmp_line) && tmp_line.find(L"bit_fail_limit") == std::wstring::npos)
        {
            ERR(L"Can not find \"bit_fail_limit\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_real >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            this->set_bit_fail_limit(tmp_input_real);
        }
        
        if((file >> tmp_line) && tmp_line.find(L"pre_training_level") == std::wstring::npos)
        {
            ERR(L"Can not find \"pre_training_level\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->pre_training_level >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"use_clip_gradient") == std::wstring::npos)
        {
            ERR(L"Can not find \"use_clip_gradient\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->use_clip_gradient >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"clip_gradient") == std::wstring::npos)
        {
            ERR(L"Can not find \"clip_gradient\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->clip_gradient >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        getline(file, tmp_line); // "|END| TRAINING PARAMETERS |END|"
        getline(file, tmp_line); // CRLF

        getline(file, tmp_line); // "|===| REGULARIZATION PARAMETERS |===|"
        
        if((file >> tmp_line) && tmp_line.find(L"regularization__max_norm_constraints") == std::wstring::npos)
        {
            ERR(L"Can not find \"regularization__max_norm_constraints\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_real >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            this->Set__Regularization__Max_Norm_Constraints(tmp_input_real);
        }
        
        if((file >> tmp_line) && tmp_line.find(L"regularization__l1") == std::wstring::npos)
        {
            ERR(L"Can not find \"regularization__l1\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_real >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            this->Set__Regularization__L1(tmp_input_real);
        }
        
        if((file >> tmp_line) && tmp_line.find(L"regularization__l2") == std::wstring::npos)
        {
            ERR(L"Can not find \"regularization__l2\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_real >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            this->Set__Regularization__L2(tmp_input_real);
        }
        
        if((file >> tmp_line) && tmp_line.find(L"regularization__srip") == std::wstring::npos)
        {
            ERR(L"Can not find \"regularization__srip\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_real >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            this->Set__Regularization__SRIP(tmp_input_real);
        }
        
        if((file >> tmp_line) && tmp_line.find(L"weight_decay") == std::wstring::npos)
        {
            ERR(L"Can not find \"weight_decay\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_real >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            this->Set__Regularization__Weight_Decay(tmp_input_real);
        }
        
        if((file >> tmp_line) && tmp_line.find(L"use_normalized_weight_decay") == std::wstring::npos)
        {
            ERR(L"Can not find \"use_normalized_weight_decay\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->use_normalized_weight_decay >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        getline(file, tmp_line); // "|END| REGULARIZATION PARAMETERS |END|"
        getline(file, tmp_line); // CRLF
        
        getline(file, tmp_line); // "|===| NORMALIZATION PARAMETERS |===|"
        
        if((file >> tmp_line) && tmp_line.find(L"normalization_momentum_average") == std::wstring::npos)
        {
            ERR(L"Can not find \"normalization_momentum_average\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_real >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            this->Set__Normalization_Momentum_Average(tmp_input_real);
        }
        
        if((file >> tmp_line) && tmp_line.find(L"normalization_epsilon") == std::wstring::npos)
        {
            ERR(L"Can not find \"normalization_epsilon\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_real >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            this->Set__Normalization_Epsilon(tmp_input_real);
        }
        
        if((file >> tmp_line) && tmp_line.find(L"batch_renormalization_r_correction_maximum") == std::wstring::npos)
        {
            ERR(L"Can not find \"batch_renormalization_r_correction_maximum\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_real >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            this->Set__Batch_Renormalization_r_Correction_Maximum(tmp_input_real);
        }
        
        if((file >> tmp_line) && tmp_line.find(L"batch_renormalization_d_correction_maximum") == std::wstring::npos)
        {
            ERR(L"Can not find \"batch_renormalization_d_correction_maximum\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_real >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            this->Set__Batch_Renormalization_d_Correction_Maximum(tmp_input_real);
        }
        
        getline(file, tmp_line); // "|END| NORMALIZATION PARAMETERS |END|"
        getline(file, tmp_line); // CRLF

        getline(file, tmp_line); // "|===| LOSS PARAMETERS |===|"
        
        if((file >> tmp_line) && tmp_line.find(L"loss_train") == std::wstring::npos)
        {
            ERR(L"Can not find \"loss_train\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->loss_train >> std::ws;

            if(file.fail())
            {
                file.clear();

                // Inf.
                file >> tmp_line >> std::ws;

                this->loss_train = (std::numeric_limits<real>::max)();
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"loss_valid") == std::wstring::npos)
        {
            ERR(L"Can not find \"loss_valid\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->loss_valid >> std::ws;

            if(file.fail())
            {
                file.clear();

                // Inf.
                file >> tmp_line >> std::ws;

                this->loss_valid = (std::numeric_limits<real>::max)();
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"loss_testg") == std::wstring::npos)
        {
            ERR(L"Can not find \"loss_testg\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->loss_testg >> std::ws;

            if(file.fail())
            {
                file.clear();
                
                // Inf.
                file >> tmp_line >> std::ws;

                this->loss_testg = (std::numeric_limits<real>::max)();
            }
        }
        
        getline(file, tmp_line); // "|END| LOSS PARAMETERS |END|"
        getline(file, tmp_line); // CRLF

        getline(file, tmp_line); // "|===| ACCURANCY PARAMETERS |===|"
        
        if((file >> tmp_line) && tmp_line.find(L"acc_var") == std::wstring::npos)
        {
            ERR(L"Can not find \"acc_var\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_real >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            this->Set__Accurancy_Variance(tmp_input_real);
        }
        
        if((file >> tmp_line) && tmp_line.find(L"acc_train") == std::wstring::npos)
        {
            ERR(L"Can not find \"acc_train\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->acc_train >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"acc_valid") == std::wstring::npos)
        {
            ERR(L"Can not find \"acc_valid\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->acc_valid >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"acc_testg") == std::wstring::npos)
        {
            ERR(L"Can not find \"acc_testg\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->acc_testg >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        getline(file, tmp_line); // "|END| ACCURANCY PARAMETERS |END|"
        getline(file, tmp_line); // CRLF

        getline(file, tmp_line); // "|===| COMPUTATION PARAMETERS |===|"
        
        if((file >> tmp_line) && tmp_line.find(L"use_cu") == std::wstring::npos)
        {
            ERR(L"Can not find \"use_cu\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->use_cu >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"use_mp") == std::wstring::npos)
        {
            ERR(L"Can not find \"use_mp\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->use_mp >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"pct_threads") == std::wstring::npos)
        {
            ERR(L"Can not find \"pct_threads\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->pct_threads >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"maximum_batch_size") == std::wstring::npos)
        {
            ERR(L"Can not find \"maximum_batch_size\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> this->maximum_batch_size >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
        }
        
        getline(file, tmp_line); // "|END| COMPUTATION PARAMETERS |END|"

        if(file.fail())
        {
            ERR(L"Logical error on i/o operation \"%ls\".",
                                     path_name.c_str());

            return false;
        }

        file.close();

    return true;
}

bool Model::load(std::wstring const &path_params,
                                        std::wstring const &path_spec_params,
                                        size_t const allowable_memory)
{
  if(path_exist(path_spec_params) == false)
  {
      ERR(L"Could not find the following path \"%ls\".",
                                path_spec_params.c_str());
      return false;
  }
  else if(path_exist(path_params) == false)
  {
      ERR(L"Could not find the following path \"%ls\".",
                                path_params.c_str());
      return false;
  }
  else if(recover_temp_file(path_params) == false)
  {
      ERR(L"An error has been triggered from the \"recover_temp_file(%ls)\" function.",
                                path_params.c_str());
      return false;
  }

  std::wifstream file(path_params, std::ios::in | std::ios::binary);

  if(file.is_open() == false)
  {
    ERR(L"The file %ls can not be opened.",
                              path_params.c_str());
    return false;
  }

        if(file.eof())
        {
            ERR(L"File \"%ls\" is empty.",
                                     path_params.c_str());

            file.close();

            return false;
        }
        
        this->clear();

        bool tmp_input_boolean;

        size_t tmp_state_layer_index(0_UZ),
                  tmp_input_integer;
        
        real tmp_input_T[2] = {0};

        std::wstring tmp_line;
        
        auto load_dropout_params_fn([self = this, &file = file](Layer *const layer_it, bool const is_hidden_layer_received = true) -> bool
        {
            size_t tmp_input_integer;

            real tmp_dropout_values[3] = {0};

            LAYER_DROPOUT::TYPE tmp_type_layer_dropout;
            
            std::wstring tmp_line;
            
            if((file >> tmp_line) && tmp_line.find(L"type_dropout") == std::wstring::npos)
            {
                ERR(L"Can not find \"type_dropout\" inside \"%ls\".",
                                            tmp_line.c_str());

                return false;
            }
            else if(file.fail())
            {
                ERR(L"Can not read properly inside \"%ls\".",
                                            tmp_line.c_str());

                return false;
            }
            else
            {
                file >> tmp_input_integer;

                if(file.fail())
                {
                    ERR(L"Can not read input of \"%ls\".",
                                                tmp_line.c_str());

                    return false;
                }

                if(tmp_input_integer >= static_cast<size_t>(LAYER_DROPOUT::LENGTH))
                {
                    ERR(L"Undefined layer dropout type %zu.",
                                                tmp_input_integer);

                    return false;
                }

                tmp_type_layer_dropout = static_cast<LAYER_DROPOUT::TYPE>(tmp_input_integer);
            }
            
            if(is_hidden_layer_received)
            {
                if((file >> tmp_line) && tmp_line.find(L"use_coded_dropout") == std::wstring::npos)
                {
                    ERR(L"Can not find \"use_coded_dropout\" inside \"%ls\".",
                                                tmp_line.c_str());

                    return false;
                }
                else if(file.fail())
                {
                    ERR(L"Can not read properly inside \"%ls\".",
                                                tmp_line.c_str());
            
                    return false;
                }
                else
                {
                    file >> layer_it->use_coded_dropout >> std::ws;

                    if(file.fail())
                    {
                        ERR(L"Can not read input of \"%ls\".",
                                                    tmp_line.c_str());
                
                        return false;
                    }
                }
            }

            if((file >> tmp_line) && tmp_line.find(L"dropout_values[0]") == std::wstring::npos)
            {
                ERR(L"Can not find \"dropout_values[0]\" inside \"%ls\".",
                                            tmp_line.c_str());

                return false;
            }
            else if(file.fail())
            {
                ERR(L"Can not read properly inside \"%ls\".",
                                            tmp_line.c_str());
            
                return false;
            }
            else
            {
                file >> tmp_dropout_values[0] >> std::ws;

                if(file.fail())
                {
                    ERR(L"Can not read input of \"%ls\".",
                                                tmp_line.c_str());
                
                    return false;
                }
            }
            
            if((file >> tmp_line) && tmp_line.find(L"dropout_values[1]") == std::wstring::npos)
            {
                ERR(L"Can not find \"dropout_values[1]\" inside \"%ls\".",
                                            tmp_line.c_str());

                return false;
            }
            else if(file.fail())
            {
                ERR(L"Can not read properly inside \"%ls\".",
                                            tmp_line.c_str());
            
                return false;
            }
            else
            {
                file >> tmp_dropout_values[1] >> std::ws;

                if(file.fail())
                {
                    ERR(L"Can not read input of \"%ls\".",
                                                tmp_line.c_str());
                
                    return false;
                }
            }
            
            if((file >> tmp_line) && tmp_line.find(L"dropout_values[2]") == std::wstring::npos)
            {
                ERR(L"Can not find \"dropout_values[2]\" inside \"%ls\".",
                                            tmp_line.c_str());

                return false;
            }
            else if(file.fail())
            {
                ERR(L"Can not read properly inside \"%ls\".",
                                            tmp_line.c_str());
            
                return false;
            }
            else
            {
                file >> tmp_dropout_values[2] >> std::ws;

                if(file.fail())
                {
                    ERR(L"Can not read input of \"%ls\".",
                                                tmp_line.c_str());
                
                    return false;
                }
            }
            
            if(self->type == MODEL::AUTOENCODER
              &&
               (layer_it == self->ptr_last_layer - (self->total_layers - 3_UZ) / 2_UZ + 2_UZ
                    ||
                layer_it >= self->ptr_last_layer - (self->total_layers - 3_UZ) / 2_UZ + 1_UZ))
            { return true; }

            if(self->set_dropout(layer_it,
                                           tmp_type_layer_dropout,
                                           tmp_dropout_values,
                                           false) == false)
            {
                ERR(L"An error has been triggered from the \"set_dropout(ptr, %u, %f, %f)\" function.",
                                            tmp_type_layer_dropout,
                                            tmp_dropout_values[0],
                                            tmp_dropout_values[1]);

                return false;
            }

            return true;
        });
        
        auto tmp_Valid__Layer__Normalization([self = this](Layer *const layer_it) -> bool
        {
            if(self->type == MODEL::AUTOENCODER
              &&
              layer_it >= self->ptr_last_layer - (self->total_layers - 3_UZ) / 2_UZ + 1_UZ)
            { return false; }

            return true;
        });

        INFO(L"");
        INFO(L"Load params `%ls`.", path_params.c_str());

        getline(file, tmp_line); // "|===| DIMENSION |===|"

        if((file >> tmp_line) && tmp_line.find(L"type") == std::wstring::npos)
        {
            ERR(L"Can not find \"type\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_integer;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            if(tmp_input_integer >= static_cast<size_t>(MODEL::LENGTH))
            {
                ERR(L"Undefined network type %zu.",
                                         tmp_input_integer);

                return false;
            }

            this->type = static_cast<MODEL::TYPE>(tmp_input_integer);
        }
        
        if((file >> tmp_line) && tmp_line.find(L"number_layers") == std::wstring::npos)
        {
            ERR(L"Can not find \"number_layers\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_integer >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
            else if(tmp_input_integer < 2)
            {
                ERR(L"The number of layers is set too small.",);

                return false;
            }
        }
        
        // allocate structure.
        INFO(L"allocate %zu layer(s).",
                                 tmp_input_integer);
        if(this->Allocate__Structure(tmp_input_integer, allowable_memory) == false)
        {
            ERR(L"An error has been triggered from the \"Allocate__Structure(%zu, %zu)\" function.",
                                     tmp_input_integer,
                                     allowable_memory);

            return false;
        }

        if((file >> tmp_line) && tmp_line.find(L"seq_w") == std::wstring::npos)
        {
            ERR(L"Can not find \"seq_w\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->seq_w >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"n_time_delay") == std::wstring::npos)
        {
            ERR(L"Can not find \"n_time_delay\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->n_time_delay >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"use_first_layer_as_input") == std::wstring::npos)
        {
            ERR(L"Can not find \"use_first_layer_as_input\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> tmp_input_boolean >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }

            if(this->Set__Input_Mode(tmp_input_boolean) == false)
            {
                ERR(L"An error has been triggered from the \"Set__Input_Mode(%ls)\" function.",
                                         tmp_input_boolean ? "true" : "false");

                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"use_last_layer_as_output") == std::wstring::npos)
        {
            ERR(L"Can not find \"use_last_layer_as_output\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> tmp_input_boolean >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }

            if(this->Set__Output_Mode(tmp_input_boolean) == false)
            {
                ERR(L"An error has been triggered from the \"Set__Output_Mode(%ls)\" function.",
                                         tmp_input_boolean ? "true" : "false");

                return false;
            }
        }
        
        Layer const *const tmp_ptr_first_layer(this->ptr_array_layers),
                                   *const last_layer(this->ptr_last_layer - 1), // Subtract output layer.
                                   *tmp_ptr_previous_layer,
                                   *tmp_ptr_layer_state(nullptr);
        Layer *layer_it(this->ptr_array_layers);
        // |END| allocate structure. |END|
        
        // allocate basic unit(s).
        if((file >> tmp_line) && tmp_line.find(L"total_basic_units") == std::wstring::npos)
        {
            ERR(L"Can not find \"total_basic_units\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->total_basic_units >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        layer_it->ptr_array_basic_units = nullptr;
        layer_it->ptr_last_basic_unit = layer_it->ptr_array_basic_units + this->total_basic_units;
        
        INFO(L"allocate %zu basic unit(s).",
                                 this->total_basic_units);
        if(this->Allocate__Basic_Units() == false)
        {
            ERR(L"An error has been triggered from the \"Allocate__Basic_Units()\" function.",);

            return false;
        }
        // |END| allocate basic unit(s). |END|
        
        // allocate basic indice unit(s).
        if((file >> tmp_line) && tmp_line.find(L"total_basic_indice_units") == std::wstring::npos)
        {
            ERR(L"Can not find \"total_basic_indice_units\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->total_basic_indice_units >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        layer_it->ptr_array_basic_indice_units = nullptr;
        layer_it->ptr_last_basic_indice_unit = layer_it->ptr_array_basic_indice_units + this->total_basic_indice_units;
        
        INFO(L"allocate %zu basic indice unit(s).",
                                 this->total_basic_indice_units);
        if(this->Allocate__Basic_Indice_Units() == false)
        {
            ERR(L"An error has been triggered from the \"Allocate__Basic_Indice_Units()\" function.",);

            return false;
        }
        // |END| allocate basic indice unit(s). |END|

        // allocate neuron unit(s).
        if((file >> tmp_line) && tmp_line.find(L"total_neuron_units") == std::wstring::npos)
        {
            ERR(L"Can not find \"total_neuron_units\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->total_neuron_units >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        layer_it->ptr_array_neuron_units = nullptr;
        layer_it->ptr_last_neuron_unit = layer_it->ptr_array_neuron_units + this->total_neuron_units;
        
        INFO(L"allocate %zu neuron unit(s).",
                                 this->total_neuron_units);
        if(this->Allocate__Neuron_Units() == false)
        {
            ERR(L"An error has been triggered from the \"Allocate__Neuron_Units()\" function.",);

            return false;
        }
        // |END| allocate neuron unit(s). |END|
        
        // allocate AF unit(s).
        if((file >> tmp_line) && tmp_line.find(L"total_AF_units") == std::wstring::npos)
        {
            ERR(L"Can not find \"total_AF_units\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->total_AF_units >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        layer_it->ptr_array_AF_units = nullptr;
        layer_it->ptr_last_AF_unit = layer_it->ptr_array_AF_units + this->total_AF_units;
        
        INFO(L"allocate %zu AF unit(s).",
                                 this->total_AF_units);
        if(this->Allocate__AF_Units() == false)
        {
            ERR(L"An error has been triggered from the \"Allocate__AF_Units()\" function.",);

            return false;
        }
        // |END| allocate AF unit(s). |END|
        
        // allocate af_ind unit(s).
        if((file >> tmp_line) && tmp_line.find(L"total_AF_Ind_recurrent_units") == std::wstring::npos)
        {
            ERR(L"Can not find \"total_AF_Ind_recurrent_units\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->total_AF_Ind_recurrent_units >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        layer_it->ptr_array_AF_Ind_recurrent_units = nullptr;
        layer_it->ptr_last_AF_Ind_recurrent_unit = layer_it->ptr_array_AF_Ind_recurrent_units + this->total_AF_Ind_recurrent_units;
        
        INFO(L"allocate %zu AF Ind recurrent unit(s).",
                                 this->total_AF_Ind_recurrent_units);
        if(this->Allocate__AF_Ind_Recurrent_Units() == false)
        {
            ERR(L"An error has been triggered from the \"Allocate__AF_Ind_Recurrent_Units()\" function.",);

            return false;
        }
        // |END| allocate af_ind unit(s). |END|

        // allocate block/cell unit(s).
        if((file >> tmp_line) && tmp_line.find(L"total_block_units") == std::wstring::npos)
        {
            ERR(L"Can not find \"total_block_units\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->total_block_units >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        layer_it->ptr_array_block_units = nullptr;
        layer_it->ptr_last_block_unit = layer_it->ptr_array_block_units + this->total_block_units;
        
        if((file >> tmp_line) && tmp_line.find(L"total_cell_units") == std::wstring::npos)
        {
            ERR(L"Can not find \"total_cell_units\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->total_cell_units >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        layer_it->ptr_array_cell_units = nullptr;
        layer_it->ptr_last_cell_unit = layer_it->ptr_array_cell_units + this->total_cell_units;

        if(this->total_block_units != 0_UZ && this->total_cell_units != 0_UZ)
        {
            INFO(L"allocate %zu block unit(s).",
                                     this->total_block_units);
            INFO(L"allocate %zu cell unit(s).",
                                     this->total_cell_units);
            if(this->Allocate__LSTM_Layers() == false)
            {
                ERR(L"An error has been triggered from the \"Allocate__LSTM_Layers()\" function.",);

                return false;
            }
        }
        // |END| allocate block/cell unit(s). |END|
        
        // allocate normalized unit(s).
        if((file >> tmp_line) && tmp_line.find(L"total_normalized_units") == std::wstring::npos)
        {
            ERR(L"Can not find \"total_normalized_units\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->total_normalized_units >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        INFO(L"allocate %zu normalized unit(s).",
                                 this->total_normalized_units);
        if(this->Allocate__Normalized_Unit(false) == false)
        {
            ERR(L"An error has been triggered from the \"Allocate__Normalized_Unit(false)\" function.",);

            return false;
        }
        // |END| allocate normalized unit(s). |END|

        // allocate parameter(s).
        if((file >> tmp_line) && tmp_line.find(L"total_parameters") == std::wstring::npos)
        {
            ERR(L"Can not find \"total_parameters\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->total_parameters >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"total_weights") == std::wstring::npos)
        {
            ERR(L"Can not find \"total_weights\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->total_weights >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        if((file >> tmp_line) && tmp_line.find(L"total_bias") == std::wstring::npos)
        {
            ERR(L"Can not find \"total_bias\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> this->total_bias >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }
        }
        
        INFO(L"allocate %zu parameter(s).",
                                 this->total_parameters);
        if(this->Allocate__Parameter() == false)
        {
            ERR(L"An error has been triggered from the \"Allocate__Parameter()\" function.",);

            return false;
        }
        // |END| allocate parameter(s). |END|

        // Initialize layer(s).
        // reset number of weights to zero. Increment the variable inside the loading layer.
        this->total_weights = 0_UZ;
        this->total_bias = 0_UZ;
        
        Basic_unit *tmp_ptr_array_basic_units(this->ptr_array_basic_units);
        
        Basic_indice_unit *tmp_ptr_array_basic_indice_units(this->ptr_array_basic_indice_units);

        Neuron_unit *tmp_ptr_array_neuron_units(this->ptr_array_neuron_units);
        
        AF_unit *tmp_ptr_array_AF_units(this->ptr_array_AF_units);

        AF_Ind_recurrent_unit *tmp_ptr_array_AF_Ind_recurrent_units(this->ptr_array_AF_Ind_recurrent_units);

        BlockUnit *tmp_ptr_array_block_units(this->ptr_array_block_units);

        CellUnit *tmp_ptr_array_cell_units(this->ptr_array_cell_units);
        
        union Normalized_unit *tmp_ptr_array_normalized_units(this->ptr_array_normalized_units);

        // Input layer.
        //  Type layer.
        INFO(L"load input layer.");
        getline(file, tmp_line); // "Input layer:"
        
        if((file >> tmp_line) && tmp_line.find(L"type_layer") == std::wstring::npos)
        {
            ERR(L"Can not find \"type_layer\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_integer;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            if(tmp_input_integer >= static_cast<size_t>(LAYER::LENGTH))
            {
                ERR(L"Undefined layer type %zu.",
                                         tmp_input_integer);

                return false;
            }

            layer_it->type_layer = static_cast<LAYER::TYPE>(tmp_input_integer);
        }
        //  |END| Type layer. |END|
        
        //  Type activation.
        if((file >> tmp_line) && tmp_line.find(L"type_activation") == std::wstring::npos)
        {
            ERR(L"Can not find \"type_activation\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else
        {
            file >> tmp_input_integer;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }

            if(tmp_input_integer >= static_cast<size_t>(LAYER_ACTIVATION::LENGTH))
            {
                ERR(L"Undefined layer activation type %zu.",
                                         tmp_input_integer);

                return false;
            }

            layer_it->type_activation = static_cast<LAYER_ACTIVATION::TYPE>(tmp_input_integer);
        }
        //  |END| Type activation. |END|
        
        //  Dropout.
        if(load_dropout_params_fn(layer_it, false) == false)
        {
            ERR(L"An error has been triggered from the \"load_dropout_params_fn(false)\" function.",);

            return false;
        }
        //  |END| Dropout. |END|
        
        //  Initialize input(s).
        if((file >> tmp_line) && tmp_line.find(L"n_inp") == std::wstring::npos)
        {
            ERR(L"Can not find \"n_inp\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            return false;
        }
        else
        {
            file >> tmp_input_integer >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                return false;
            }

            *layer_it->ptr_number_outputs = this->n_inp = tmp_input_integer;

            layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
            tmp_ptr_array_neuron_units += tmp_input_integer;
            layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
            
            this->Order__Layer__Neuron(layer_it);
        }
        //  |END| Initialize input(s). |END|
        
        //  Initialize normalized unit(s).
        layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
        if(this->total_normalized_units_allocated != 0_UZ) { tmp_ptr_array_normalized_units += *layer_it->ptr_number_outputs; } // If use normalization.
        layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
        //  |END| Initialize normalized unit(s). |END|
        
        // Initialize AF unit(s).
        layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
        tmp_ptr_array_AF_units += *layer_it->ptr_number_outputs;
        layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
        // |END| Initialize AF unit(s). |END|
        
        // Initialize AF Ind recurrent unit(s).
        layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
        layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
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
        for(++layer_it; layer_it != last_layer; ++layer_it)
        {
            // Type layer.
            getline(file, tmp_line); // "Hidden layer %u:"
            
            if((file >> tmp_line) && tmp_line.find(L"type_layer") == std::wstring::npos)
            {
                ERR(L"Can not find \"type_layer\" inside \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
            else if(file.fail())
            {
                ERR(L"Can not read properly inside \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
            else
            {
                file >> tmp_input_integer;

                if(file.fail())
                {
                    ERR(L"Can not read input of \"%ls\".",
                                             tmp_line.c_str());

                    return false;
                }

                if(tmp_input_integer >= static_cast<size_t>(LAYER::LENGTH))
                {
                    ERR(L"Undefined layer type %zu.",
                                             tmp_input_integer);

                    return false;
                }

                layer_it->type_layer = static_cast<LAYER::TYPE>(tmp_input_integer);
            }
            
            INFO(L"load hidden layer %zu (%ls | %u).",
                                    static_cast<size_t>(layer_it - tmp_ptr_first_layer),
                                    LAYER_NAME[layer_it->type_layer].c_str(),
                                    layer_it->type_layer);
            // |END| Type layer. |END|
            
            this->Organize__Previous_Layers_Connected(tmp_state_layer_index,
                                                                               layer_it,
                                                                               tmp_ptr_layer_state);

            tmp_ptr_previous_layer = layer_it->previous_connected_layers[0];

            // Use bidirectional.
            if((file >> tmp_line) && tmp_line.find(L"use_bidirectional") == std::wstring::npos)
            {
                ERR(L"Can not find \"use_bidirectional\" inside \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
            else if(file.fail())
            {
                ERR(L"Can not read properly inside \"%ls\".",
                                         tmp_line.c_str());

                return false;
            }
            else
            {
                file >> layer_it->use_bidirectional;

                if(file.fail())
                {
                    ERR(L"Can not read input of \"%ls\".",
                                             tmp_line.c_str());

                    return false;
                }
            }
            // |END| Use bidirectional. |END|
            
            switch(layer_it->type_layer)
            {
                case LAYER::AVERAGE_POOLING:
                    // Pooling.
                    if((file >> tmp_line) && tmp_line.find(L"kernel_size") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"kernel_size\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->pooling_values[0] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"stride") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"stride\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->pooling_values[1] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"padding") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"padding\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->pooling_values[2] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"dilation") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"dilation\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->pooling_values[3] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    // |END| Pooling. |END|

                    //  Initialize normalized unit(s).
                    layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                    layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                    //  |END| Initialize normalized unit(s). |END|
                    
                    // Initialize basic unit(s).
                    if((file >> tmp_line) && tmp_line.find(L"number_basic_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_basic_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> *layer_it->ptr_number_outputs >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }

                        layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
                        tmp_ptr_array_basic_units += *layer_it->ptr_number_outputs;
                        layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;

                        this->Order__Layer__Basic(layer_it);
                    }
                    // |END| Initialize basic unit(s). |END|
                    
                    //  Initialize basic indice unit(s).
                    layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
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
                    layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                    layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
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
                    if((file >> tmp_line) && tmp_line.find(L"type_activation") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"type_activation\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());

                            return false;
                        }

                        if(tmp_input_integer >= static_cast<size_t>(LAYER_ACTIVATION::LENGTH))
                        {
                            ERR(L"Undefined layer activation type %zu.",
                                                     tmp_input_integer);

                            return false;
                        }

                        layer_it->type_activation = static_cast<LAYER_ACTIVATION::TYPE>(tmp_input_integer);
                    }
                    // |END| Type activation. |END|

                    // Dropout.
                    if(load_dropout_params_fn(layer_it) == false)
                    {
                        ERR(L"An error has been triggered from the \"load_dropout_params_fn()\" function.",);

                        return false;
                    }
                    // |END| Dropout. |END|
                    
                    // Normalization.
                    if((file >> tmp_line) && tmp_line.find(L"type_normalization") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"type_normalization\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());

                            return false;
                        }

                        if(tmp_input_integer >= static_cast<size_t>(LAYER_NORM::LENGTH))
                        {
                            ERR(L"Undefined layer normalization type %zu.",
                                                     tmp_input_integer);

                            return false;
                        }
                        
                        if(tmp_Valid__Layer__Normalization(layer_it)
                          &&
                          this->Set__Layer_Normalization(layer_it,
                                                                         static_cast<LAYER_NORM::TYPE>(tmp_input_integer),
                                                                         false,
                                                                         false) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Layer_Normalization(%zu)\" function.",
                                                     tmp_input_integer);
                            
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"use_layer_normalization_before_activation") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"use_layer_normalization_before_activation\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->use_layer_normalization_before_activation >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"number_normalized_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_normalized_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }
                        
                        layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                        tmp_ptr_array_normalized_units += tmp_input_integer;
                        layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                        
                        this->Order__Layer__Normalization(layer_it);

                        if(this->Load_Dimension__Normalization(layer_it, file) == false)
                        {
                            ERR(L"An error has been triggered from the \"Load_Dimension__Normalization()\" function.",);

                            return false;
                        }
                    }
                    // |END| Normalization. |END|

                    if((file >> tmp_line) && tmp_line.find(L"use_tied_parameter") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"use_tied_parameter\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_boolean >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__Tied_Parameter(layer_it,
                                                              tmp_input_boolean,
                                                              false) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Tied_Parameter(ptr, %ls, false)\" function.",
                                                     tmp_input_boolean ? "true" : "false");

                            return false;
                        }
                    }
                    
                    // k-Sparse filters.
                    if((file >> tmp_line) && tmp_line.find(L"k_sparsity") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"k_sparsity\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__K_Sparsity(layer_it, tmp_input_integer) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__K_Sparsity(ptr, %zu)\" function.",
                                                     tmp_input_integer);

                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"alpha_sparsity") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"alpha_sparsity\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_T[0] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__Alpha_Sparsity(layer_it, tmp_input_T[0]) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Alpha_Sparsity(ptr, %f)\" function.",
                                                     tmp_input_T[0]);

                            return false;
                        }
                    }
                    // |END| k-Sparse filters. |END|
                    
                    // Constraint.
                    if((file >> tmp_line) && tmp_line.find(L"constraint_recurrent_weight_lower_bound") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"constraint_recurrent_weight_lower_bound\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_T[0] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"constraint_recurrent_weight_upper_bound") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"constraint_recurrent_weight_upper_bound\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_T[1] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__Regularization__Constraint_Recurrent_Weight(layer_it,
                                                                                       tmp_input_T[0],
                                                                                       tmp_input_T[1]) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Regularization__Constraint_Recurrent_Weight(ptr, %f, %f)\" function.",
                                                     tmp_input_T[0],
                                                     tmp_input_T[1]);

                            return false;
                        }
                    }
                    // |END| Constraint. |END|
                    
                    // Initialize basic unit(s).
                    layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
                    layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
                    // |END| Initialize basic unit(s). |END|
                    
                    // Initialize basic indice unit(s).
                    layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
                    layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
                    // |END| Initialize basic indice unit(s). |END|
                    
                    // Initialize neuron unit(s).
                    if((file >> tmp_line) && tmp_line.find(L"number_neuron_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_neuron_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> *layer_it->ptr_number_outputs >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }
                        
                        layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
                        tmp_ptr_array_neuron_units += *layer_it->ptr_number_outputs;
                        layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;

                        this->Order__Layer__Neuron(layer_it);

                        switch(tmp_ptr_previous_layer->type_layer)
                        {
                            case LAYER::AVERAGE_POOLING:
                            case LAYER::RESIDUAL:
                                if(this->Load_Dimension__FC<Basic_unit, LAYER::AVERAGE_POOLING>(layer_it,
                                                                                                                                                                                                                            this->ptr_array_basic_units,
                                                                                                                                                                                                                            file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                                    
                                    file.close();

                                    return false;
                                }
                                    break;
                            case LAYER::FULLY_CONNECTED:
                            case LAYER::FULLY_CONNECTED_RECURRENT:
                            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                                if(this->Load_Dimension__FC<Neuron_unit, LAYER::FULLY_CONNECTED>(layer_it,
                                                                                                                                                                                                                              this->ptr_array_neuron_units,
                                                                                                                                                                                                                              file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                                    
                                    file.close();

                                    return false;
                                }
                                    break;
                            case LAYER::LSTM:
                                if(this->Load_Dimension__FC<CellUnit, LAYER::LSTM>(layer_it,
                                                                                                                                                                                                  this->ptr_array_cell_units,
                                                                                                                                                                                                  file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                                    
                                    file.close();

                                    return false;
                                }
                                    break;
                            case LAYER::MAX_POOLING:
                                if(this->Load_Dimension__FC<Basic_indice_unit, LAYER::MAX_POOLING>(layer_it,
                                                                                                                                                                                                                              this->ptr_array_basic_indice_units,
                                                                                                                                                                                                                              file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                                    
                                    file.close();

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
                    if((file >> tmp_line) && tmp_line.find(L"number_AF_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_AF_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }
                        
                        layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
                        tmp_ptr_array_AF_units += tmp_input_integer;
                        layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
                        
                        this->Order__Layer__AF(layer_it);

                        if(this->Load_Dimension__AF(layer_it, file) == false)
                        {
                            ERR(L"An error has been triggered from the \"Load_Dimension__AF()\" function.",);

                            return false;
                        }
                    }
                    // |END| Initialize AF unit(s). |END|
                    
                    // Initialize AF Ind recurrent unit(s).
                    layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                    layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
                    // |END| Initialize AF Ind recurrent unit(s). |END|
                    
                    //  Initialize block/cell unit(s).
                    layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
                    layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

                    layer_it->ptr_array_cell_units = tmp_ptr_array_cell_units;
                    layer_it->ptr_last_cell_unit = tmp_ptr_array_cell_units;
                    //  |END| Initialize block/cell unit(s). |END|
                    
                    // Initialize bias parameter(s).
                    if(this->Load_Dimension__Bias(layer_it, file) == false)
                    {
                        ERR(L"An error has been triggered from the \"Load_Dimension__Bias()\" function.",);

                        return false;
                    }
                    // |END| Initialize bias parameter(s). |END|
                        break;
                case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    // Type activation.
                    if((file >> tmp_line) && tmp_line.find(L"type_activation") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"type_activation\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());

                            return false;
                        }

                        if(tmp_input_integer >= static_cast<size_t>(LAYER_ACTIVATION::LENGTH))
                        {
                            ERR(L"Undefined layer activation type %zu.",
                                                     tmp_input_integer);

                            return false;
                        }

                        layer_it->type_activation = static_cast<LAYER_ACTIVATION::TYPE>(tmp_input_integer);
                    }
                    // |END| Type activation. |END|

                    // Dropout.
                    if(load_dropout_params_fn(layer_it) == false)
                    {
                        ERR(L"An error has been triggered from the \"load_dropout_params_fn()\" function.",);

                        return false;
                    }
                    // |END| Dropout. |END|
                    
                    // Normalization.
                    if((file >> tmp_line) && tmp_line.find(L"type_normalization") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"type_normalization\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());

                            return false;
                        }

                        if(tmp_input_integer >= static_cast<size_t>(LAYER_NORM::LENGTH))
                        {
                            ERR(L"Undefined layer normalization type %zu.",
                                                     tmp_input_integer);

                            return false;
                        }
                        
                        if(this->Set__Layer_Normalization(layer_it,
                                                                          static_cast<LAYER_NORM::TYPE>(tmp_input_integer),
                                                                          false,
                                                                          false) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Layer_Normalization(%zu)\" function.",
                                                     tmp_input_integer);
                            
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"use_layer_normalization_before_activation") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"use_layer_normalization_before_activation\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->use_layer_normalization_before_activation >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"number_normalized_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_normalized_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }
                        
                        layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                        tmp_ptr_array_normalized_units += tmp_input_integer;
                        layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                        
                        this->Order__Layer__Normalization(layer_it);

                        if(this->Load_Dimension__Normalization(layer_it, file) == false)
                        {
                            ERR(L"An error has been triggered from the \"Load_Dimension__Normalization()\" function.",);

                            return false;
                        }
                    }
                    // |END| Normalization. |END|

                    if((file >> tmp_line) && tmp_line.find(L"use_tied_parameter") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"use_tied_parameter\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_boolean >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__Tied_Parameter(layer_it,
                                                              tmp_input_boolean,
                                                              false) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Tied_Parameter(ptr, %ls, false)\" function.",
                                                     tmp_input_boolean ? "true" : "false");

                            return false;
                        }
                    }
                    
                    // k-Sparse filters.
                    if((file >> tmp_line) && tmp_line.find(L"k_sparsity") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"k_sparsity\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__K_Sparsity(layer_it, tmp_input_integer) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__K_Sparsity(ptr, %zu)\" function.",
                                                     tmp_input_integer);

                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"alpha_sparsity") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"alpha_sparsity\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_T[0] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__Alpha_Sparsity(layer_it, tmp_input_T[0]) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Alpha_Sparsity(ptr, %f)\" function.",
                                                     tmp_input_T[0]);

                            return false;
                        }
                    }
                    // |END| k-Sparse filters. |END|
                    
                    // Constraint.
                    if((file >> tmp_line) && tmp_line.find(L"constraint_recurrent_weight_lower_bound") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"constraint_recurrent_weight_lower_bound\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_T[0] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"constraint_recurrent_weight_upper_bound") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"constraint_recurrent_weight_upper_bound\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_T[1] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__Regularization__Constraint_Recurrent_Weight(layer_it,
                                                                                       tmp_input_T[0],
                                                                                       tmp_input_T[1]) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Regularization__Constraint_Recurrent_Weight(ptr, %f, %f)\" function.",
                                                     tmp_input_T[0],
                                                     tmp_input_T[1]);

                            return false;
                        }
                    }
                    // |END| Constraint. |END|
                    
                    // Initialize basic unit(s).
                    layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
                    layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
                    // |END| Initialize basic unit(s). |END|
                    
                    // Initialize basic indice unit(s).
                    layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
                    layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;
                    // |END| Initialize basic indice unit(s). |END|
                    
                    // Initialize neuron unit(s).
                    if((file >> tmp_line) && tmp_line.find(L"number_neuron_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_neuron_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> *layer_it->ptr_number_outputs >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }
                        
                        layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
                        tmp_ptr_array_neuron_units += *layer_it->ptr_number_outputs;
                        layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;

                        this->Order__Layer__Neuron(layer_it);

                        switch(tmp_ptr_previous_layer->type_layer)
                        {
                            case LAYER::AVERAGE_POOLING:
                            case LAYER::RESIDUAL:
                                if(this->Load_Dimension__FC<Basic_unit, LAYER::AVERAGE_POOLING>(layer_it,
                                                                                                                                                                                                                             this->ptr_array_basic_units,
                                                                                                                                                                                                                             file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                                    
                                    file.close();

                                    return false;
                                }
                                    break;
                            case LAYER::FULLY_CONNECTED:
                            case LAYER::FULLY_CONNECTED_RECURRENT:
                            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                                if(this->Load_Dimension__FC<Neuron_unit, LAYER::FULLY_CONNECTED>(layer_it,
                                                                                                                                                                                                                              this->ptr_array_neuron_units,
                                                                                                                                                                                                                              file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                                    
                                    file.close();

                                    return false;
                                }
                                    break;
                            case LAYER::LSTM:
                                if(this->Load_Dimension__FC<CellUnit, LAYER::LSTM>(layer_it,
                                                                                                                                                                                                  this->ptr_array_cell_units,
                                                                                                                                                                                                  file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                                    
                                    file.close();

                                    return false;
                                }
                                    break;
                            case LAYER::MAX_POOLING:
                                if(this->Load_Dimension__FC<Basic_indice_unit, LAYER::MAX_POOLING>(layer_it,
                                                                                                                                                                                                                              this->ptr_array_basic_indice_units,
                                                                                                                                                                                                                              file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                                    
                                    file.close();

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
                    if((file >> tmp_line) && tmp_line.find(L"number_AF_Ind_recurrent_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_AF_Ind_recurrent_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }
                        
                        layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                        tmp_ptr_array_AF_Ind_recurrent_units += tmp_input_integer;
                        layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
                        
                        this->Order__Layer__AF_Ind_Recurrent(layer_it);

                        if(this->Load_Dimension__AF_Ind_Recurrent(layer_it, file) == false)
                        {
                            ERR(L"An error has been triggered from the \"Load_Dimension__AF_Ind_Recurrent()\" function.",);

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
                    if(this->Load_Dimension__Bias(layer_it, file) == false)
                    {
                        ERR(L"An error has been triggered from the \"Load_Dimension__Bias()\" function.",);

                        return false;
                    }
                    // |END| Initialize bias parameter(s). |END|
                        break;
                case LAYER::LSTM:
                    // Type activation.
                    if((file >> tmp_line) && tmp_line.find(L"type_activation") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"type_activation\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());

                            return false;
                        }

                        if(tmp_input_integer >= static_cast<size_t>(LAYER_ACTIVATION::LENGTH))
                        {
                            ERR(L"Undefined layer activation type %zu.",
                                                     tmp_input_integer);

                            return false;
                        }

                        layer_it->type_activation = static_cast<LAYER_ACTIVATION::TYPE>(tmp_input_integer);
                    }
                    // |END| Type activation. |END|

                    // Dropout.
                    if(load_dropout_params_fn(layer_it) == false)
                    {
                        ERR(L"An error has been triggered from the \"load_dropout_params_fn()\" function.",);

                        return false;
                    }
                    // |END| Dropout. |END|
                    
                    // Normalization.
                    if((file >> tmp_line) && tmp_line.find(L"type_normalization") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"type_normalization\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());

                            return false;
                        }

                        if(tmp_input_integer >= static_cast<size_t>(LAYER_NORM::LENGTH))
                        {
                            ERR(L"Undefined layer normalization type %zu.",
                                                     tmp_input_integer);

                            return false;
                        }
                        
                        if(this->Set__Layer_Normalization(layer_it,
                                                                          static_cast<LAYER_NORM::TYPE>(tmp_input_integer),
                                                                          false,
                                                                          false) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Layer_Normalization(%zu)\" function.",
                                                     tmp_input_integer);
                            
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"use_layer_normalization_before_activation") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"use_layer_normalization_before_activation\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->use_layer_normalization_before_activation >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"number_normalized_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_normalized_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }
                        
                        layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                        tmp_ptr_array_normalized_units += tmp_input_integer;
                        layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                        
                        this->Order__Layer__Normalization(layer_it);

                        if(this->Load_Dimension__Normalization(layer_it, file) == false)
                        {
                            ERR(L"An error has been triggered from the \"Load_Dimension__Normalization()\" function.",);

                            return false;
                        }
                    }
                    // |END| Normalization. |END|

                    if((file >> tmp_line) && tmp_line.find(L"use_tied_parameter") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"use_tied_parameter\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_boolean >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__Tied_Parameter(layer_it,
                                                              tmp_input_boolean,
                                                              false) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Tied_Parameter(ptr, %ls, false)\" function.",
                                                     tmp_input_boolean ? "true" : "false");

                            return false;
                        }
                    }
                    
                    // k-Sparse filters.
                    if((file >> tmp_line) && tmp_line.find(L"k_sparsity") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"k_sparsity\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__K_Sparsity(layer_it, tmp_input_integer) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__K_Sparsity(ptr, %zu)\" function.",
                                                     tmp_input_integer);

                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"alpha_sparsity") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"alpha_sparsity\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_T[0] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__Alpha_Sparsity(layer_it, tmp_input_T[0]) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Alpha_Sparsity(ptr, %f)\" function.",
                                                     tmp_input_T[0]);

                            return false;
                        }
                    }
                    // |END| k-Sparse filters. |END|
                    
                    // Constraint.
                    if((file >> tmp_line) && tmp_line.find(L"constraint_recurrent_weight_lower_bound") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"constraint_recurrent_weight_lower_bound\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_T[0] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"constraint_recurrent_weight_upper_bound") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"constraint_recurrent_weight_upper_bound\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_T[1] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }

                        if(this->Set__Regularization__Constraint_Recurrent_Weight(layer_it,
                                                                                       tmp_input_T[0],
                                                                                       tmp_input_T[1]) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Regularization__Constraint_Recurrent_Weight(ptr, %f, %f)\" function.",
                                                     tmp_input_T[0],
                                                     tmp_input_T[1]);

                            return false;
                        }
                    }
                    // |END| Constraint. |END|
                    
                    // Initialize basic unit(s).
                    layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
                    layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;
                    // |END| Initialize basic unit(s). |END|
                    
                    // Initialize basic indice unit(s).
                    layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
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
                    layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                    layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
                    // |END| Initialize AF Ind recurrent unit(s). |END|
                    
                    // Initialize block/cell unit(s).
                    if((file >> tmp_line) && tmp_line.find(L"number_block_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_block_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }
                            
                        layer_it->ptr_array_block_units = tmp_ptr_array_block_units;
                        tmp_ptr_array_block_units += tmp_input_integer;
                        layer_it->ptr_last_block_unit = tmp_ptr_array_block_units;

                        if(this->Load_Dimension__Cell_Units(layer_it,
                                                                                tmp_ptr_array_cell_units,
                                                                                file) == false)
                        {
                            ERR(L"An error has been triggered from the \"Load_Dimension__Cell_Units()\" function.",);

                            return false;
                        }
                    
                        *layer_it->ptr_number_outputs = static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units);

                        this->Order__Layer__LSTM(layer_it);

                        switch(tmp_ptr_previous_layer->type_layer)
                        {
                            case LAYER::AVERAGE_POOLING:
                            case LAYER::RESIDUAL:
                                if(this->Load_Dimension__LSTM<Basic_unit, LAYER::AVERAGE_POOLING>(layer_it,
                                                                                                                                                                                                                                 this->ptr_array_basic_units,
                                                                                                                                                                                                                                 file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__LSTM()\" function.",);
                                    
                                    file.close();

                                    return false;
                                }
                                    break;
                            case LAYER::FULLY_CONNECTED:
                            case LAYER::FULLY_CONNECTED_RECURRENT:
                            case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                                if(this->Load_Dimension__LSTM<Neuron_unit, LAYER::FULLY_CONNECTED>(layer_it,
                                                                                                                                                                                                                                  this->ptr_array_neuron_units,
                                                                                                                                                                                                                                  file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__LSTM()\" function.",);
                                    
                                    file.close();

                                    return false;
                                }
                                    break;
                            case LAYER::LSTM:
                                if(this->Load_Dimension__LSTM<CellUnit, LAYER::LSTM>(layer_it,
                                                                                                                                                                                                       this->ptr_array_cell_units,
                                                                                                                                                                                                       file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__LSTM()\" function.",);
                                    
                                    file.close();

                                    return false;
                                }
                                    break;
                            case LAYER::MAX_POOLING:
                                if(this->Load_Dimension__LSTM<Basic_indice_unit, LAYER::MAX_POOLING>(layer_it,
                                                                                                                                                                                                                                  this->ptr_array_basic_indice_units,
                                                                                                                                                                                                                                  file) == false)
                                {
                                    ERR(L"An error has been triggered from the \"Load_Dimension__LSTM()\" function.",);
                                        
                                    file.close();

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
                    if(this->Load_Dimension__Bias(layer_it, file) == false)
                    {
                        ERR(L"An error has been triggered from the \"Load_Dimension__Bias()\" function.",);

                        return false;
                    }
                    // |END| Initialize bias parameter(s). |END|
                        break;
                case LAYER::MAX_POOLING:
                    // Pooling.
                    if((file >> tmp_line) && tmp_line.find(L"kernel_size") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"kernel_size\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->pooling_values[0] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"stride") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"stride\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->pooling_values[1] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"padding") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"padding\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->pooling_values[2] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"dilation") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"dilation\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->pooling_values[3] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
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
                    if((file >> tmp_line) && tmp_line.find(L"number_basic_indice_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_basic_indice_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> *layer_it->ptr_number_outputs >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }

                        layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
                        tmp_ptr_array_basic_indice_units += *layer_it->ptr_number_outputs;
                        layer_it->ptr_last_basic_indice_unit = tmp_ptr_array_basic_indice_units;

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
                    layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                    layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
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
                    if((file >> tmp_line) && tmp_line.find(L"block_depth") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"block_depth\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> layer_it->block_depth >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }
                    }
                    // |END| Initialize block depth. |END|
                    
                    // Initialize padding.
                    if((file >> tmp_line) && tmp_line.find(L"padding") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"padding\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
            
                        return false;
                    }
                    else
                    {
                        file >> layer_it->pooling_values[2] >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                
                            return false;
                        }
                    }
                    // |END| Initialize padding. |END|
                    
                    // Dropout.
                    if(load_dropout_params_fn(layer_it) == false)
                    {
                        ERR(L"An error has been triggered from the \"load_dropout_params_fn()\" function.",);

                        return false;
                    }
                    // |END| Dropout. |END|
                    
                    // Normalization.
                    if((file >> tmp_line) && tmp_line.find(L"type_normalization") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"type_normalization\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());

                            return false;
                        }

                        if(tmp_input_integer >= static_cast<size_t>(LAYER_NORM::LENGTH))
                        {
                            ERR(L"Undefined layer normalization type %zu.",
                                                     tmp_input_integer);

                            return false;
                        }
                        
                        if(this->Set__Layer_Normalization(layer_it,
                                                                          static_cast<LAYER_NORM::TYPE>(tmp_input_integer),
                                                                          false,
                                                                          false) == false)
                        {
                            ERR(L"An error has been triggered from the \"Set__Layer_Normalization(%zu)\" function.",
                                                     tmp_input_integer);
                            
                            return false;
                        }
                    }
                    
                    if((file >> tmp_line) && tmp_line.find(L"number_normalized_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_normalized_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> tmp_input_integer >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }
                        
                        layer_it->ptr_array_normalized_units = tmp_ptr_array_normalized_units;
                        tmp_ptr_array_normalized_units += tmp_input_integer;
                        layer_it->ptr_last_normalized_unit = tmp_ptr_array_normalized_units;
                        
                        this->Order__Layer__Normalization(layer_it);

                        if(this->Load_Dimension__Normalization(layer_it, file) == false)
                        {
                            ERR(L"An error has been triggered from the \"Load_Dimension__Normalization()\" function.",);

                            return false;
                        }
                    }
                    // |END| Normalization. |END|
                    
                    // Initialize basic unit(s).
                    if((file >> tmp_line) && tmp_line.find(L"number_basic_units") == std::wstring::npos)
                    {
                        ERR(L"Can not find \"number_basic_units\" inside \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                    else if(file.fail())
                    {
                        ERR(L"Can not read properly inside \"%ls\".",
                                                 tmp_line.c_str());
                
                        return false;
                    }
                    else
                    {
                        file >> *layer_it->ptr_number_outputs >> std::ws;

                        if(file.fail())
                        {
                            ERR(L"Can not read input of \"%ls\".",
                                                     tmp_line.c_str());
                    
                            return false;
                        }

                        layer_it->ptr_array_basic_units = tmp_ptr_array_basic_units;
                        tmp_ptr_array_basic_units += *layer_it->ptr_number_outputs;
                        layer_it->ptr_last_basic_unit = tmp_ptr_array_basic_units;

                        this->Order__Layer__Basic(layer_it);
                    }
                    // |END| Initialize basic unit(s). |END|
                    
                    //  Initialize basic indice unit(s).
                    layer_it->ptr_array_basic_indice_units = tmp_ptr_array_basic_indice_units;
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
                    layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
                    layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
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
                                             layer_it->type_layer,
                                             LAYER_NAME[layer_it->type_layer].c_str());
                        return false;
            }
        }
        // |END| Hidden layer. |END|

        // allocate bidirectional layer(s).
        if(this->Allocate__Bidirectional__Layers() == false)
        {
            ERR(L"An error has been triggered from the \"Allocate__Bidirectional__Layers()\" function.",);

            file.close();

            return false;
        }
        // |END| allocate bidirectional layer(s). |END|

        // Output layer.
        //  Type layer.
        INFO(L"load output layer.");
        getline(file, tmp_line); // "Output layer:"
        
        if((file >> tmp_line) && tmp_line.find(L"type_layer") == std::wstring::npos)
        {
            ERR(L"Can not find \"type_layer\" inside \"%ls\".",
                                     tmp_line.c_str());
            
            file.close();

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            file.close();

            return false;
        }
        else
        {
            file >> tmp_input_integer;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                file.close();

                return false;
            }

            if(tmp_input_integer >= static_cast<size_t>(LAYER::LENGTH))
            {
                ERR(L"Undefined layer type %zu.",
                                         tmp_input_integer);
                
                file.close();

                return false;
            }

            layer_it->type_layer = static_cast<LAYER::TYPE>(tmp_input_integer);
        }
        //  |END| Type layer. |END|
        
        this->Organize__Previous_Layers_Connected(tmp_state_layer_index,
                                                                           layer_it,
                                                                           tmp_ptr_layer_state);
        
        tmp_ptr_previous_layer = layer_it->previous_connected_layers[0];

        //  Type activation.
        if((file >> tmp_line) && tmp_line.find(L"type_activation") == std::wstring::npos)
        {
            ERR(L"Can not find \"type_activation\" inside \"%ls\".",
                                        tmp_line.c_str());
            
            file.close();

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                        tmp_line.c_str());
            
            file.close();

            return false;
        }
        else
        {
            file >> tmp_input_integer;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                            tmp_line.c_str());
                
                file.close();

                return false;
            }

            if(tmp_input_integer >= static_cast<size_t>(LAYER_ACTIVATION::LENGTH))
            {
                ERR(L"Undefined layer activation type %zu.",
                                            tmp_input_integer);
                
                file.close();

                return false;
            }

            layer_it->type_activation = static_cast<LAYER_ACTIVATION::TYPE>(tmp_input_integer);
        }
        //  |END| Type activation. |END|
        
        //  Initialize output unit(s).
        if((file >> tmp_line) && tmp_line.find(L"n_out") == std::wstring::npos)
        {
            ERR(L"Can not find \"n_out\" inside \"%ls\".",
                                     tmp_line.c_str());
            
            file.close();

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
            
            file.close();

            return false;
        }
        else
        {
            file >> tmp_input_integer >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                
                file.close();

                return false;
            }

            *layer_it->ptr_number_outputs = this->n_out = tmp_input_integer;
            
            layer_it->ptr_array_neuron_units = tmp_ptr_array_neuron_units;
            tmp_ptr_array_neuron_units += tmp_input_integer;
            layer_it->ptr_last_neuron_unit = tmp_ptr_array_neuron_units;
            
            this->Order__Layer__Neuron(layer_it);

            switch(tmp_ptr_previous_layer->type_layer)
            {
                case LAYER::AVERAGE_POOLING:
                case LAYER::RESIDUAL:
                    if(this->Load_Dimension__FC<Basic_unit, LAYER::AVERAGE_POOLING>(layer_it,
                                                                                                                                                                                                                 this->ptr_array_basic_units,
                                                                                                                                                                                                                 file) == false)
                    {
                        ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                                    
                        file.close();

                        return false;
                    }
                        break;
                case LAYER::FULLY_CONNECTED:
                case LAYER::FULLY_CONNECTED_RECURRENT:
                case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
                    if(this->Load_Dimension__FC<Neuron_unit, LAYER::FULLY_CONNECTED>(layer_it,
                                                                                                                                                                                                                  this->ptr_array_neuron_units,
                                                                                                                                                                                                                  file) == false)
                    {
                        ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                        
                        file.close();

                        return false;
                    }
                        break;
                case LAYER::LSTM:
                    if(this->Load_Dimension__FC<CellUnit, LAYER::LSTM>(layer_it,
                                                                                                                                                                                      this->ptr_array_cell_units,
                                                                                                                                                                                      file) == false)
                    {
                        ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                        
                        file.close();

                        return false;
                    }
                        break;
                case LAYER::MAX_POOLING:
                    if(this->Load_Dimension__FC<Basic_indice_unit, LAYER::MAX_POOLING>(layer_it,
                                                                                                                                                                                                                  this->ptr_array_basic_indice_units,
                                                                                                                                                                                                                  file) == false)
                    {
                        ERR(L"An error has been triggered from the \"Load_Dimension__FC()\" function.",);
                                        
                        file.close();

                        return false;
                    }
                        break;
                default:
                    ERR(L"Layer type (%d | %ls) is not managed in the switch.",
                                             tmp_ptr_previous_layer->type_layer,
                                             LAYER_NAME[tmp_ptr_previous_layer->type_layer].c_str());
                    file.close();
                        return false;
            }
        }
        //  |END| Initialize output unit(s). |END|
        
        // Initialize AF unit(s).
        if((file >> tmp_line) && tmp_line.find(L"number_AF_units") == std::wstring::npos)
        {
            ERR(L"Can not find \"number_AF_units\" inside \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                     tmp_line.c_str());
                
            return false;
        }
        else
        {
            file >> tmp_input_integer >> std::ws;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                         tmp_line.c_str());
                    
                return false;
            }
                        
            layer_it->ptr_array_AF_units = tmp_ptr_array_AF_units;
            tmp_ptr_array_AF_units += tmp_input_integer;
            layer_it->ptr_last_AF_unit = tmp_ptr_array_AF_units;
                        
            this->Order__Layer__AF(layer_it);

            if(this->Load_Dimension__AF(layer_it, file) == false)
            {
                ERR(L"An error has been triggered from the \"Load_Dimension__AF()\" function.",);

                return false;
            }
        }
        // |END| Initialize AF unit(s). |END|
        
        // Initialize AF Ind recurrent unit(s).
        layer_it->ptr_array_AF_Ind_recurrent_units = tmp_ptr_array_AF_Ind_recurrent_units;
        layer_it->ptr_last_AF_Ind_recurrent_unit = tmp_ptr_array_AF_Ind_recurrent_units;
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
        if(this->Load_Dimension__Bias(layer_it, file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Bias()\" function.",);

            return false;
        }
        //  |END| Initialize bias parameter(s). |END|
        // |END| Output layer. |END|
        
        if(file.fail())
        {
            ERR(L"Logical error on i/o operation \"%ls\".",
                                     path_params.c_str());
            
            file.close();

            return false;
        }

        file.close();
        
        if(this->total_weights != this->total_weights_allocated)
        {
            ERR(L"Total weights prepared (%zu) differ from the total weights allocated (%zu).",
                                     this->total_weights,
                                     this->total_weights_allocated);

            return false;
        }
        else if(this->total_bias != this->total_bias_allocated)
        {
            ERR(L"Total bias prepared (%zu) differ from the total bias allocated (%zu).",
                                     this->total_bias,
                                     this->total_bias_allocated);

            return false;
        }
        // |END| Initialize layer(s). |END|
        
        // Layers, connections.
        this->Order__Layers__Connection();
        
        // Layers, outputs pointers.
        this->Order__Layers__Output();

        if(this->load_spec_params(path_spec_params) == false)
        {
            ERR(L"An error has been triggered from the \"load_spec_params(%ls)\" function.",
                                     path_spec_params.c_str());

            return false;
        }

    return true;
}

bool Model::Load_Dimension__Neuron(Neuron_unit *const ptr_neuron_received, std::wifstream &file)
{
    std::wstring tmp_line;

    getline(file, tmp_line); // "Neuron_unit[%zu]"
    
    // Number connection(s).
    if((file >> tmp_line) && tmp_line.find(L"number_connections") == std::wstring::npos)
    {
        ERR(L"Can not find \"number_connections\" inside \"%ls\".",
                                 tmp_line.c_str());

        return false;
    }
    else if(file.fail())
    {
        ERR(L"Can not read properly inside \"%ls\".",
                                 tmp_line.c_str());

        return false;
    }
    else
    {
        file >> *ptr_neuron_received->ptr_number_connections >> std::ws;

        if(file.fail())
        {
            ERR(L"Can not read input of \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
    }

    *ptr_neuron_received->ptr_first_connection_index = this->total_weights;
    this->total_weights += *ptr_neuron_received->ptr_number_connections;
    *ptr_neuron_received->ptr_last_connection_index = this->total_weights;
    // |END| Number connection(s). |END|

    return true;
}

bool Model::Load_Dimension__AF(AF_unit *const ptr_AF_received, std::wifstream &file)
{
    size_t tmp_input_integer;

    std::wstring tmp_line;

    getline(file, tmp_line); // "AF[%zu]"
    
    // Activation function.
    if((file >> tmp_line) && tmp_line.find(L"activation_function") == std::wstring::npos)
    {
        ERR(L"Can not find \"activation_function\" inside \"%ls\".",
                                 tmp_line.c_str());

        return false;
    }
    else if(file.fail())
    {
        ERR(L"Can not read properly inside \"%ls\".",
                                 tmp_line.c_str());

        return false;
    }
    else
    {
        file >> tmp_input_integer >> std::ws;

        if(file.fail())
        {
            ERR(L"Can not read input of \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }

        if(tmp_input_integer >= static_cast<size_t>(ACTIVATION::LENGTH))
        {
            ERR(L"Undefined activation function type %zu.",
                                     tmp_input_integer);

            return false;
        }

        *ptr_AF_received->ptr_type_activation_function = static_cast<ACTIVATION::TYPE>(tmp_input_integer);
    }
    // |END| Activation function. |END|

    return true;
}

bool Model::Load_Dimension__AF_Ind_Recurrent(AF_Ind_recurrent_unit *const ptr_AF_Ind_received, std::wifstream &file)
{
    size_t tmp_input_integer;

    std::wstring tmp_line;

    getline(file, tmp_line); // "AF_Ind_R[%zu]"
    
    // Activation function.
    if((file >> tmp_line) && tmp_line.find(L"activation_function") == std::wstring::npos)
    {
        ERR(L"Can not find \"activation_function\" inside \"%ls\".",
                                 tmp_line.c_str());

        return false;
    }
    else if(file.fail())
    {
        ERR(L"Can not read properly inside \"%ls\".",
                                 tmp_line.c_str());

        return false;
    }
    else
    {
        file >> tmp_input_integer >> std::ws;

        if(file.fail())
        {
            ERR(L"Can not read input of \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }

        if(tmp_input_integer >= static_cast<size_t>(ACTIVATION::LENGTH))
        {
            ERR(L"Undefined activation function type %zu.",
                                     tmp_input_integer);

            return false;
        }

        *ptr_AF_Ind_received->ptr_type_activation_function = static_cast<ACTIVATION::TYPE>(tmp_input_integer);
    }
    // |END| Activation function. |END|
    
    *ptr_AF_Ind_received->ptr_recurrent_connection_index = this->total_weights++;
    
    AF_Ind_recurrent_unit **tmp_ptr_array_U_ptr_connections(reinterpret_cast<AF_Ind_recurrent_unit **>(this->ptr_array_ptr_connections));
    
    if(this->Load_Dimension__Connection<AF_Ind_recurrent_unit, LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT>(*ptr_AF_Ind_received->ptr_recurrent_connection_index,
                                                                                                                                                                                                                                                                this->ptr_array_parameters,
                                                                                                                                                                                                                                                                this->ptr_array_AF_Ind_recurrent_units,
                                                                                                                                                                                                                                                                tmp_ptr_array_U_ptr_connections,
                                                                                                                                                                                                                                                                file) == false)
    {
        ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                 *ptr_AF_Ind_received->ptr_recurrent_connection_index);

        return false;
    }

    return true;
}

bool Model::Load_Dimension__Normalized_Unit(size_t const number_units_received,
                                                                                    LAYER_NORM::TYPE const type_normalization_received,
                                                                                    union Normalized_unit *const ptr_normalized_unit_received,
                                                                                    std::wifstream &file)
{
    size_t tmp_time_step_index,
              tmp_unit_timed_index;

    std::wstring tmp_line;

    real out;
    
    getline(file, tmp_line); // "NormU[%zu]"
    
    switch(type_normalization_received)
    {
        case LAYER_NORM::BATCH_NORMALIZATION:
        case LAYER_NORM::BATCH_RENORMALIZATION:
        case LAYER_NORM::GHOST_BATCH_NORMALIZATION:
            // Scale.
            if((file >> tmp_line) && tmp_line.find(L"scale") == std::wstring::npos)
            {
                ERR(L"Can not find \"scale\" inside \"%ls\".",
                                            tmp_line.c_str());

                return false;
            }
            else if(file.fail())
            {
                ERR(L"Can not read properly inside \"%ls\".",
                                            tmp_line.c_str());

                return false;
            }
            else
            {
              file >> out >> std::ws;
              *ptr_normalized_unit_received->normalized_batch_units.ptr_scale =
                  out;

                if(file.fail())
                {
                    ERR(L"Can not read input of \"%ls\".",
                                                tmp_line.c_str());

                    return false;
                }
            }
            // |END| Scale. |END|
                
            // shift.
            if((file >> tmp_line) && tmp_line.find(L"shift") == std::wstring::npos)
            {
                ERR(L"Can not find \"scale\" inside \"%ls\".",
                                            tmp_line.c_str());

                return false;
            }
            else if(file.fail())
            {
                ERR(L"Can not read properly inside \"%ls\".",
                                            tmp_line.c_str());

                return false;
            }
            else
            {
              file >> out >> std::ws;
              *ptr_normalized_unit_received->normalized_batch_units.ptr_shift =
                  out;

                if(file.fail())
                {
                    ERR(L"Can not read input of \"%ls\".",
                                                tmp_line.c_str());

                    return false;
                }
            }
            // |END| shift. |END|
                
            for(tmp_time_step_index = 0_UZ; tmp_time_step_index != this->seq_w; ++tmp_time_step_index)
            {
                tmp_unit_timed_index = number_units_received * tmp_time_step_index;
                
                // Mean average.
                if((file >> tmp_line) && tmp_line.find(L"mean_average[" + std::to_wstring(tmp_time_step_index) + L"]") == std::wstring::npos)
                {
                    ERR(L"Can not find \"mean_average[%zu]\" inside \"%ls\".",
                                             tmp_time_step_index,
                                             tmp_line.c_str());

                    return false;
                }
                else if(file.fail())
                {
                    ERR(L"Can not read properly inside \"%ls\".",
                                             tmp_line.c_str());

                    return false;
                }
                else {
                  file >> out >> std::ws;
                  ptr_normalized_unit_received->normalized_batch_units.ptr_mean_average[tmp_unit_timed_index] = out;

                    if(file.fail())
                    {
                        ERR(L"Can not read input of \"%ls\".",
                                                 tmp_line.c_str());

                        return false;
                    }
                }
                // |END| Mean average. |END|
                
                // Variance average.
                if((file >> tmp_line) && tmp_line.find(L"variance_average[" + std::to_wstring(tmp_time_step_index) + L"]") == std::wstring::npos)
                {
                    ERR(L"Can not find \"variance_average[%zu]\" inside \"%ls\".",
                                             tmp_time_step_index,
                                             tmp_line.c_str());

                    return false;
                }
                else if(file.fail())
                {
                    ERR(L"Can not read properly inside \"%ls\".",
                                             tmp_line.c_str());

                    return false;
                }
                else {
                  file >> out >> std::ws;
                  ptr_normalized_unit_received->normalized_batch_units
                      .ptr_variance_average[tmp_unit_timed_index] = out;

                    if(file.fail())
                    {
                        ERR(L"Can not read input of \"%ls\".",
                                                 tmp_line.c_str());

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

bool Model::Load_Dimension__Bias(Layer *const layer_it, std::wifstream &file)
{
    size_t tmp_input_integer;

    std::wstring tmp_line;

    if((file >> tmp_line) && tmp_line.find(L"number_bias_parameters") == std::wstring::npos)
    {
        ERR(L"Can not find \"number_bias_parameters\" inside \"%ls\".",
                                 tmp_line.c_str());

        return false;
    }
    else if(file.fail())
    {
        ERR(L"Can not read properly inside \"%ls\".",
                                 tmp_line.c_str());
                
        return false;
    }
    else
    {
        file >> tmp_input_integer >> std::ws;

        if(file.fail())
        {
            ERR(L"Can not read input of \"%ls\".",
                                     tmp_line.c_str());
                    
            return false;
        }
    }

    var *const tmp_ptr_array_parameters(this->ptr_array_parameters + this->total_weights_allocated + this->total_bias);

    real out;

    layer_it->first_bias_connection_index = this->total_weights_allocated + this->total_bias;

    for(size_t tmp_connection_index(0_UZ); tmp_connection_index != tmp_input_integer; ++tmp_connection_index)
    {
        if((file >> tmp_line) && tmp_line.find(L"weight") == std::wstring::npos)
        {
            ERR(L"Can not find \"weight\" inside \"%ls\".",
                                        tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                        tmp_line.c_str());

            return false;
        }
        else
        {
          file >> out >> std::ws;
          tmp_ptr_array_parameters[tmp_connection_index] = out;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                            tmp_line.c_str());

                return false;
            }
        }
    }

    this->total_bias += tmp_input_integer;

    layer_it->last_bias_connection_index = this->total_weights_allocated + this->total_bias;

    return true;
}

bool Model::Load_Dimension__Block(size_t const layer_number_block_units_received,
                                                                     size_t const layer_number_cell_units_received,
                                                                     LAYER_NORM::TYPE const type_normalization_received,
                                                                     BlockUnit *const ptr_block_unit_it_received,
                                                                     std::wifstream &file)
{
    CellUnit const *const tmp_ptr_block_ptr_cell_unit(ptr_block_unit_it_received->ptr_array_cell_units),
                                    *const tmp_ptr_block_ptr_last_cell_unit(ptr_block_unit_it_received->ptr_last_cell_unit);
    CellUnit *tmp_ptr_block_ptr_cell_unit_it;
    
    size_t const tmp_block_number_cell_units(static_cast<size_t>(tmp_ptr_block_ptr_last_cell_unit - tmp_ptr_block_ptr_cell_unit));
    size_t tmp_input_integer;
    
    std::wstring tmp_line;

    getline(file, tmp_line); // "Block[%zu]"

    // Activation function.
    if((file >> tmp_line) && tmp_line.find(L"activation_function") == std::wstring::npos)
    {
        ERR(L"Can not find \"activation_function\" inside \"%ls\".",
                                 tmp_line.c_str());

        return false;
    }
    else if(file.fail())
    {
        ERR(L"Can not read properly inside \"%ls\".",
                                 tmp_line.c_str());

        return false;
    }
    else
    {
        file >> tmp_input_integer >> std::ws;

        if(file.fail())
        {
            ERR(L"Can not read input of \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }

        if(tmp_input_integer >= static_cast<size_t>(ACTIVATION::LENGTH))
        {
            ERR(L"Undefined activation function type %zu.",
                                     tmp_input_integer);

            return false;
        }

        ptr_block_unit_it_received->activation_function_io = static_cast<ACTIVATION::TYPE>(tmp_input_integer);
    }
    // |END| Activation function. |END|

    // Number connection(s).
    if((file >> tmp_line) && tmp_line.find(L"number_connections") == std::wstring::npos)
    {
        ERR(L"Can not find \"number_connections\" inside \"%ls\".",
                                 tmp_line.c_str());

        return false;
    }
    else if(file.fail())
    {
        ERR(L"Can not read properly inside \"%ls\".",
                                 tmp_line.c_str());

        return false;
    }
    else
    {
        file >> tmp_input_integer >> std::ws;

        if(file.fail())
        {
            ERR(L"Can not read input of \"%ls\".",
                                     tmp_line.c_str());

            return false;
        }
    }

#ifndef NO_PEEPHOLE
    size_t const tmp_number_inputs((tmp_input_integer - layer_number_cell_units_received * 4_UZ - tmp_block_number_cell_units * 3_UZ) / 4_UZ);
#else
    size_t const tmp_number_inputs((tmp_input_integer - layer_number_cell_units_received * 4_UZ) / 4_UZ);
#endif

    ptr_block_unit_it_received->first_index_connection = this->total_weights;

    // [0] Cell input.
    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        //    [1] Input, cell.
        tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input = this->total_weights;
        this->total_weights += tmp_number_inputs;
        tmp_ptr_block_ptr_cell_unit_it->last_index_feedforward_connection_cell_input = this->total_weights;
        //    [1] |END| Input, cell. |END|

        //    [1] Recurrent, cell.
        tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input = this->total_weights;
        this->total_weights += layer_number_cell_units_received;
        tmp_ptr_block_ptr_cell_unit_it->last_index_recurrent_connection_cell_input = this->total_weights;
        //    [1] |END| Recurrent, cell. |END|
    }
    // [0] |END| Cell input. |END|
    
    // [0] Input, gates.
    //    [1] Input gate.
    ptr_block_unit_it_received->first_index_feedforward_connection_input_gate = this->total_weights;
    this->total_weights += tmp_number_inputs;
    ptr_block_unit_it_received->last_index_feedforward_connection_input_gate = this->total_weights;
    //    [1] |END| Input gate. |END|
    
    //    [1] Forget gate.
    ptr_block_unit_it_received->first_index_feedforward_connection_forget_gate = this->total_weights;
    this->total_weights += tmp_number_inputs;
    ptr_block_unit_it_received->last_index_feedforward_connection_forget_gate = this->total_weights;
    //    [1] |END| Forget gate. |END|

    //    [1] Output gate.
    ptr_block_unit_it_received->first_index_feedforward_connection_output_gate = this->total_weights;
    this->total_weights += tmp_number_inputs;
    ptr_block_unit_it_received->last_index_feedforward_connection_output_gate = this->total_weights;
    //    [1] |END| Output gate. |END|
    // [0] |END| Input, gates. |END|
    
    // [0] Recurrent, gates.
    //    [1] Input gate.
    ptr_block_unit_it_received->first_index_recurrent_connection_input_gate = this->total_weights;
    this->total_weights += layer_number_cell_units_received;
    ptr_block_unit_it_received->last_index_recurrent_connection_input_gate = this->total_weights;
    //    [1] |END| Input gate. |END|

    //    [1] Forget gate.
    ptr_block_unit_it_received->first_index_recurrent_connection_forget_gate = this->total_weights;
    this->total_weights += layer_number_cell_units_received;
    ptr_block_unit_it_received->last_index_recurrent_connection_forget_gate = this->total_weights;
    //    [1] |END| Forget gate. |END|

    //    [1] Output gate.
    ptr_block_unit_it_received->first_index_recurrent_connection_output_gate = this->total_weights;
    this->total_weights += layer_number_cell_units_received;
    ptr_block_unit_it_received->last_index_recurrent_connection_output_gate = this->total_weights;
    //    [1] |END| Output gate. |END|
    // [0] |END| Recurrent, gates. |END|

#ifndef NO_PEEPHOLE
    // [0] Peepholes.
    //    [1] Input gate.
    ptr_block_unit_it_received->first_index_peephole_input_gate = this->total_weights;
    this->total_weights += tmp_block_number_cell_units;
    ptr_block_unit_it_received->last_index_peephole_input_gate = this->total_weights;
    //    [1] |END| Input gate. |END|

    //    [1] Forget gate.
    ptr_block_unit_it_received->first_index_peephole_forget_gate = this->total_weights;
    this->total_weights += tmp_block_number_cell_units;
    ptr_block_unit_it_received->last_index_peephole_forget_gate = this->total_weights;
    //    [1] |END| Forget gate. |END|

    //    [1] Output gate.
    ptr_block_unit_it_received->first_index_peephole_output_gate = this->total_weights;
    this->total_weights += tmp_block_number_cell_units;
    ptr_block_unit_it_received->last_index_peephole_output_gate = this->total_weights;
    //    [1] |END| Output gate. |END|

    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        tmp_ptr_block_ptr_cell_unit_it->index_peephole_input_gate = ptr_block_unit_it_received->first_index_peephole_input_gate + static_cast<size_t>(tmp_ptr_block_ptr_cell_unit_it - tmp_ptr_block_ptr_cell_unit);
        tmp_ptr_block_ptr_cell_unit_it->index_peephole_forget_gate = ptr_block_unit_it_received->first_index_peephole_forget_gate + static_cast<size_t>(tmp_ptr_block_ptr_cell_unit_it - tmp_ptr_block_ptr_cell_unit);
        tmp_ptr_block_ptr_cell_unit_it->index_peephole_output_gate = ptr_block_unit_it_received->first_index_peephole_output_gate + static_cast<size_t>(tmp_ptr_block_ptr_cell_unit_it - tmp_ptr_block_ptr_cell_unit);
    }
    // [0] |END| Peepholes. |END|
#endif

    ptr_block_unit_it_received->last_index_connection = this->total_weights;
    // |END| Number connection(s). |END|

    return true;
}

bool Model::Load_Dimension__Cell_Units(Layer *const layer_it,
                                                                           CellUnit *&ptr_reference_array_cells_received,
                                                                           std::wifstream &file)
{
    size_t tmp_input_integer;

    std::wstring tmp_line;

    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it(layer_it->ptr_array_block_units);
    
    // Number cell unit(s).
    if((file >> tmp_line) && tmp_line.find(L"number_cell_units") == std::wstring::npos)
    {
        ERR(L"Can not find \"number_cell_units\" inside \"%ls\".",
                                    tmp_line.c_str());

        return false;
    }
    else if(file.fail())
    {
        ERR(L"Can not read properly inside \"%ls\".",
                                    tmp_line.c_str());

        return false;
    }
    else
    {
        file >> tmp_input_integer >> std::ws;

        if(file.fail())
        {
            ERR(L"Can not read input of \"%ls\".",
                                        tmp_line.c_str());

            return false;
        }
    }

    size_t const tmp_layer_number_block_units(static_cast<size_t>(tmp_ptr_last_block_unit - tmp_ptr_block_unit_it)),
                       tmp_block_number_cell_units(tmp_input_integer / tmp_layer_number_block_units);
    // |END| Number cell unit(s). |END|

    layer_it->ptr_array_cell_units = ptr_reference_array_cells_received;

    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        tmp_ptr_block_unit_it->ptr_array_cell_units = ptr_reference_array_cells_received;
        ptr_reference_array_cells_received += tmp_block_number_cell_units;
        tmp_ptr_block_unit_it->ptr_last_cell_unit = ptr_reference_array_cells_received;
    }

    layer_it->ptr_last_cell_unit = ptr_reference_array_cells_received;

    return true;
}

template<class U, LAYER::TYPE const E> inline bool Model::Load_Dimension__Connection(size_t index_received,
                                                                                                                                                                                                         var *const ptr_array_parameters_received,
                                                                                                                                                                                                         U *const ptr_first_U_unit_received,
                                                                                                                                                                                                         U **ptr_array_ptr_U_unit_connection_received,
                                                                                                                                                                                                         std::wifstream &file)
{
    size_t tmp_input_integer;

    std::wstring tmp_line;

    real out;

    file >> tmp_line; // "connected_to_%ls=%u"

    if(file.fail())
    {
        ERR(L"Can not read properly inside \"%ls\".",
                                    tmp_line.c_str());

        return false;
    }

    if(tmp_line.find(LAYER_CONN_NAME[E]) != std::wstring::npos) // If is "connected_to_neuron".
    {
        file >> tmp_input_integer >> std::ws;

        if(file.fail())
        {
            ERR(L"Can not read input of \"%ls\".",
                                        tmp_line.c_str());

            return false;
        }

        ptr_array_ptr_U_unit_connection_received[index_received] = ptr_first_U_unit_received + tmp_input_integer;
            
        if((file >> tmp_line) && tmp_line.find(L"weight") == std::wstring::npos)
        {
            ERR(L"Can not find \"weight\" inside \"%ls\".",
                                        tmp_line.c_str());

            return false;
        }
        else if(file.fail())
        {
            ERR(L"Can not read properly inside \"%ls\".",
                                        tmp_line.c_str());

            return false;
        }
        else
        {
            file >> out >> std::ws;
          ptr_array_parameters_received[index_received] = out;

            if(file.fail())
            {
                ERR(L"Can not read input of \"%ls\".",
                                            tmp_line.c_str());

                return false;
            }
        }
    }
    else
    {
        ERR(L"Can not find \"connected_to_neuron\" inside \"%ls\".",
                                    tmp_line.c_str());

        return false;
    }

    return true;
}

template<class U, LAYER::TYPE const E> bool Model::Load_Dimension__Neuron__Forward__Connection(Neuron_unit *const ptr_neuron_received,
                                                                                                                                                                                                                                U *const ptr_first_U_unit_received,
                                                                                                                                                                                                                                std::wifstream &file)
{
    size_t const tmp_number_connections(*ptr_neuron_received->ptr_number_connections);
    size_t tmp_connection_index;

    var *const tmp_ptr_array_parameters(this->ptr_array_parameters + *ptr_neuron_received->ptr_first_connection_index);
    
    U **tmp_ptr_array_U_ptr_connections(reinterpret_cast<U **>(this->ptr_array_ptr_connections + *ptr_neuron_received->ptr_first_connection_index));
    
    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_connections; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<U, E>(tmp_connection_index,
                                                                          tmp_ptr_array_parameters,
                                                                          ptr_first_U_unit_received,
                                                                          tmp_ptr_array_U_ptr_connections,
                                                                          file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                     tmp_connection_index);

            return false;
        }
    }

    return true;
}

template<class U, LAYER::TYPE const E> bool Model::Load_Dimension__Block__Connection(BlockUnit *const ptr_block_unit_it_received,
                                                                                                                                                                                                             U *const ptr_first_U_unit_received,
                                                                                                                                                                                                             std::wifstream &file)
{
    size_t const tmp_number_inputs_connections(ptr_block_unit_it_received->last_index_feedforward_connection_input_gate - ptr_block_unit_it_received->first_index_feedforward_connection_input_gate),
                       tmp_number_recurrents_connection(ptr_block_unit_it_received->last_index_recurrent_connection_input_gate - ptr_block_unit_it_received->first_index_recurrent_connection_input_gate);
    size_t tmp_connection_index;
    
    var *tmp_ptr_array_parameters;

    CellUnit const *const tmp_ptr_block_ptr_last_cell_unit(ptr_block_unit_it_received->ptr_last_cell_unit);
    CellUnit *tmp_ptr_block_ptr_cell_unit_it,
                           **tmp_ptr_array_ptr_connections_layer_cell_units;
    
#ifndef NO_PEEPHOLE
    CellUnit **tmp_ptr_array_ptr_connections_peepholes_cell_units(reinterpret_cast<CellUnit **>(this->ptr_array_ptr_connections));
#endif

    U **tmp_ptr_array_U_ptr_connections;
    
    // [0] Cell input.
    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        //    [1] Input, cell input.
        tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(this->ptr_array_ptr_connections + tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input);

        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_feedforward_connection_cell_input;

        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
        {
            if(this->Load_Dimension__Connection<U, E>(tmp_connection_index,
                                                                              tmp_ptr_array_parameters,
                                                                              ptr_first_U_unit_received,
                                                                              tmp_ptr_array_U_ptr_connections,
                                                                              file) == false)
            {
                ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                         tmp_connection_index);

                return false;
            }
        }
        //    [1] |END| Input, cell input. |END|
        
        //    [1] Recurrent, cell input.
        tmp_ptr_array_ptr_connections_layer_cell_units = reinterpret_cast<CellUnit **>(this->ptr_array_ptr_connections + tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input);

        tmp_ptr_array_parameters = this->ptr_array_parameters + tmp_ptr_block_ptr_cell_unit_it->first_index_recurrent_connection_cell_input;
        
        for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
        {
            if(this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(tmp_connection_index,
                                                                                                                                                                                             tmp_ptr_array_parameters,
                                                                                                                                                                                             this->ptr_array_cell_units,
                                                                                                                                                                                             tmp_ptr_array_ptr_connections_layer_cell_units,
                                                                                                                                                                                             file) == false)
            {
                ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                         tmp_connection_index);

                return false;
            }
        }
        //    [1] |END| Recurrent, cell input. |END|
    }
    // [0] |END| Cell input. |END|

    // [0] Input, Gates.
    //  [1] Input gate.
    tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_feedforward_connection_input_gate);

    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_feedforward_connection_input_gate;
    
    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<U, E>(tmp_connection_index,
                                                                          tmp_ptr_array_parameters,
                                                                          ptr_first_U_unit_received,
                                                                          tmp_ptr_array_U_ptr_connections,
                                                                          file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                     tmp_connection_index);

            return false;
        }
    }
    //  [1] |END| Input gate. |END|

    //  [1] Forget gate.
    tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_feedforward_connection_forget_gate);
    
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_feedforward_connection_forget_gate;
    
    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<U, E>(tmp_connection_index,
                                                                          tmp_ptr_array_parameters,
                                                                          ptr_first_U_unit_received,
                                                                          tmp_ptr_array_U_ptr_connections,
                                                                          file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                     tmp_connection_index);

            return false;
        }
    }
    //  [1] |END| Forget gate. |END|

    //  [1] Output gate.
    tmp_ptr_array_U_ptr_connections = reinterpret_cast<U **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_feedforward_connection_output_gate);
    
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_feedforward_connection_output_gate;
    
    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_inputs_connections; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<U, E>(tmp_connection_index,
                                                                          tmp_ptr_array_parameters,
                                                                          ptr_first_U_unit_received,
                                                                          tmp_ptr_array_U_ptr_connections,
                                                                          file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                     tmp_connection_index);

            return false;
        }
    }
    //  [1] |END| Output gate. |END|
    // [0] |END| Input, Gates. |END|
    
    // [0] Recurrent, Gates.
    //  [1] Input gate.
    tmp_ptr_array_ptr_connections_layer_cell_units = reinterpret_cast<CellUnit **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_recurrent_connection_input_gate);

    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_recurrent_connection_input_gate;
    
    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(tmp_connection_index,
                                                                                                                                                                                         tmp_ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_layer_cell_units,
                                                                                                                                                                                         file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                     tmp_connection_index);

            return false;
        }
    }
    //  [1] |END| Input gate. |END|
    
    //  [1] Forget gate.
    tmp_ptr_array_ptr_connections_layer_cell_units = reinterpret_cast<CellUnit **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_recurrent_connection_forget_gate);
    
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_recurrent_connection_forget_gate;
    
    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(tmp_connection_index,
                                                                                                                                                                                         tmp_ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_layer_cell_units,
                                                                                                                                                                                         file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                     tmp_connection_index);

            return false;
        }
    }
    //  [1] |END| Forget gate. |END|
    
    //  [1] Output gate.
    tmp_ptr_array_ptr_connections_layer_cell_units = reinterpret_cast<CellUnit **>(this->ptr_array_ptr_connections + ptr_block_unit_it_received->first_index_recurrent_connection_output_gate);
    
    tmp_ptr_array_parameters = this->ptr_array_parameters + ptr_block_unit_it_received->first_index_recurrent_connection_output_gate;
    
    for(tmp_connection_index = 0_UZ; tmp_connection_index != tmp_number_recurrents_connection; ++tmp_connection_index)
    {
        if(this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(tmp_connection_index,
                                                                                                                                                                                         tmp_ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_layer_cell_units,
                                                                                                                                                                                         file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                     tmp_connection_index);

            return false;
        }
    }
    //  [1] |END| Output gate. |END|
    // [0] |END| Recurrent, Gates. |END|

#ifndef NO_PEEPHOLE
    // [0] Peepholes.
    //  [1] Input gate.
    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        if(this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(tmp_ptr_block_ptr_cell_unit_it->index_peephole_input_gate,
                                                                                                                                                                                         this->ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_peepholes_cell_units,
                                                                                                                                                                                         file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                     tmp_ptr_block_ptr_cell_unit_it->index_peephole_input_gate);

            return false;
        }
    }

    //  [1] Forget gate.
    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        if(this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(tmp_ptr_block_ptr_cell_unit_it->index_peephole_forget_gate,
                                                                                                                                                                                         this->ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_peepholes_cell_units,
                                                                                                                                                                                         file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                     tmp_ptr_block_ptr_cell_unit_it->index_peephole_forget_gate);

            return false;
        }
    }

    //  [1] Output gate.
    for(tmp_ptr_block_ptr_cell_unit_it = ptr_block_unit_it_received->ptr_array_cell_units; tmp_ptr_block_ptr_cell_unit_it != tmp_ptr_block_ptr_last_cell_unit; ++tmp_ptr_block_ptr_cell_unit_it)
    {
        if(this->Load_Dimension__Connection<CellUnit, LAYER::LSTM>(tmp_ptr_block_ptr_cell_unit_it->index_peephole_output_gate,
                                                                                                                                                                                         this->ptr_array_parameters,
                                                                                                                                                                                         this->ptr_array_cell_units,
                                                                                                                                                                                         tmp_ptr_array_ptr_connections_peepholes_cell_units,
                                                                                                                                                                                         file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Connection(%zu)\" function.",
                                     tmp_ptr_block_ptr_cell_unit_it->index_peephole_output_gate);

            return false;
        }
    }
    // [0] |END| Peepholes. |END|
#endif

    return true;
}

template<class U, LAYER::TYPE const E> bool Model::Load_Dimension__FC(Layer *const layer_it,
                                                                                                                                                                                     U *const ptr_first_U_unit_received,
                                                                                                                                                                                     std::wifstream &file)
{
    Neuron_unit const *const tmp_ptr_last_neuron_unit(layer_it->ptr_last_neuron_unit);
    Neuron_unit *tmp_ptr_neuron_unit_it;
    
    *layer_it->ptr_first_connection_index = this->total_weights;

    // Forward connection.
    for(tmp_ptr_neuron_unit_it = layer_it->ptr_array_neuron_units; tmp_ptr_neuron_unit_it != tmp_ptr_last_neuron_unit; ++tmp_ptr_neuron_unit_it)
    {
        if(this->Load_Dimension__Neuron(tmp_ptr_neuron_unit_it, file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Neuron()\" function.",);

            return false;
        }
        else if(this->Load_Dimension__Neuron__Forward__Connection<U, E>(tmp_ptr_neuron_unit_it,
                                                                                                              ptr_first_U_unit_received,
                                                                                                              file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Neuron__Forward__Connection()\" function.",);

            return false;
        }
    }

    *layer_it->ptr_last_connection_index = this->total_weights;

    return true;
}

bool Model::Load_Dimension__AF(Layer *const layer_it, std::wifstream &file)
{
    AF_unit const *const tmp_ptr_last_AF_unit(layer_it->ptr_last_AF_unit);
    AF_unit *tmp_ptr_AF_unit_it(layer_it->ptr_array_AF_units);
    
    for(; tmp_ptr_AF_unit_it != tmp_ptr_last_AF_unit; ++tmp_ptr_AF_unit_it)
    {
        if(this->Load_Dimension__AF(tmp_ptr_AF_unit_it, file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__AF_Unit(\" function.",);

            return false;
        }
    }

    return true;
}

bool Model::Load_Dimension__AF_Ind_Recurrent(Layer *const layer_it, std::wifstream &file)
{
    AF_Ind_recurrent_unit const *const tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit(layer_it->ptr_last_AF_Ind_recurrent_unit);
    AF_Ind_recurrent_unit *tmp_ptr_AF_Ind_recurrent_unit_it(layer_it->ptr_array_AF_Ind_recurrent_units);
    
    for(; tmp_ptr_AF_Ind_recurrent_unit_it != tmp_ptr_layer_ptr_last_AF_Ind_recurrent_unit; ++tmp_ptr_AF_Ind_recurrent_unit_it)
    {
        if(this->Load_Dimension__AF_Ind_Recurrent(tmp_ptr_AF_Ind_recurrent_unit_it, file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__AF_Ind_Recurrent(\" function.",);

            return false;
        }
    }

    return true;
}

template<class U, LAYER::TYPE const E> bool Model::Load_Dimension__LSTM(Layer *const layer_it,
                                                                                                                                                                                         U *const ptr_first_U_unit_received,
                                                                                                                                                                                         std::wifstream &file)
{
    BlockUnit const *const tmp_ptr_last_block_unit(layer_it->ptr_last_block_unit);
    BlockUnit *tmp_ptr_block_unit_it(layer_it->ptr_array_block_units);
    
    size_t const tmp_layer_number_block_units(static_cast<size_t>(tmp_ptr_last_block_unit - tmp_ptr_block_unit_it)),
                       tmp_layer_number_cell_units(static_cast<size_t>(layer_it->ptr_last_cell_unit - layer_it->ptr_array_cell_units));
    
    LAYER_NORM::TYPE const tmp_type_layer_normalization(layer_it->type_normalization);
    
    *layer_it->ptr_first_connection_index = this->total_weights;

    for(; tmp_ptr_block_unit_it != tmp_ptr_last_block_unit; ++tmp_ptr_block_unit_it)
    {
        if(this->Load_Dimension__Block(tmp_layer_number_block_units,
                                                        tmp_layer_number_cell_units >> static_cast<size_t>(layer_it->Use__Bidirectional()),
                                                        tmp_type_layer_normalization,
                                                        tmp_ptr_block_unit_it,
                                                        file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Block(%zu, %zu, %u, ptr, ref)\" function.",
                                     tmp_layer_number_block_units,
                                     tmp_layer_number_cell_units >> static_cast<size_t>(layer_it->Use__Bidirectional()),
                                     tmp_type_layer_normalization);

            return false;
        }
        else if(this->Load_Dimension__Block__Connection<U, E>(tmp_ptr_block_unit_it,
                                                                                             ptr_first_U_unit_received,
                                                                                             file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__LSTM__Connection()\" function.",);

            return false;
        }
    }

    *layer_it->ptr_last_connection_index = this->total_weights;

    return true;
}

bool Model::Load_Dimension__Normalization(Layer *const layer_it, std::wifstream &file)
{
    LAYER_NORM::TYPE const tmp_type_layer_normalization(layer_it->type_normalization);

    union Normalized_unit const *const tmp_ptr_last_normalized_unit(layer_it->ptr_last_normalized_unit);
    union Normalized_unit *tmp_ptr_normalized_unit_it(layer_it->ptr_array_normalized_units);
    
    size_t const tmp_number_normalized_units(static_cast<size_t>(tmp_ptr_last_normalized_unit - tmp_ptr_normalized_unit_it));
    
    for(; tmp_ptr_normalized_unit_it != tmp_ptr_last_normalized_unit; ++tmp_ptr_normalized_unit_it)
    {
        if(this->Load_Dimension__Normalized_Unit(tmp_number_normalized_units,
                                                                       tmp_type_layer_normalization,
                                                                       tmp_ptr_normalized_unit_it,
                                                                       file) == false)
        {
            ERR(L"An error has been triggered from the \"Load_Dimension__Normalized_Unit(%zu, %u)\" function.",
                                     tmp_number_normalized_units,
                                     tmp_type_layer_normalization);

            return false;
        }
    }

    return true;
}
}