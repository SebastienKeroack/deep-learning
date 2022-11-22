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
#include "deep-learning-lib/io/term/input.hpp"

#include <iostream>
#include <array>

using namespace DL::Term;

namespace DL::v1 {
bool Model::User_Controls__Weights_Initializer(void) {
  struct Weights_Initializer tmp_Weights_Initializer;

  if (tmp_Weights_Initializer.Input_Initialize() == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Input_Initialize()\" function.",);

    return false;
  }

  if (tmp_Weights_Initializer.Output_Initialize(this) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Output_Initialize()\" function.",);

    return false;
  }

  return true;
}

bool Model::User_Controls__Optimizer_Function_Initializer(void) {
  struct Optimizer_Function_Initializer tmp_Optimizer_Function_Initializer;

  if (tmp_Optimizer_Function_Initializer.Input_Initialize() == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Input_Initialize()\" function.",);

    return false;
  }

  if (tmp_Optimizer_Function_Initializer.Output_Initialize(this) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Output_Initialize()\" function.",);

    return false;
  }

  return true;
}

bool Model::User_Controls__Loss_Function_Initializer(void) {
  struct Loss_Function_Initializer tmp_Loss_Function_Initializer;

  if (tmp_Loss_Function_Initializer.Input_Initialize() == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Input_Initialize()\" function.",);

    return false;
  }

  tmp_Loss_Function_Initializer.Output_Initialize(this);

  return true;
}

bool Model::User_Controls__Accuracy_Function_Initializer(void) {
  struct Accuracy_Function_Initializer tmp_Accuracy_Function_Initializer;

  if (tmp_Accuracy_Function_Initializer.Input_Initialize() == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Input_Initialize()\" function.",);

    return false;
  }

  tmp_Accuracy_Function_Initializer.Output_Initialize(this);

  return true;
}

bool Model::User_Controls__Optimizer_Function(void) {
  if (this->Usable_Warm_Restarts()) {
    while (true) {
      INFO(L"");
      INFO(L"User controls, optimizer function %ls:",
             OPTIMIZER_NAME[this->type_optimizer_function].c_str());
      INFO(L"[0]: Modify optimizer function hyper-parameters.");
      INFO(L"[1]: Modify warm restarts.");
      INFO(L"[2]: Change optimizer function.");
      INFO(L"[3]: Quit.");

      switch (parse_discrete<int>(
          0, 3, L"Option: ")) {
        case 0:
          switch (this->type_optimizer_function) {
            case OPTIMIZER::GD:
              if (this->User_Controls__Optimizer__Gradient_Descent() == false) {
                ERR(
                    L"An error has been triggered from the "
                    "\"User_Controls__Optimizer__Gradient_Descent()\" "
                    "function.",);

                return false;
              }
              break;
            case OPTIMIZER::IRPROP_MINUS:
            case OPTIMIZER::IRPROP_PLUS:
              if (this->User_Controls__Optimizer__iRPROP() == false) {
                ERR(
                    L"An error has been triggered from the "
                    "\"User_Controls__Optimizer__iRPROP()\" function.",);

                return false;
              }
              break;
            case OPTIMIZER::ADAM:
            case OPTIMIZER::ADAMAX:
            case OPTIMIZER::AMSGRAD:
              if (this->User_Controls__Optimizer__Adam() == false) {
                ERR(
                    L"An error has been triggered from the "
                    "\"User_Controls__Optimizer__Adam()\" function.",);

                return false;
              }
              break;
            case OPTIMIZER::NOSADAM:
              if (this->User_Controls__Optimizer__NosAdam() == false) {
                ERR(
                    L"An error has been triggered from the "
                    "\"User_Controls__Optimizer__NosAdam()\" function.",);

                return false;
              }
              break;
            case OPTIMIZER::ADABOUND:
            case OPTIMIZER::AMSBOUND:
              if (this->User_Controls__Optimizer__AdaBound() == false) {
                ERR(
                    L"An error has been triggered from the "
                    "\"User_Controls__Optimizer__AdaBound()\" function.",);

                return false;
              }
              break;
            default:
              ERR(L"Unknow type optimizer function (%u | %ls) in the switch.",
                  this->type_optimizer_function,
                  OPTIMIZER_NAME[this->type_optimizer_function].c_str());
              break;
          }
          break;
        case 1:
          if (this->User_Controls__Warm_Restarts() == false) {
            ERR(
                L"An error has been triggered from the "
                "\"User_Controls__Warm_Restarts()\" function.",);

            return false;
          }
          break;
        case 2:
          if (this->User_Controls__Optimizer_Function_Initializer() == false) {
            ERR(
                L"An error has been triggered from the "
                "\"User_Controls__Optimizer_Function_Initializer()\" function.",);

            return false;
          }
          break;
        case 3:
          return true;
        default:
          ERR(
              L"An error has been triggered from the "
              "\"parse_discrete<int>(%u, %u)\" function.", 0, 3u);
          break;
      }
    }
  } else {
    while (true) {
      INFO(L"");
      INFO(L"User controls, optimizer function %ls:",
             OPTIMIZER_NAME[this->type_optimizer_function].c_str());
      INFO(L"[0]: Modify optimizer function hyper-parameters.");
      INFO(L"[1]: Change optimizer function.");
      INFO(L"[2]: Quit.");

      switch (parse_discrete<int>(
          0, 2, L"Option: ")) {
        case 0:
          switch (this->type_optimizer_function) {
            case OPTIMIZER::GD:
              if (this->User_Controls__Optimizer__Gradient_Descent() == false) {
                ERR(
                    L"An error has been triggered from the "
                    "\"User_Controls__Optimizer__Gradient_Descent()\" "
                    "function.",);

                return false;
              }
              break;
            case OPTIMIZER::IRPROP_MINUS:
            case OPTIMIZER::IRPROP_PLUS:
              if (this->User_Controls__Optimizer__iRPROP() == false) {
                ERR(
                    L"An error has been triggered from the "
                    "\"User_Controls__Optimizer__iRPROP()\" function.",);

                return false;
              }
              break;
            case OPTIMIZER::ADAM:
            case OPTIMIZER::ADAMAX:
            case OPTIMIZER::AMSGRAD:
              if (this->User_Controls__Optimizer__Adam() == false) {
                ERR(
                    L"An error has been triggered from the "
                    "\"User_Controls__Optimizer__Adam()\" function.",);

                return false;
              }
              break;
            case OPTIMIZER::NOSADAM:
              if (this->User_Controls__Optimizer__NosAdam() == false) {
                ERR(
                    L"An error has been triggered from the "
                    "\"User_Controls__Optimizer__NosAdam()\" function.",);

                return false;
              }
              break;
            case OPTIMIZER::ADABOUND:
            case OPTIMIZER::AMSBOUND:
              if (this->User_Controls__Optimizer__AdaBound() == false) {
                ERR(
                    L"An error has been triggered from the "
                    "\"User_Controls__Optimizer__AdaBound()\" function.",);

                return false;
              }
              break;
            default:
              ERR(L"Unknow type optimizer function (%u | %ls) in the switch.",
                  this->type_optimizer_function,
                  OPTIMIZER_NAME[this->type_optimizer_function].c_str());
              break;
          }
          break;
        case 1:
          if (this->User_Controls__Optimizer_Function_Initializer() == false) {
            ERR(
                L"An error has been triggered from the "
                "\"User_Controls__Optimizer_Function_Initializer()\" function.",);

            return false;
          }
          break;
        case 2:
          return true;
        default:
          ERR(
              L"An error has been triggered from the "
              "\"parse_discrete<int>(%u, %u)\" function.", 0, 2u);
          break;
      }
    }
  }

  return false;
}

bool Model::User_Controls__Optimizer__Gradient_Descent(void) {
  bool tmp_parameters_has_change(false);

  real tmp_hyper_paramater;

  while (true) {
    INFO(L"");
    INFO(L"User controls, %ls:",
           OPTIMIZER_NAME[this->type_optimizer_function].c_str());
    INFO(L"[0]: Modify learning rate (%.9f).", this->learning_rate);
    INFO(L"[1]: Modify learning momentum (%.9f).", this->learning_momentum);
    INFO(L"[2]: Use Nesterov (%ls).", this->use_nesterov ? "Yes" : "No");
    INFO(L"[3]: Quit.");

    switch (parse_discrete<int>(
        0, 3, L"Option: ")) {
      case 0:
        INFO(L"");
        INFO(L"Learning rate.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=0.01.");

        tmp_hyper_paramater = this->learning_rate;

        this->learning_rate = parse_real(
            0_r, L"Learning rate: ");

        if (tmp_hyper_paramater != this->learning_rate) {
          tmp_parameters_has_change = true;
        }
        break;
      case 1:
        INFO(L"");
        INFO(L"Learning momentum.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=0.9.");

        tmp_hyper_paramater = this->learning_momentum;

        this->learning_momentum = parse_real(
            0_r, L"Learning rate: ");

        if (tmp_hyper_paramater != this->learning_momentum) {
          tmp_parameters_has_change = true;
        }

        if (tmp_hyper_paramater == 0_r) {
          if (this->Allocate__Parameter__Gradient_Descent() == false) {
            ERR(
                L"An error has been triggered from the "
                "\"Allocate__Parameter__Gradient_Descent()\" function.",);

            return false;
          }
        } else if (this->learning_momentum == 0_r) {
          this->Deallocate__Parameter__Gradient_Descent();
        }
        break;
      case 2:
        INFO(L"");
        if (this->learning_momentum != 0_r) {
          this->use_nesterov = accept(L"Do you want to use Nesterov?");
        } else {
          INFO(L"WARNING: Can not use Nesterov without momentum.");
        }
        break;
      case 3:
        if (this->is_cu_initialized && tmp_parameters_has_change) {
#ifdef COMPILE_CUDA
          this->cumodel->Copy__Gradient_Descent_Parameters(
              this);
#else
          ERR(L"`CUDA` functionality was not built. Pass `-DCOMPILE_CUDA` to the compiler.");
#endif
        }
        return true;
      default:
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 0, 3u);
        break;
    }
  }

  return false;
}

bool Model::User_Controls__Optimizer__iRPROP(void) {
#ifdef COMPILE_CUDA
  bool tmp_parameters_has_change(false);

  real tmp_hyper_paramater;
#endif

  while (true) {
    INFO(L"");
    INFO(L"User controls, %ls:",
           OPTIMIZER_NAME[this->type_optimizer_function].c_str());
    INFO(L"[0]: Modify increase factor (%.9f).",
           this->rprop_increase_factor);
    INFO(L"[1]: Modify decrease factor (%.9f).",
           this->rprop_decrease_factor);
    INFO(L"[2]: Modify delta maximum (%.9f).", this->rprop_delta_max);
    INFO(L"[3]: Modify delta minimum (%.9f).", this->rprop_delta_min);
    INFO(L"[4]: Modify delta zero (%.9f).", this->rprop_delta_zero);
    INFO(L"[5]: Quit.");

    switch (parse_discrete<int>(
        0, 5, L"Option: ")) {
      case 0:
        INFO(L"");
        INFO(L"Increase factor.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=1.2.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->rprop_increase_factor;
#endif

        this->rprop_increase_factor = parse_real(
            0_r, L"Increase factor: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->rprop_increase_factor) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 1:
        INFO(L"");
        INFO(L"Decrease factor.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=0.5.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->rprop_decrease_factor;
#endif

        this->rprop_decrease_factor = parse_real(
            0_r, L"Decrease factor: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->rprop_decrease_factor) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 2:
        INFO(L"");
        INFO(L"Delta maximum.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=50.0.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->rprop_delta_max;
#endif

        this->rprop_delta_max = parse_real(
            0_r, L"Delta maximum: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->rprop_delta_max) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 3:
        INFO(L"");
        INFO(L"Delta minimum.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=1e-6.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->rprop_delta_min;
#endif

        this->rprop_delta_min = parse_real(
            0_r, L"Delta minimum: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->rprop_delta_min) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 4:
        INFO(L"");
        INFO(L"Delta zero.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=0.1.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->rprop_delta_zero;
#endif

        this->rprop_delta_zero = parse_real(
            0_r, L"Delta zero: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->rprop_delta_zero) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 5:
#ifdef COMPILE_CUDA
        if (this->is_cu_initialized && tmp_parameters_has_change) {
          this->cumodel->Copy__RPROP_minus_Parameters(this);
        }
#endif
        return true;
      default:
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 0, 5u);
        break;
    }
  }

  return false;
}

bool Model::User_Controls__Optimizer__AdaBound(void) {
#ifdef COMPILE_CUDA
  bool tmp_parameters_has_change(false);

  real tmp_hyper_paramater;
#endif

  while (true) {
    INFO(L"");
    INFO(L"User controls, %ls:",
           OPTIMIZER_NAME[this->type_optimizer_function].c_str());
    INFO(L"[0]: Modify learning rate (%.9f).", this->adam_learning_rate);
    INFO(L"[1]: Modify learning rate, final (%.9f).", this->learning_rate_final);
    INFO(L"[2]: Modify beta1 (%.9f).", this->adam_beta1);
    INFO(L"[3]: Modify beta2 (%.9f).", this->adam_beta2);
    INFO(L"[4]: Modify epsilon (%.9f).", this->adam_epsilon);
    INFO(L"[5]: Bias correction (%ls).",
           this->use_adam_bias_correction ? "true" : "false");
    INFO(L"[6]: Modify gamma (%.9f).", this->learning_gamma);
    INFO(L"[7]: Quit.");

    switch (parse_discrete<int>(
        0, 7, L"Option: ")) {
      case 0:
        INFO(L"");
        INFO(L"Learning rate.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=0.001.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_learning_rate;
#endif

        this->adam_learning_rate = parse_real(
            0_r, L"Learning rate: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_learning_rate) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 1:
        INFO(L"");
        INFO(L"Learning rate, final.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=0.1.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->learning_rate_final;
#endif

        this->learning_rate_final = parse_real(
            0_r, L"Learning rate, final: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->learning_rate_final) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 2:
        INFO(L"");
        INFO(L"Beta1.");
        INFO(L"Range[0.0, 0.99...9].");
        INFO(L"default=0.9.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_beta1;
#endif

        this->adam_beta1 = parse_real(
            0_r, 1_r - 1e-7_r, L"Beta1: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_beta1) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 3:
        INFO(L"");
        INFO(L"Beta2.");
        INFO(L"Range[0.0, 0.99...9].");
        INFO(L"default=0.999.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_beta2;
#endif

        this->adam_beta2 = parse_real(
            0_r, 1_r - 1e-7_r, L"Beta2: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_beta2) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 4:
        INFO(L"");
        INFO(L"Epsilon.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=1e-8.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_epsilon;
#endif

        this->adam_epsilon = parse_real(
            0_r, L"Epsilon: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_epsilon) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 5:
        INFO(L"");
        INFO(L"Bias correction.");
        INFO(L"default=true.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = static_cast<real>(this->use_adam_bias_correction);
#endif

        this->use_adam_bias_correction =
            accept(L"Bias correction: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater !=
            static_cast<real>(this->use_adam_bias_correction)) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 6:
        INFO(L"");
        INFO(L"Gamma.");
        INFO(L"Range[0.0, 0.99...9].");
        INFO(L"default=1e-3.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->learning_gamma;
#endif

        this->learning_gamma = parse_real(
            0_r, 1_r - 1e-7_r, L"Gamma: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->learning_gamma) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 7:
#ifdef COMPILE_CUDA
        if (this->is_cu_initialized && tmp_parameters_has_change) {
          this->cumodel->Copy__AdaBound_Parameters(this);
        }
#endif
        return true;
      default:
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 0, 7u);
        break;
    }
  }

  return false;
}

bool Model::User_Controls__Optimizer__Adam(void) {
#ifdef COMPILE_CUDA
  bool tmp_parameters_has_change(false);

  real tmp_hyper_paramater;
#endif

  while (true) {
    INFO(L"");
    INFO(L"User controls, %ls:",
           OPTIMIZER_NAME[this->type_optimizer_function].c_str());
    INFO(L"[0]: Modify learning rate (%.9f).", this->adam_learning_rate);
    INFO(L"[1]: Modify beta1 (%.9f).", this->adam_beta1);
    INFO(L"[2]: Modify beta2 (%.9f).", this->adam_beta2);
    INFO(L"[3]: Modify epsilon (%.9f).", this->adam_epsilon);
    INFO(L"[4]: Bias correction (%ls).",
           this->use_adam_bias_correction ? "true" : "false");
    INFO(L"[5]: Quit.");

    switch (parse_discrete<int>(
        0, 5, L"Option: ")) {
      case 0:
        INFO(L"");
        INFO(L"Learning rate.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=0.001.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_learning_rate;
#endif

        this->adam_learning_rate = parse_real(
            0_r, L"Learning rate: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_learning_rate) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 1:
        INFO(L"");
        INFO(L"Beta1.");
        INFO(L"Range[0.0, 0.99...9].");
        INFO(L"default=0.9.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_beta1;
#endif

        this->adam_beta1 = parse_real(
            0_r, 1_r - 1e-7_r, L"Beta1: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_beta1) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 2:
        INFO(L"");
        INFO(L"Beta2.");
        INFO(L"Range[0.0, 0.99...9].");
        INFO(L"default=0.999.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_beta2;
#endif

        this->adam_beta2 = parse_real(
            0_r, 1_r - 1e-7_r, L"Beta2: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_beta2) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 3:
        INFO(L"");
        INFO(L"Epsilon.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=1e-8.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_epsilon;
#endif

        this->adam_epsilon = parse_real(
            0_r, L"Epsilon: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_epsilon) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 4:
        INFO(L"");
        INFO(L"Bias correction.");
        INFO(L"default=true.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = static_cast<real>(this->use_adam_bias_correction);
#endif

        this->use_adam_bias_correction =
            accept(L"Bias correction: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater !=
            static_cast<real>(this->use_adam_bias_correction)) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 5:
#ifdef COMPILE_CUDA
        if (this->is_cu_initialized && tmp_parameters_has_change) {
          this->cumodel->Copy__Adam_Parameters(this);
        }
#endif
        return true;
      default:
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 0, 5u);
        break;
    }
  }

  return false;
}

bool Model::User_Controls__Optimizer__NosAdam(void) {
#ifdef COMPILE_CUDA
  bool tmp_parameters_has_change(false);

  real tmp_hyper_paramater;
#endif

  while (true) {
    INFO(L"");
    INFO(L"User controls, %ls:",
           OPTIMIZER_NAME[this->type_optimizer_function].c_str());
    INFO(L"[0]: Modify learning rate (%.9f).", this->adam_learning_rate);
    INFO(L"[1]: Modify beta1 (%.9f).", this->adam_beta1);
    INFO(L"[2]: Modify beta2 (%.9f).", this->adam_beta2);
    INFO(L"[3]: Modify epsilon (%.9f).", this->adam_epsilon);
    INFO(L"[4]: Bias correction (%ls).",
           this->use_adam_bias_correction ? "true" : "false");
    INFO(L"[5]: Modify gamma (%.9f).", this->adam_gamma);
    INFO(L"[6]: Quit.");

    switch (parse_discrete<int>(
        0, 6, L"Option: ")) {
      case 0:
        INFO(L"");
        INFO(L"Learning rate.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=0.001.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_learning_rate;
#endif

        this->adam_learning_rate = parse_real(
            0_r, L"Learning rate: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_learning_rate) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 1:
        INFO(L"");
        INFO(L"Beta1.");
        INFO(L"Range[0.0, 0.99...9].");
        INFO(L"default=0.9.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_beta1;
#endif

        this->adam_beta1 = parse_real(
            0_r, 1_r - 1e-7_r, L"Beta1: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_beta1) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 2:
        INFO(L"");
        INFO(L"Beta2.");
        INFO(L"Range[0.0, 0.99...9].");
        INFO(L"default=0.999.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_beta2;
#endif

        this->adam_beta2 = parse_real(
            0_r, 1_r - 1e-7_r, L"Beta2: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_beta2) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 3:
        INFO(L"");
        INFO(L"Epsilon.");
        INFO(L"Range[0.0, inf].");
        INFO(L"default=1e-8.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_epsilon;
#endif

        this->adam_epsilon = parse_real(
            0_r, L"Epsilon: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_epsilon) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 4:
        INFO(L"");
        INFO(L"Bias correction.");
        INFO(L"default=true.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = static_cast<real>(this->use_adam_bias_correction);
#endif

        this->use_adam_bias_correction =
            accept(L"Bias correction: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater !=
            static_cast<real>(this->use_adam_bias_correction)) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 5:
        INFO(L"");
        INFO(L"Gamma.");
        INFO(L"Range[1e-7, inf].");
        INFO(L"default=0.1.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->adam_gamma;
#endif

        this->adam_gamma = parse_real(
            1e-7_r, L"Gamma: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->adam_gamma) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 6:
#ifdef COMPILE_CUDA
        if (this->is_cu_initialized && tmp_parameters_has_change) {
          this->cumodel->Copy__Adam_Parameters(this);
        }
#endif
        return true;
      default:
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 0, 6u);
        break;
    }
  }

  return false;
}

bool Model::User_Controls__Warm_Restarts(void) {
#ifdef COMPILE_CUDA
  bool tmp_parameters_has_change(false);

  real tmp_hyper_paramater;
#endif

  while (true) {
    INFO(L"");
    INFO(L"User controls, warm restarts:");
    INFO(L"[0]: Use warm restarts (%ls).",
           this->use_warm_restarts ? "Yes" : "No");
    INFO(L"[1]: Modify learning rate, decay (%.9f).",
           this->warm_restarts_decay_learning_rate);
    INFO(L"[2]: Modify maximum learning rate (%.9f).",
           this->warm_restarts_initial_maximum_learning_rate);
    INFO(L"[3]: Modify minimum learning rate (%.9f).",
           this->warm_restarts_minimum_learning_rate);
    INFO(L"[4]: Modify initial Ti (%.9f).",
           this->warm_restarts_initial_T_i);
    INFO(L"[5]: Modify warm restarts multiplier (%.9f).",
           this->warm_restarts_multiplier);
    INFO(L"[6]: Quit.");

    switch (parse_discrete<int>(
        0, 6, L"Option: ")) {
      case 0:
        INFO(L"");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = static_cast<real>(this->use_warm_restarts);
#endif

        this->use_warm_restarts = accept(L"Use warm restarts: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != static_cast<real>(this->use_warm_restarts)) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 1:
        INFO(L"");
        INFO(L"Learning rate, decay:");
        INFO(L"Range[1e-5, 1.0].");
        INFO(L"default=0.95.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->warm_restarts_decay_learning_rate;
#endif

        this->warm_restarts_decay_learning_rate =
            parse_real(
                1e-5_r, 1_r, L"Learning rate, decay: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->warm_restarts_decay_learning_rate) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 2:
        INFO(L"");
        INFO(L"Maximum learning rate:");
        INFO(L"Range[0.0, 1.0].");
        INFO(L"default=1.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->warm_restarts_initial_maximum_learning_rate;
#endif

        this->warm_restarts_initial_maximum_learning_rate =
            parse_real(
                0_r, 1_r, L"Maximum learning rate: ");

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater !=
            this->warm_restarts_initial_maximum_learning_rate) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 3:
        INFO(L"");
        INFO(L"Minimum learning rate:");
        INFO(L"Range[0.0, %f].",
               this->warm_restarts_initial_maximum_learning_rate);
        INFO(L"default=0.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->warm_restarts_minimum_learning_rate;
#endif

        this->warm_restarts_minimum_learning_rate =
            parse_real(
                0_r, this->warm_restarts_initial_maximum_learning_rate, L"Minimum learning rate: ");
        if (this->warm_restarts_minimum_learning_rate == 0_r) {
          this->warm_restarts_minimum_learning_rate =
              this->warm_restarts_initial_maximum_learning_rate / 1e+7_r;
        }

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->warm_restarts_minimum_learning_rate) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 4:
        INFO(L"");
        INFO(L"Initial Ti:");
        INFO(L"Range[0, inf].");
        INFO(L"default=1.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->warm_restarts_initial_T_i;
#endif

        this->warm_restarts_initial_T_i =
            static_cast<real>(parse_discrete<size_t>(
                0_UZ, L"Initial Ti: "));

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->warm_restarts_initial_T_i) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 5:
        INFO(L"");
        INFO(L"Warm restarts multiplier:");
        INFO(L"Range[0, inf].");
        INFO(L"default=2.");

#ifdef COMPILE_CUDA
        tmp_hyper_paramater = this->warm_restarts_multiplier;
#endif

        this->warm_restarts_multiplier =
            static_cast<real>(parse_discrete<size_t>(
                0_UZ, L"Warm restarts multiplier: "));

#ifdef COMPILE_CUDA
        if (tmp_hyper_paramater != this->warm_restarts_multiplier) {
          tmp_parameters_has_change = true;
        }
#endif
        break;
      case 6:
#ifdef COMPILE_CUDA
        if (this->is_cu_initialized && tmp_parameters_has_change) {
          this->cumodel->Copy__Warm_Restarts_Parameters(this);
        }
#endif
        return true;
      default:
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 0, 6u);
        break;
    }
  }

  return false;
}

bool Model::User_Controls__Accuracy_Variance(void) {
  INFO(L"");
  INFO(L"Accuracy variance (%f).",
         this->acc_var);
  INFO(L"Range[0.0, 1.0].");
  if (this->Set__Accurancy_Variance(parse_real(
          0_r, 1_r, L"Accuracy variance: ")) ==
      false) {
    ERR(
        L"An error has been triggered from the "
        "\"Set__Accurancy_Variance()\" function.",);

    return false;
  }

  return true;
}

bool Model::User_Controls__Time_Delays(void) {
  INFO(L"");
  INFO(L"Time delays (%zu).",
         this->n_time_delay);
  INFO(L"Range[0, %zu].",
         this->seq_w - 1_UZ);
  if (this->set_seq_w(parse_discrete<size_t>(
          0_UZ, this->seq_w - 1_UZ, L"Time delays: ")) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"set_seq_w()\" function.",);

    return false;
  }

  return true;
}

bool Model::User_Controls__Clip_Gradient(void) {
  INFO(L"");
  INFO(L"clip gradient:");
  this->set_clip_gradient(accept(L"Do you want to use clip gradient?"));

  if (this->Use__Clip_Gradient()) {
    INFO(L"");
    INFO(L"clip gradient:");
    INFO(L"Range[0 , inf].");
    INFO(L"default=1.0.");
    if (this->set_clip_gradient(parse_real(
            0_r, L"clip gradient: ")) == false) {
      ERR(
          L"An error has been triggered from the "
          "\"set_clip_gradient()\" function.",);

      return false;
    }
  }

  return true;
}

bool Model::User_Controls__Max_Norm_Constaints(void) {
  INFO(L"");
  INFO(L"Max-norm constraints:");
  INFO(L"Range[0, inf]. Off = 0.");
  INFO(L"default=4.0.");
  if (this->Set__Regularization__Max_Norm_Constraints(
          parse_real(
              0_r, L"Max-norm constraints: ")) ==
      false) {
    ERR(
        L"An error has been triggered from the "
        "\"Set__Regularization__Max_Norm_Constraints()\" function.",);

    return false;
  }

  return true;
}

bool Model::User_Controls__L1_Regularization(void) {
  INFO(L"");
  INFO(L"L1:");
  INFO(L"Range[0.0, 1.0]. Off = 0.");
  INFO(L"default=0.0.");
  if (this->Set__Regularization__L1(parse_real(
          0_r, 1_r, L"L1: ")) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Set__Regularization__L1()\" function.",);

    return false;
  }

  return true;
}

bool Model::User_Controls__L2_Regularization(void) {
  INFO(L"");
  INFO(L"L2:");
  INFO(L"Range[0.0, 1.0]. Off = 0.");
  INFO(L"default=1e-5.");
  if (this->Set__Regularization__L2(parse_real(
          0_r, 1_r, L"L2: ")) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Set__Regularization__L2()\" function.",);

    return false;
  }

  return true;
}

bool Model::User_Controls__SRIP_Regularization(void) {
  INFO(L"");
  INFO(L"SRIP:");
  INFO(L"Range[0.0, 1.0]. Off = 0.");
  INFO(L"default=1e-5.");
  if (this->Set__Regularization__SRIP(parse_real(
          0_r, 1_r, L"SRIP: ")) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Set__Regularization__SRIP()\" function.",);

    return false;
  }

  return true;
}

bool Model::User_Controls__Maximum__Batch_Size(void) {
  INFO(L"");
  INFO(L"Maximum batch size:");
  INFO(L"Range[1, inf].");
  INFO(L"default=8192.");
  if (this->set_max_batch_size(parse_discrete<size_t>(
          1_UZ, L"Maximum batch size: ")) ==
      false) {
    ERR(
        L"An error has been triggered from the "
        "\"set_max_batch_size()\" function.",);

    return false;
  }

  return true;
}

bool Model::User_Controls__OpenMP(void) {
  while (true) {
    INFO(L"");
    INFO(L"User controls, OpenMP:");
    INFO(L"[0]: Use OpenMP (%ls | %ls).", this->use_mp ? "Yes" : "No",
           this->is_mp_initialized ? "Yes" : "No");
    INFO(L"[1]: Maximum threads (%.2f%%).", this->pct_threads);
    INFO(L"[2]: Quit.");

    switch (parse_discrete<int>(
        0, 2, L"Option: ")) {
      case 0:
        INFO(L"");
        if (this->set_mp(accept(L"Use OpenMP: ")) == false) {
          ERR(
              L"An error has been triggered from the "
              "\"set_mp()\" function.",);

          return false;
        }
        break;
      case 1:
        INFO(L"");
        INFO(L"Maximum threads:");
        INFO(L"Range[0.0%%, 100.0%%].");
        if (this->Set__Maximum_Thread_Usage(parse_real(
                0_r, 100_r, L"Maximum threads (percent): ")) ==
            false) {
          ERR(
              L"An error has been triggered from the "
              "\"Set__Maximum_Thread_Usage()\" function.",);

          return false;
        }
        break;
      case 2:
        return true;
      default:
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 0, 2u);
        break;
    }
  }

  return false;
}

bool Model::User_Controls__Regularization(void) {
  while (true) {
    INFO(L"");
    INFO(L"User controls, regularization:");
    INFO(L"[0]: Modify max-norm constraints (%.9f).",
           this->regularization__max_norm_constraints);
    INFO(L"[1]: Modify L1 (%.9f).",
           this->regularization__l1);
    INFO(L"[2]: Modify L2 (%.9f).",
           this->regularization__l2);
    INFO(L"[3]: Modify SRIP (%.9f).",
           this->regularization__srip);
    INFO(L"[4]: Modify weight decay (%.9f).",
           this->weight_decay);
    INFO(L"[5]: Modify dropout.");
    INFO(L"[6]: Modify normalization.");
    INFO(L"[7]: Modify tied parameter.");
    INFO(L"[8]: Modify k-sparse.");
    INFO(L"[9]: Quit.");

    switch (parse_discrete<int>(
        0, 9, L"Option: ")) {
      case 0:
        if (this->User_Controls__Max_Norm_Constaints() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Max_Norm_Constaints()\" function.",);

          return false;
        }
        break;
      case 1:
        if (this->User_Controls__L1_Regularization() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__L1_Regularization()\" function.",);

          return false;
        }
        break;
      case 2:
        if (this->User_Controls__L2_Regularization() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__L2_Regularization()\" function.",);

          return false;
        }
        break;
      case 3:
        if (this->User_Controls__SRIP_Regularization() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__SRIP_Regularization()\" function.",);

          return false;
        }
        break;
      case 4:
        INFO(L"");
        INFO(L"Weight decay:");
        INFO(L"Range[0.0, 1.0]. Off = 0.");
        INFO(L"default=1e-5.");
        if (this->Set__Regularization__Weight_Decay(
                parse_real(
                    0_r, 1_r, L"Weight decay: ")) ==
            false) {
          ERR(
              L"An error has been triggered from the "
              "\"Set__Regularization__Weight_Decay()\" function.",);

          return false;
        }
        break;
      case 5:
        if (this->User_Controls__Dropout() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Dropout()\" function.",);

          return false;
        }
        break;
      case 6:
        if (this->User_Controls__Normalization() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Normalization()\" function.",);

          return false;
        }
        break;
      case 7:
        if (this->User_Controls__Tied__Parameter() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Tied__Parameter()\" function.",);

          return false;
        }
        break;
      case 8:
        if (this->User_Controls__K_Sparse() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__K_Sparse()\" function.",);

          return false;
        }
        break;
      case 9:
        return true;
      default:
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 0, 9u);
        break;
    }
  }

  return false;
}

bool Model::User_Controls__Dropout(void) {
  long long int tmp_option;

  unsigned int tmp_type_dropout_layer_index;

  size_t const tmp_option_end(this->type == MODEL::AUTOENCODER
                                  ? this->total_layers / 2_UZ + 1_UZ
                                  : this->total_layers - 1_UZ);
  size_t tmp_layer_index;

  real tmp_hyper_parameters[2] = {0};

  real tmp_parameters[2] = {0};

  LAYER_DROPOUT::TYPE tmp_type_layer_dropout;

  struct Layer *layer_it;

  std::wstring tmp_layer_name;

  while (true) {
    INFO(L"");
    INFO(L"User controls, Layer dropout:");
    INFO(L"[-1]: All.");
    INFO(L"[0]: Input layer: (%f, %f, %f), %ls.",
           this->ptr_array_layers[0].dropout_values[0],
           this->ptr_array_layers[0].dropout_values[1],
           this->ptr_array_layers[0].dropout_values[2],
           LAYER_DROPOUT_NAME[this->ptr_array_layers[0].type_dropout]
               .c_str());
    for (tmp_layer_index = 1_UZ; tmp_layer_index != tmp_option_end;
         ++tmp_layer_index) {
      INFO(L"[%zu]: Hidden layer[%zu]: (%f, %f, %f), %ls.", tmp_layer_index,
             tmp_layer_index - 1_UZ,
             this->ptr_array_layers[tmp_layer_index].dropout_values[0],
             this->ptr_array_layers[tmp_layer_index].dropout_values[1],
             this->ptr_array_layers[tmp_layer_index].dropout_values[2],
             LAYER_DROPOUT_NAME[this->ptr_array_layers[tmp_layer_index]
                                        .type_dropout]
                 .c_str());
    }
    INFO(L"[%zu]: Quit.",
           tmp_option_end);

    tmp_option = parse_discrete<long long>(
        -1ll, static_cast<long long int>(tmp_option_end), L"Option: ");

    if (tmp_option < static_cast<long long int>(tmp_option_end)) {
      layer_it = this->ptr_array_layers + tmp_option;

      tmp_layer_name = tmp_option == 0ll
                           ? L"Input"
                           : L"Hidden[" + std::to_wstring(tmp_option - 1ll) + L"]";

      INFO(L"");
      INFO(L"Dropout layer:");
      for (tmp_type_dropout_layer_index = 0u;
           tmp_type_dropout_layer_index != LAYER_DROPOUT::LENGTH;
           ++tmp_type_dropout_layer_index) {
        INFO(L"[%d]: %ls.",
               tmp_type_dropout_layer_index,
               LAYER_DROPOUT_NAME[static_cast<LAYER_DROPOUT::TYPE>(
                                          tmp_type_dropout_layer_index)]
                   .c_str());
      }
      INFO(L"default=%ls.",
             LAYER_DROPOUT_NAME[LAYER_DROPOUT::BERNOULLI].c_str());

      switch ((tmp_type_layer_dropout = static_cast<LAYER_DROPOUT::TYPE>(
                   parse_discrete<size_t>(
                       0_UZ, LAYER_DROPOUT::LENGTH - 1,
                       (tmp_layer_name +
                           L" layer, type: ").c_str())))) {
        case LAYER_DROPOUT::NONE:
          tmp_hyper_parameters[0] = 0_r;
          tmp_hyper_parameters[1] = 0_r;
          break;
        case LAYER_DROPOUT::ALPHA:
          INFO(L"");
          INFO(L"Alpha dropout: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          if (layer_it->type_dropout == LAYER_DROPOUT::ALPHA) {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, dropout probability (" +
                    std::to_wstring(
                        layer_it->dropout_values[0]) +
                    L"): ").c_str());
          } else {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, dropout probability: ").c_str());
          }

          tmp_hyper_parameters[1] = 0_r;
          break;
        case LAYER_DROPOUT::BERNOULLI:
          INFO(L"");
          INFO(L"Dropout bernoulli: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          if (layer_it->type_dropout == LAYER_DROPOUT::BERNOULLI) {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, retention probability (" +
                    std::to_wstring(
                        layer_it->dropout_values[0]) +
                    L"): ").c_str());
          } else {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, retention probability: ").c_str());
          }

          tmp_hyper_parameters[1] = 0_r;
          break;
        case LAYER_DROPOUT::BERNOULLI_INVERTED:
          INFO(L"");
          INFO(L"Dropout bernoulli inverted: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          if (layer_it->type_dropout ==
              LAYER_DROPOUT::BERNOULLI_INVERTED) {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, retention probability (" +
                    std::to_wstring(
                        layer_it->dropout_values[0]) +
                    L"): ").c_str());
          } else {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, retention probability: ").c_str());
          }

          tmp_hyper_parameters[1] = 0_r;
          break;
        case LAYER_DROPOUT::GAUSSIAN:
          INFO(L"");
          INFO(L"Dropout gaussian: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          if (layer_it->type_dropout == LAYER_DROPOUT::GAUSSIAN) {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, dropout probability (" +
                    std::to_wstring(
                        layer_it->dropout_values[0]) +
                    L"): ").c_str());
          } else {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, dropout probability: ").c_str());
          }

          tmp_hyper_parameters[1] = 0_r;
          break;
        case LAYER_DROPOUT::SHAKEDROP:
          INFO(L"");
          INFO(L"Dropout ShakeDrop: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          if (layer_it->type_dropout == LAYER_DROPOUT::SHAKEDROP) {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, dropout probability (" +
                    std::to_wstring(
                        layer_it->dropout_values[0]) +
                    L"): ").c_str());
          } else {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, dropout probability: ").c_str());
          }

          tmp_hyper_parameters[1] = 0_r;
          break;
        case LAYER_DROPOUT::UOUT:
          INFO(L"");
          INFO(L"Dropout Uout: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          if (layer_it->type_dropout == LAYER_DROPOUT::UOUT) {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, dropout probability (" +
                    std::to_wstring(
                        layer_it->dropout_values[0]) +
                    L"): ").c_str());
          } else {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, dropout probability: ").c_str());
          }

          tmp_hyper_parameters[1] = 0_r;
          break;
        case LAYER_DROPOUT::ZONEOUT:
          INFO(L"");
          INFO(L"Zoneout cell: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.5.");

          if (layer_it->type_dropout == LAYER_DROPOUT::ZONEOUT) {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, zoneout cell probability (" +
                    std::to_wstring(
                        layer_it->dropout_values[0]) +
                    L"): ").c_str());
          } else {
            tmp_hyper_parameters[0] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, zoneout cell probability: ").c_str());
          }

          INFO(L"");
          INFO(L"Zoneout hidden: ");
          INFO(L"Range[0.0, 1.0]: ");
          INFO(L"default=0.05.");

          if (layer_it->type_dropout == LAYER_DROPOUT::ZONEOUT) {
            tmp_hyper_parameters[1] = parse_real(
                0_r, 1_r,
                (tmp_layer_name +
                    L" layer, zoneout hidden probability (" +
                    std::to_wstring(
                        layer_it->dropout_values[1]) +
                    L"): ").c_str());
          } else {
            tmp_hyper_parameters[1] = parse_real(
                0_r, 1_r,
                (tmp_layer_name + L" layer, zoneout hidden probability: ")
                    .c_str());
          }
          break;
        default:
          ERR(
              L"Type dropout layer (%u | %ls) is not managed in the switch.",
              tmp_type_layer_dropout,
              LAYER_DROPOUT_NAME[tmp_type_layer_dropout].c_str());
          return false;
      }

      if (tmp_option != -1ll) {
        if (this->set_dropout(layer_it, tmp_type_layer_dropout,
                               tmp_hyper_parameters) == false) {
          ERR(
              L"An error has been triggered from the "
              "\"set_dropout(ptr, %u, %f, %f)\" function.",
              tmp_type_layer_dropout, tmp_hyper_parameters[0],
              tmp_hyper_parameters[1]);

          return false;
        }

        if (this->type == MODEL::AUTOENCODER &&
            tmp_type_layer_dropout != LAYER_DROPOUT::NONE &&
            tmp_option != 0ll) {
          layer_it->use_coded_dropout = accept(
              L"Pre-training: Use dropout inside the coded layer?");
        }
      } else {
        switch ((tmp_type_layer_dropout = static_cast<LAYER_DROPOUT::TYPE>(
                     parse_discrete<size_t>(
                         0_UZ, LAYER_DROPOUT::LENGTH - 1,
                         (tmp_layer_name + L" layer, type: ").c_str())))) {
          case LAYER_DROPOUT::NONE:
            for (layer_it = this->ptr_array_layers;
                 layer_it != this->ptr_last_layer - 1;
                 ++layer_it) {
              if (this->set_dropout(layer_it, tmp_type_layer_dropout,
                                     tmp_hyper_parameters) == false) {
                ERR(
                    L"An error has been triggered from the "
                    "\"set_dropout(ptr, %u)\" function.",
                    tmp_type_layer_dropout);

                return false;
              }
            }
            break;
          case LAYER_DROPOUT::ALPHA:
            for (layer_it = this->ptr_array_layers;
                 layer_it != this->ptr_last_layer - 1;
                 ++layer_it) {
              if ((layer_it->type_layer == LAYER::FULLY_CONNECTED &&
                   *layer_it->ptr_array_AF_units
                           ->ptr_type_activation_function ==
                       ACTIVATION::SELU) ||
                  (layer_it->type_layer ==
                       LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT &&
                   *layer_it->ptr_array_AF_Ind_recurrent_units
                           ->ptr_type_activation_function ==
                       ACTIVATION::SELU)) {
                if (this->set_dropout(layer_it, tmp_type_layer_dropout,
                                       tmp_hyper_parameters) == false) {
                  ERR(
                      L"An error has been triggered from the "
                      "\"set_dropout(ptr, %u, %f, %f)\" function.",
                      tmp_type_layer_dropout, tmp_hyper_parameters[0],
                      tmp_hyper_parameters[1]);

                  return false;
                }
              }
            }
            break;
          case LAYER_DROPOUT::BERNOULLI:
          case LAYER_DROPOUT::BERNOULLI_INVERTED:
          case LAYER_DROPOUT::GAUSSIAN:
          case LAYER_DROPOUT::UOUT:
            for (layer_it = this->ptr_array_layers;
                 layer_it != this->ptr_last_layer - 1;
                 ++layer_it) {
              if (layer_it->type_layer == LAYER::FULLY_CONNECTED ||
                  layer_it->type_layer ==
                      LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT) {
                if (this->set_dropout(layer_it, tmp_type_layer_dropout,
                                       tmp_hyper_parameters) == false) {
                  ERR(
                      L"An error has been triggered from the "
                      "\"set_dropout(ptr, %u, %f, %f)\" function.",
                      tmp_type_layer_dropout, tmp_hyper_parameters[0],
                      tmp_hyper_parameters[1]);

                  return false;
                }
              }
            }
            break;
          case LAYER_DROPOUT::SHAKEDROP:
            tmp_parameters[0] = 0.0f;
            tmp_parameters[1] = 0.0f;

            for (layer_it = this->ptr_array_layers;
                 layer_it != this->ptr_last_layer - 1;
                 ++layer_it) {
              if (layer_it->type_layer == LAYER::RESIDUAL) {
                ++tmp_parameters[0];
              }
            }

            for (layer_it = this->ptr_array_layers;
                 layer_it != this->ptr_last_layer - 1;
                 ++layer_it) {
              if (layer_it->type_layer == LAYER::RESIDUAL) {
                if (this->set_dropout(
                        layer_it, tmp_type_layer_dropout,
                        std::array<real, 1_UZ>{
                            1_r -
                            (((++tmp_parameters[1]) / tmp_parameters[0]) *
                             (1_r - tmp_hyper_parameters[0]))}
                            .data()) == false) {
                  ERR(
                      L"An error has been triggered from the "
                      "\"set_dropout(ptr, %u, %f)\" function.",
                      tmp_type_layer_dropout,
                      1_r - ((tmp_parameters[1] / tmp_parameters[0]) *
                                    (1_r - tmp_hyper_parameters[0])));

                  return false;
                }
              }
            }
            break;
          case LAYER_DROPOUT::ZONEOUT:
            for (layer_it = this->ptr_array_layers;
                 layer_it != this->ptr_last_layer - 1;
                 ++layer_it) {
              if (layer_it->type_layer == LAYER::LSTM) {
                if (this->set_dropout(layer_it, tmp_type_layer_dropout,
                                       tmp_hyper_parameters) == false) {
                  ERR(
                      L"An error has been triggered from the "
                      "\"set_dropout(ptr, %u, %f, %f)\" function.",
                      tmp_type_layer_dropout, tmp_hyper_parameters[0],
                      tmp_hyper_parameters[1]);

                  return false;
                }
              }
            }
            break;
          default:
            ERR(
                L"Type dropout layer (%u | %ls) is not managed in "
                "the switch.",
                tmp_type_layer_dropout,
                LAYER_DROPOUT_NAME[tmp_type_layer_dropout].c_str());
            return false;
        }
      }
    } else if (tmp_option == static_cast<long long int>(tmp_option_end)) {
      return true;
    } else {
      ERR(
          L"An error has been triggered from the "
          "\"parse_discrete<long long int>(%lld, %zu)\" function.", -1ll, tmp_option_end);
    }
  }

  return false;
}

bool Model::User_Controls__Normalization(void) {
  while (true) {
    INFO(L"");
    INFO(L"User controls, normalization:");
    INFO(L"[0]: Modify momentum average (%.9f).",
           this->normalization_momentum_average);
    INFO(L"[1]: Modify epsilon (%.9f).",
           this->normalization_epsilon);
    INFO(L"[2]: Modify r correction maximum (%.9f).",
           this->batch_renormalization_r_correction_maximum);
    INFO(L"[3]: Modify d correction maximum (%.9f).",
           this->batch_renormalization_d_correction_maximum);
    INFO(L"[4]: Modify normalization (%ls).",
           this->Use__Normalization() ? "Yes" : "No");
    INFO(L"[5]: Quit.");

    switch (parse_discrete<int>(
        0, 5, L"Option: ")) {
      case 0:
        INFO(L"");
        INFO(L"Momentum average:");
        INFO(L"default=0.999.");
        if (this->Set__Normalization_Momentum_Average(
                parse_real(
                    0_r, 1_r,
                    L"Momentum average: ")) ==
            false) {
          ERR(
              L"An error has been triggered from the "
              "\"Set__Normalization_Momentum_Average()\" function.",);

          return false;
        }
        break;
      case 1:
        INFO(L"");
        INFO(L"Epsilon:");
        INFO(L"default=1e-5.");
        if (this->Set__Normalization_Epsilon(parse_real(
                0_r, L"Epsilon: ")) == false) {
          ERR(
              L"An error has been triggered from the "
              "\"Set__Normalization_Epsilon()\" function.",);

          return false;
        }
        break;
      case 2:
        INFO(L"r correction maximum:");
        INFO(L"default=1.");
        if (this->Set__Batch_Renormalization_r_Correction_Maximum(
                parse_real(
                    0_r, L"r correction maximum: ")) == false) {
          ERR(
              L"An error has been triggered from the "
              "\"Set__Batch_Renormalization_r_Correction_Maximum()\" function.",);

          return false;
        }
        break;
      case 3:
        INFO(L"");
        INFO(L"d correction maximum:");
        INFO(L"default=0.");
        if (this->Set__Batch_Renormalization_d_Correction_Maximum(
                parse_real(
                    0_r, L"d correction maximum: ")) == false) {
          ERR(
              L"An error has been triggered from the "
              "\"Set__Batch_Renormalization_d_Correction_Maximum()\" function.",);

          return false;
        }
        break;
      case 4:
        if (this->User_Controls__Normalization_Layer() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Normalization_Layer()\" function.",);

          return false;
        }
        break;
      case 5:
        return true;
      default:
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%u, %u)\" function.", 0, 5u);
        break;
    }
  }

  return false;
}

bool Model::User_Controls__Normalization_Layer(void) {
#ifdef COMPILE_CUDA
  bool tmp_parameters_has_change(false);

  LAYER_NORM::TYPE tmp_type_normalization_parameter;
#endif

  unsigned int tmp_type_normalization_layer_index;

  size_t const tmp_option_end(this->type == MODEL::AUTOENCODER
                                  ? this->total_layers / 2_UZ + 1_UZ
                                  : this->total_layers - 1_UZ);
  size_t tmp_option, tmp_layer_index;

  struct Layer *layer_it;

  while (true) {
    INFO(L"");
    INFO(L"User controls, Layer normalization:");
    for (tmp_layer_index = 1_UZ; tmp_layer_index != tmp_option_end;
         ++tmp_layer_index) {
      INFO(L"[%zu]: Hidden layer[%zu]: %ls, %ls.", tmp_layer_index - 1_UZ,
             tmp_layer_index - 1_UZ,
             LAYER_NORM_NAME[this->ptr_array_layers[tmp_layer_index]
                                     .type_normalization]
                 .c_str(),
             this->ptr_array_layers[tmp_layer_index]
                     .use_layer_normalization_before_activation
                 ? "true"
                 : "false");
    }
    INFO(L"[%zu]: Quit.",
           tmp_option_end - 1_UZ);

    tmp_option = parse_discrete<size_t>(
                     0_UZ, tmp_option_end - 1_UZ, L"Option: ") +
                 1_UZ;

    if (tmp_option < tmp_option_end) {
      layer_it = this->ptr_array_layers + tmp_option;

#ifdef COMPILE_CUDA
      tmp_type_normalization_parameter = layer_it->type_normalization;
#endif

      INFO(L"");
      INFO(L"Layer normalization:");
      for (tmp_type_normalization_layer_index = 0u;
           tmp_type_normalization_layer_index != LAYER_NORM::LENGTH;
           ++tmp_type_normalization_layer_index) {
        INFO(L"[%d]: %ls.",
               tmp_type_normalization_layer_index,
               LAYER_NORM_NAME[static_cast<LAYER_NORM::TYPE>(
                                       tmp_type_normalization_layer_index)]
                   .c_str());
      }
      INFO(
          L"default=%ls.",
          LAYER_NORM_NAME[LAYER_NORM::BATCH_RENORMALIZATION].c_str());

      if (this->Set__Layer_Normalization(
              layer_it,
              static_cast<LAYER_NORM::TYPE>(parse_discrete<size_t>(
                  0_UZ, LAYER_NORM::LENGTH - 1,
                  (L"Hidden layer " +
                      std::to_wstring(static_cast<size_t>(
                          layer_it - this->ptr_array_layers)) +
                      L", type: ").c_str()))) == false) {
        ERR(
            L"An error has been triggered from the "
            "\"Set__Layer_Normalization()\" function.",);

        return false;
      }

      if (layer_it->type_layer == LAYER::FULLY_CONNECTED ||
          layer_it->type_layer ==
              LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT ||
          layer_it->type_layer ==
              LAYER::FULLY_CONNECTED_RECURRENT) {
        layer_it->use_layer_normalization_before_activation =
            accept(L"Use normalization before activation?");
      }

#ifdef COMPILE_CUDA
      if (tmp_type_normalization_parameter !=
          layer_it->type_normalization) {
        tmp_parameters_has_change = true;
      }
#endif
    } else if (tmp_option == tmp_option_end) {
#ifdef COMPILE_CUDA
      if (this->is_cu_initialized && tmp_parameters_has_change) {
        this->cumodel->Copy__Normalization(this);
      }
#endif

      return true;
    } else {
      ERR(
          L"An error has been triggered from the "
          "\"parse_discrete<size_t>(%zu, %zu)\" function.", 0_UZ,
          tmp_option_end - 1_UZ);
    }
  }

  return false;
}

bool Model::User_Controls__Tied__Parameter(void) {
  size_t const tmp_option_end(this->total_layers / 2_UZ + 1_UZ);
  size_t tmp_option;

  struct Layer *layer_it;

  while (true) {
    INFO(L"");
    INFO(L"User controls, Tied parameter:");
    for (size_t tmp_layer_index(1_UZ); tmp_layer_index != tmp_option_end;
         ++tmp_layer_index) {
      INFO(L"[%zu]: Hidden layer[%zu]: %ls.", tmp_layer_index - 1_UZ,
             tmp_layer_index - 1_UZ,
             this->ptr_array_layers[tmp_layer_index].Use__Tied_Parameter()
                 ? "true"
                 : "false");
    }
    INFO(L"[%zu]: Quit.",
           tmp_option_end - 1_UZ);

    tmp_option = parse_discrete<size_t>(
                     0_UZ, tmp_option_end - 1_UZ,
                     L"Option: ") +
                 1_UZ;

    if (tmp_option < tmp_option_end) {
      layer_it = this->ptr_array_layers + tmp_option;

      INFO(L"");
      INFO(L"Tied parameter:");
      INFO(L"default=%ls.",
             this->type == MODEL::AUTOENCODER ? L"true" : L"false");
      if (this->Set__Tied_Parameter(
              layer_it,
              layer_it->Use__Tied_Parameter() == false) == false) {
        ERR(
            L"An error has been triggered from the "
            "\"Set__Alpha_Sparsity(ptr)\" function.",);

        continue;
      }
    } else if (tmp_option == tmp_option_end) {
      return true;
    } else {
      ERR(
          L"An error has been triggered from the "
          "\"parse_discrete<size_t>(%zu, %zu)\" function.", 0_UZ,
          tmp_option_end - 1_UZ);
    }
  }

  return false;
}

bool Model::User_Controls__K_Sparse(void) {
  size_t const tmp_option_end(this->type == MODEL::AUTOENCODER
                                  ? this->total_layers / 2_UZ + 1_UZ
                                  : this->total_layers - 1_UZ);
  size_t tmp_option, tmp_layer_size;

  struct Layer *layer_it;

  while (true) {
    INFO(L"");
    INFO(L"User controls, k-Sparse:");
    for (size_t tmp_layer_index(1_UZ); tmp_layer_index != tmp_option_end;
         ++tmp_layer_index) {
      INFO(L"[%zu]: Hidden layer[%zu]: %zu, %f.", tmp_layer_index - 1_UZ,
             tmp_layer_index - 1_UZ,
             this->ptr_array_layers[tmp_layer_index].k_sparsity,
             this->ptr_array_layers[tmp_layer_index].alpha_sparsity);
    }
    INFO(L"[%zu]: Quit.",
           tmp_option_end - 1_UZ);

    tmp_option = parse_discrete<size_t>(
                     0_UZ, tmp_option_end - 1_UZ,
                     L"Option: ") +
                 1_UZ;

    if (tmp_option < tmp_option_end) {
      layer_it = this->ptr_array_layers + tmp_option;

      tmp_layer_size = *layer_it->ptr_number_outputs;

      INFO(L"");
      INFO(L"k-Sparse:");
      INFO(L"Range[0, %zu].",
             tmp_layer_size);
      INFO(L"default=%zu.",
             tmp_layer_size / 4_UZ == 0_UZ ? 1_UZ : tmp_layer_size / 4_UZ);
      if (this->Set__K_Sparsity(
              layer_it,
              parse_discrete<size_t>(
                  0_UZ, tmp_layer_size,
                  (L"k-sparse (" +
                      std::to_wstring(layer_it->k_sparsity) + L"): ").c_str())) ==
          false) {
        ERR(
            L"An error has been triggered from the "
            "\"Set__K_Sparsity(ptr)\" function.",);

        continue;
      }

      INFO(L"");
      INFO(L"Alpha k-sparse:");
      INFO(L"Range[1, inf].");
      INFO(L"default=2.");
      if (this->Set__Alpha_Sparsity(
              layer_it,
              parse_real(
                  0_r,
                  (L"Alpha k-sparse (" +
                      Str::to_wstring(
                          layer_it->alpha_sparsity) +
                      L"): ").c_str())) == false) {
        ERR(
            L"An error has been triggered from the "
            "\"Set__Alpha_Sparsity(ptr)\" function.",);

        continue;
      }
    } else if (tmp_option == tmp_option_end) {
      return true;
    } else {
      ERR(
          L"An error has been triggered from the "
          "\"parse_discrete<size_t>(%zu, %zu)\" function.", 0_UZ,
          tmp_option_end - 1_UZ);
    }
  }

  return false;
}

bool Model::user_controls(void) {
  while (true) {
    INFO(L"");
    INFO(L"User controls:");
    INFO(L"[0]: clear training arrays.");
    INFO(L"[1]: reset global loss.");
    INFO(L"[2]: Weights initializer.");
    INFO(L"[3]: Optimizer function (%ls).",
           OPTIMIZER_NAME[this->type_optimizer_function].c_str());
    INFO(L"[4]: Loss function (%ls).",
           LOSS_FN_NAME[this->type_loss_function].c_str());
    INFO(L"[5]: Accuracy function (%ls).",
           ACC_FN_NAME[this->type_accuracy_function].c_str());
    INFO(L"[6]: clip gradient (%ls).",
           this->Use__Clip_Gradient() ? "true" : "false");
    INFO(L"[7]: Regularization.");
    INFO(L"[8]: Modify accuracy variance (%f).", this->acc_var);
    INFO(L"[9]: Modify time delays.");
    INFO(L"[10]: OpenMP.");
    INFO(L"[11]: Batch size (%zu).", this->maximum_batch_size);
    INFO(L"[12]: Print information.");
    INFO(L"[13]: Quit.");

    switch (parse_discrete<int>(
        0, 13, L"Option: ")) {
      case 0:
        this->clear_training_arrays();
        break;
      case 1:
        this->reset_global_loss();
        break;
      case 2:
        if (this->User_Controls__Weights_Initializer() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Weights_Initializer()\" function.",);

          return false;
        }
        break;
      case 3:
        if (this->User_Controls__Optimizer_Function() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Optimizer_Function()\" function.",);

          return false;
        }
        break;
      case 4:
        if (this->User_Controls__Loss_Function_Initializer() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Loss_Function_Initializer()\" function.",);

          return false;
        }
        break;
      case 5:
        if (this->User_Controls__Accuracy_Function_Initializer() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Accuracy_Function_Initializer()\" function.",);

          return false;
        }
        break;
      case 6:
        if (this->User_Controls__Clip_Gradient() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Clip_Gradient()\" function.",);

          return false;
        }
        break;
      case 7:
        if (this->User_Controls__Regularization() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Regularization()\" function.",);

          return false;
        }
        break;
      case 8:
        if (this->User_Controls__Accuracy_Variance() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Accuracy_Variance()\" function.",);

          return false;
        }
        break;
      case 9:
        if (this->User_Controls__Time_Delays() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Time_Delays()\" function.",);

          return false;
        }
        break;
      case 10:
        if (this->User_Controls__OpenMP() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__OpenMP()\" function.",);

          return false;
        }
        break;
      case 11:
        if (this->User_Controls__Maximum__Batch_Size() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"User_Controls__Maximum__Batch_Size()\" function.",);

          return false;
        }
        break;
      case 12:
        INFO(L"%ls", this->Get__Parameters().c_str());
        break;
      case 13:
        return true;
      default:
        ERR(
            L"An error has been triggered from the "
            "\"parse_discrete<int>(%d, %d)\" function.", 0, 13);
        break;
    }
  }

  return false;
}
}
