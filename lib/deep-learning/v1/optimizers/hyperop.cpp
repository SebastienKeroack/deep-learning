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
#include "deep-learning/io/term/input.hpp"
#include "deep-learning/io/logger.hpp"

#ifdef COMPILE_CUDA
#include "deep-learning/v1/data/datasets.cuh"
#endif

#include <iostream>

using namespace DL::Term;

namespace DL::v1 {
HyperOpt::HyperOpt(void) {}

void HyperOpt::reset(void) {
  this->p_optimization_iterations_since_hyper_optimization = 0_UZ;
}

bool HyperOpt::Optimize(Datasets *const datasets, Model *const model) {
  switch (this->_type_hyperparameter_optimization) {
    case HYPEROPT::NONE:
      return true;
    case HYPEROPT::GAUSSIAN_SEARCH:
      if (this->gaussian_opt->Optimize(
              this->p_number_hyper_optimization_iterations, datasets, model) ==
          false) {
        ERR(L"An error has been triggered from the "
            L"`Optimize(%zu, ptr, ptr)` function.",
            this->p_number_hyper_optimization_iterations);

        return false;
      }
      break;
    default:
      ERR(L"Hyperparameter optimization type (%d | %ls) is not "
          L"managed in the switch.",
          this->_type_hyperparameter_optimization,
          HYPEROPT_NAME[this->_type_hyperparameter_optimization].c_str());
      return false;
  }

  this->_evaluation_require = true;

  return true;
}

bool HyperOpt::Evaluation(void) {
  switch (this->_type_hyperparameter_optimization) {
    case HYPEROPT::NONE:
      return true;
    case HYPEROPT::GAUSSIAN_SEARCH:
      if (this->gaussian_opt->Evaluation() == false) {
        ERR(L"An error has been triggered from the "
            L"`Evaluation()` function.");

        return false;
      }
      break;
    default:
      ERR(L"Hyperparameter optimization type (%d | %ls) is not "
          L"managed in the switch.",
          this->_type_hyperparameter_optimization,
          HYPEROPT_NAME[this->_type_hyperparameter_optimization].c_str());
      return false;
  }

  this->_evaluation_require = false;

  return true;
}

bool HyperOpt::Evaluation(Datasets *const datasets) {
  switch (this->_type_hyperparameter_optimization) {
    case HYPEROPT::NONE:
      return true;
    case HYPEROPT::GAUSSIAN_SEARCH:
      if (this->gaussian_opt->Evaluation(datasets) == false) {
        ERR(L"An error has been triggered from the "
            L"`Evaluation(ptr)` function.");

        return false;
      }
      break;
    default:
      ERR(L"Hyperparameter optimization type (%d | %ls) is not "
          L"managed in the switch.",
          this->_type_hyperparameter_optimization,
          HYPEROPT_NAME[this->_type_hyperparameter_optimization].c_str());
      return false;
  }

  this->_evaluation_require = false;

  return true;
}

bool HyperOpt::Set__Hyperparameter_Optimization(
    HYPEROPT::TYPE const type_hyper_optimization_received) {
  // Deallocate.
  switch (this->_type_hyperparameter_optimization) {
    case HYPEROPT::NONE:
      break;
    case HYPEROPT::GAUSSIAN_SEARCH:
      this->Deallocate__Gaussian_Search();
      break;
    default:
      ERR(L"Hyperparameter optimization type (%d | %ls) is not "
          L"managed in the switch.",
          this->_type_hyperparameter_optimization,
          HYPEROPT_NAME[this->_type_hyperparameter_optimization].c_str());
      return false;
  }

  // allocate.
  switch (type_hyper_optimization_received) {
    case HYPEROPT::NONE:
      break;
    case HYPEROPT::GAUSSIAN_SEARCH:
      if (this->allocate_gaussian_opt() == false) {
        ERR(L"An error has been triggered from the "
            L"`allocate_gaussian_opt()` function.");

        return false;
      }
      break;
    default:
      ERR(L"Hyperparameter optimization type (%d | %ls) is not "
          L"managed in the switch.",
          type_hyper_optimization_received,
          HYPEROPT_NAME[type_hyper_optimization_received].c_str());
      return false;
  }

  this->_type_hyperparameter_optimization = type_hyper_optimization_received;

  return true;
}

bool HyperOpt::Set__Number_Hyperparameter_Optimization_Iterations(
    size_t const number_hyper_optimization_iterations_received) {
  if (number_hyper_optimization_iterations_received == 0_UZ) {
    ERR(L"The number of hyperparameter optimization iterations "
        L"can not be zero.");

    return false;
  }

  this->p_number_hyper_optimization_iterations =
      number_hyper_optimization_iterations_received;

  return true;
}

bool HyperOpt::Set__Number_Hyperparameter_Optimization_Iterations_Delay(
    size_t const number_hyper_optimization_iterations_delay_received) {
  if (number_hyper_optimization_iterations_delay_received == 0_UZ) {
    ERR(L"The number of hyperparameter optimization iterations "
        L"delay can not be zero.");

    return false;
  }

  this->p_number_hyper_optimization_iterations_delay =
      number_hyper_optimization_iterations_delay_received;

  return true;
}

bool HyperOpt::Get__Evaluation_Require(void) const {
  return (this->_evaluation_require);
}

bool HyperOpt::User_Controls__Change__Hyperparameter_Optimization(void) {
  INFO(L"");
  INFO(L"User controls, hyperparameter optimization type.");
  for (int i(0); i != HYPEROPT::LENGTH; ++i)
    INFO(L"[%d]: %ls.", i,
         HYPEROPT_NAME[static_cast<HYPEROPT::TYPE>(i)].c_str());
  INFO(L"default=%ls.", HYPEROPT_NAME[HYPEROPT::GAUSSIAN_SEARCH].c_str());

  if (this->Set__Hyperparameter_Optimization(static_cast<HYPEROPT::TYPE>(
          parse_discrete(0, HYPEROPT::LENGTH - 1, L"Type: "))) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Hyperparameter_Optimization()` function.");

    return false;
  }

  return true;
}

bool HyperOpt::user_controls(void) {
  while (true) {
    INFO(L"");
    INFO(L"User controls:");
    INFO(
        L"[0]: Number hyperparameter optimization iteration(s) "
        L"(%zu).",
        this->p_number_hyper_optimization_iterations);
    INFO(
        L"[1]: Number hyperparameter optimization iteration(s) delay "
        L"(%zu).",
        this->p_number_hyper_optimization_iterations_delay);
    INFO(L"[2]: Change hyperparameter optimization.");
    INFO(L"[3]: Hyperparameter optimization (%ls).",
         HYPEROPT_NAME[this->_type_hyperparameter_optimization].c_str());
    INFO(L"[4]: Quit.");

    switch (parse_discrete(0, 4, L"Option: ")) {
      case 0:
        INFO(L"");
        INFO(L"Number hyperparameter optimization iteration(s):");
        INFO(L"Range[1, 8].");
        INFO(L"default=10.");
        if (this->Set__Number_Hyperparameter_Optimization_Iterations(
                parse_discrete(0_UZ, L"Iteration(s): ")) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Number_Hyperparameter_Optimization_Iterations()` "
              L"function.");

          return false;
        }
        break;
      case 1:
        INFO(L"");
        INFO(
            L"Number hyperparameter optimization iteration(s) "
            L"delay:");
        INFO(L"Range[1, 8].");
        INFO(L"default=25.");
        if (this->Set__Number_Hyperparameter_Optimization_Iterations_Delay(
                parse_discrete(0_UZ, L"Iteration(s) delay: ")) ==
            false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Number_Hyperparameter_Optimization_Iterations_Delay()` "
              L"function.");

          return false;
        }
        break;
      case 2:
        if (this->User_Controls__Change__Hyperparameter_Optimization() ==
            false) {
          ERR(L"An error has been triggered from the "
              L"`User_Controls__Change__Hyperparameter_Optimization()` "
              L"function.");

          return false;
        }
        break;
      case 3:
        switch (this->_type_hyperparameter_optimization) {
          case HYPEROPT::NONE:
            break;
          case HYPEROPT::GAUSSIAN_SEARCH:
            if (this->gaussian_opt->user_controls() == false) {
              ERR(L"An error has been triggered from the "
                  L"`user_controls()` function.");

              return false;
            }
            break;
          default:
            ERR(L"Hyperparameter optimization type (%d | %ls) is not managed "
                L"in the switch.",
                this->_type_hyperparameter_optimization,
                HYPEROPT_NAME[this->_type_hyperparameter_optimization].c_str());
            return false;
        }
        break;
      case 4:
        return true;
      default:
        ERR(L"An error has been triggered from the `parse_discrete<unsigned "
            L"int>(%d, %d)` function.",
            0, 4);
        return false;
    }
  }

  return true;
}

bool HyperOpt::allocate_gaussian_opt(void) {
  if (this->gaussian_opt == nullptr) this->gaussian_opt = new Gaussian_Search;
  return true;
}

double HyperOpt::optimize(Datasets *const datasets, Model *const model) {
  if (++this->p_optimization_iterations_since_hyper_optimization >=
      this->p_number_hyper_optimization_iterations_delay) {
    this->p_optimization_iterations_since_hyper_optimization = 0_UZ;

    if (this->Optimize(datasets, model) == false) {
      ERR(L"An error has been triggered from the `Optimize(ptr, ptr)` "
          L"function.");

      return HUGE_VAL;
    }

    return model->get_loss(ENV::TRAIN);
  }

#ifdef COMPILE_CUDA
  if (model->Use__CUDA()) return datasets->Get__CUDA()->train(model);
#endif

  return datasets->train(model);
}

HYPEROPT::TYPE HyperOpt::Get__Hyperparameter_Optimization(void) const {
  return this->_type_hyperparameter_optimization;
}

void HyperOpt::Deallocate__Gaussian_Search(void) {
  SAFE_DELETE(this->gaussian_opt);
}

bool HyperOpt::Deallocate(void) {
  this->Deallocate__Gaussian_Search();

  return true;
}

HyperOpt::~HyperOpt(void) { this->Deallocate(); }
}  // namespace DL