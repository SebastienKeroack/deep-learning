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
#include "run/pch.hpp"

// Deep learning:
#include "deep-learning/device/system/info.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/io/term/input.hpp"
#include "deep-learning/v1/learner/models.hpp"
#include "deep-learning/v1/optimizers/grid_search.hpp"

// Standard:
#ifdef _WIN32
#include <windows.h>
#endif

using namespace DL;
using namespace DL::v1;
using namespace DL::Sys;
using namespace DL::Term;

bool grid_search_opt(void) {
  std::wstring model_name(parse_wstring(L"Model name:"));

#ifdef _WIN32
  if (SetConsoleTitleW((model_name + L" - Grid Search").c_str()) == FALSE)
    WARN(L"Couldn't set a title to the console.");
#endif

  Models models;

  models.auto_save_trainer(false);
  models.auto_save_trained(false);
  models.auto_save_competitor(false);

  if (models.set_desired_loss(0.0) == false) {
    ERR(L"An error has been triggered from the "
        L"`set_desired_loss()` function.");
    return false;
  } else if (models.initialize_dirs(model_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_dirs(%ls, %ls)` function.",
        model_name.c_str(), model_name.c_str());
    return false;
  } else if (models.initialize_datasets() == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_datasets(ptr)` function.");
    return false;
  }

  INFO(L"[Yes=VALID | No=TESTG]");
  models.get_datasets()->set_eval_env(
      accept(L"Evaluation on valid set: ") ? ENV::VALID : ENV::TESTG);
  INFO(L"");

  models.get_datasets()->set_desired_optimization_time_between_reports(
      parse_real(
          0.0, L"Desired optimization time between reports (seconds): "));
  INFO(L"");

  // Memory allocate.
  size_t const remaining_mem_(remaining_mem(10.0L, 1000_UZ * ::MEGABYTE));

  INFO(L"");
  INFO(L"Allowable memory:");
  INFO(L"Range[1, %zu] MBs.", remaining_mem_ / ::MEGABYTE);

  size_t const allowable_host_mem(
      parse_discrete(1_UZ, remaining_mem_ / ::MEGABYTE) * ::MEGABYTE);

  INFO(L"");
  // |END| Memory allocate. |END|

  if (accept(L"Do you want to load the model?")) {
    size_t allowable_devc_mem(0_UZ);

#ifdef COMPILE_CUDA
    INFO(L"");
    if (accept(L"Do you want to use CUDA?")) {
      int device_id(-1);

      INFO(L"");
      models.set_use_cuda(
          CUDA__Input__Use__CUDA(device_id, allowable_devc_mem));
    }
#endif

    if (models.load_model(HIERARCHY::TRAINER, allowable_host_mem,
                          allowable_devc_mem, false) == false) {
      ERR(L"An error has been triggered from the "
          L"`load_model(%ls, %zu, %zu, false)` function.",
          HIERARCHY_NAME[HIERARCHY::TRAINER].c_str(), allowable_host_mem,
          allowable_devc_mem);
      return false;
    } else if (models.load_model(HIERARCHY::TRAINED, allowable_host_mem,
                                 allowable_devc_mem, true) == false) {
      ERR(L"An error has been triggered from the "
          L"`load_model(%ls, %zu, %zu, true)` function.",
          HIERARCHY_NAME[HIERARCHY::TRAINED].c_str(), allowable_host_mem,
          allowable_devc_mem);
      return false;
    }
  } else {
    if (models.create(allowable_host_mem) == false) {
      ERR(L"An error has been triggered from the `create()` function.");
      return false;
    }
  }

  Grid_Search grid_search;

  if (grid_search.link(models) == false) {
    ERR(L"An error has been triggered from the `Input(ref)` function.");
    return false;
  }

  if (grid_search.update_tree() == false) {
    ERR(L"An error has been triggered from the `update_tree()` function.");
    return false;
  }

  if (grid_search.optimize(models) == false) {
    ERR(L"An error has been triggered from the `optimize(ptr)` function.");
    return false;
  }

  INFO(L"Best hyper-parameters:");
  INFO(L"%ls",
       models.get_model(HIERARCHY::COMPETITOR)->Get__Parameters().c_str());

  return true;
}