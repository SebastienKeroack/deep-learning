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

// Deep learning:
#include "deep-learning/device/system/info.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/io/term/input.hpp"
#include "deep-learning/v1/learner/models.hpp"

// Standard:
#ifdef _WIN32
#include <windows.h>
#endif

using namespace DL;
using namespace DL::v1;
using namespace DL::Sys;
using namespace DL::Term;

bool optimize(void) {
  std::wstring model_name(parse_wstring(L"Model name:"));
  INFO(L"");

#ifdef _WIN32
  if (SetConsoleTitleW(model_name.c_str()) == FALSE)
    WARN(L"Couldn't set a title to the console.");
#endif

  Models models;

  models.auto_save_trainer(true);
  models.auto_save_trained(true);
  models.auto_save_competitor(true);

  if (models.set_desired_loss(parse_real(0.0, 1.0, L"Desired loss: ")) ==
      false) {
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
        L"`initialize_datasets()` function.");
    return false;
  }

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
#endif  // COMPILE_CUDA

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
  } else if (models.create(allowable_host_mem) == false) {
    ERR(L"An error has been triggered from the `create(%zu)` function.",
        allowable_host_mem);
    return false;
  }

  WhileCond while_cond;
  while_cond.type = WHILE_MODE::FOREVER;

  if (models.set_while_cond(while_cond) == false) {
    ERR(L"An error has been triggered from the "
        L"`set_while_cond()` function.");
    return false;
  }

  if (models.evaluate_envs() == false) {
    ERR(L"An error has been triggered from the "
        L"`evaluate_envs()` function.");
    return false;
  }

  models.optimize();

  models.compare_trained();

  if (models.save_model(HIERARCHY::ALL) == false) {
    ERR(L"An error has been triggered from the "
        L"`save_model(%ls)` function.",
        HIERARCHY_NAME[HIERARCHY::ALL].c_str());
    return false;
  }

  return true;
}
