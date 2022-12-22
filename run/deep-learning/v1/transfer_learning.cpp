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
#include "deep-learning/io/term/spinner.hpp"
#include "deep-learning/v1/learner/models.hpp"

// Standard:
#ifdef _WIN32
#include <windows.h>
#endif

using namespace DL;
using namespace DL::v1;
using namespace DL::Sys;
using namespace DL::Term;

bool transfer_learning(void) {
  std::wstring model_src_name(parse_wstring(L"Model src name: "));
  std::wstring model_dst_name(parse_wstring(L"Model dst name: "));
  INFO(L"");

#ifdef _WIN32
  if (SetConsoleTitleW((model_src_name + L" Transfer learning").c_str()) ==
      FALSE)
    WARN(L"Couldn't set a title to the console.");
#endif

  Models models_src, models_dst;

  if (models_src.initialize_dirs(model_src_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_dirs(%ls, %ls)` function.",
        model_src_name.c_str(), model_src_name.c_str());
    return false;
  } else if (models_dst.initialize_dirs(model_dst_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_dirs(%ls, %ls)` function.",
        model_dst_name.c_str(), model_dst_name.c_str());
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

  Model *model_src, *model_dst;

  // load source neural network.
  if (models_src.load_model(HIERARCHY::TRAINED, allowable_host_mem, 0_UZ,
                            false) == false) {
    ERR(L"An error has been triggered from the "
        L"`load_model(%ls, %zu, %zu, false)` function.",
        HIERARCHY_NAME[HIERARCHY::TRAINED].c_str(), allowable_host_mem, 0_UZ);
    return false;
  }

  model_src = models_src.get_model(HIERARCHY::TRAINED);
  // |END| load source neural network. |END|

  // Create/load destination neural network.
  INFO(L"");
  if (accept(L"Do you want to load the model (destination) from a file?")) {
    if (models_dst.load_model(HIERARCHY::TRAINER, allowable_host_mem, 0_UZ,
                              false) == false) {
      ERR(L"An error has been triggered from the "
          L"`load_model(%ls, %zu, %zu, false)` function.",
          HIERARCHY_NAME[HIERARCHY::TRAINER].c_str(), allowable_host_mem, 0_UZ);
      return false;
    }
  } else if (models_dst.create(allowable_host_mem) == false) {
    ERR(L"An error has been triggered from the "
        L"`create(%zu)` function.",
        allowable_host_mem);
    return false;
  }

  model_dst = models_dst.get_model(HIERARCHY::TRAINER);
  // |END| Create/load destination neural network. |END|

  INFO(L"");
  Spinner spinner;
  spinner.start(L"Transfer learning... ");

  if (model_src->transfer_learning(model_dst) == false) {
    ERR(L"An error has been triggered from the "
        L"`transfer_learning(ptr)` function.");

    return false;
  }

  spinner.join();
  INFO(L"");

  INFO(L"");
  spinner.start((L"Saving into " +
                 models_dst.get_model_path(HIERARCHY::TRAINER, L"") +
                 L"(nn|net)... ")
                    .c_str());

  if (models_dst.save_model(HIERARCHY::TRAINER) == false) {
    ERR(L"An error has been triggered from the "
        L"`save_model(%ls)` function.",
        HIERARCHY_NAME[HIERARCHY::ALL].c_str());
    return false;
  }

  spinner.join();
  INFO(L"");

  return true;
}
