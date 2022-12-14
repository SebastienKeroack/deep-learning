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

bool fx_simulator(void) {
  std::wstring model_name(parse_wstring(L"Model name:"));
  INFO(L"");

#ifdef _WIN32
  if (SetConsoleTitleW((model_name + L" - FX simulator").c_str()) == FALSE)
    WARN(L"Couldn't set a title to the console.");
#endif

  Models models;

  if (models.initialize_dirs(model_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_dirs(%ls, %ls)` function.",
        model_name.c_str(), model_name.c_str());
    return false;
  }

  // DatasetV1 Manager Parameters.
  DatasetsParams datasets_params;

  datasets_params.type_train = 0;

  if (models.initialize_datasets(&datasets_params) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_datasets(ptr)` function.");
    return false;
  }
  // |END| DatasetV1 Manager Parameters. |END|

  // Memory allocate.
  size_t const remaining_mem_(remaining_mem(10.0L, 1000_UZ * ::MEGABYTE));

  INFO(L"");
  INFO(L"Allowable memory:");
  INFO(L"Range[1, %zu] MBs.", remaining_mem_ / ::MEGABYTE);

  size_t const allowable_host_mem(
      parse_discrete(1_UZ, remaining_mem_ / ::MEGABYTE) * ::MEGABYTE);

  INFO(L"");
  // |END| Memory allocate. |END|

  Model *model(nullptr);

  size_t allowable_devc_mem(0_UZ);

#ifdef COMPILE_CUDA
  INFO(L"");
  if (accept(L"Do you want to use CUDA?")) {
    int device_id(-1);

    INFO(L"");
    models.set_use_cuda(CUDA__Input__Use__CUDA(device_id, allowable_devc_mem));
  }
#endif

  if (accept(L"Do you want to load the model (Tainer)?")) {
    if (models.load_model(HIERARCHY::TRAINER, allowable_host_mem,
                          allowable_devc_mem, false) == false) {
      ERR(L"An error has been triggered from the "
          L"`load_model(%ls, %zu, %zu, false)` function.",
          HIERARCHY_NAME[HIERARCHY::TRAINER].c_str(), allowable_host_mem,
          allowable_devc_mem);
      return false;
    }

    model = models.get_model(HIERARCHY::TRAINER);
  } else {
    if (models.load_model(HIERARCHY::TRAINED, allowable_host_mem,
                          allowable_devc_mem, true) == false) {
      ERR(L"An error has been triggered from the "
          L"`load_model(%ls, %zu, %zu, true)` function.",
          HIERARCHY_NAME[HIERARCHY::TRAINED].c_str(), allowable_host_mem,
          allowable_devc_mem);
      return false;
    }

    model = models.get_model(HIERARCHY::TRAINED);
  }

  INFO(L"Train set:");
  models.get_datasets()->get_dataset(ENV::TRAIN)->simulate_trading(model);

  switch (models.get_datasets()->get_storage_type()) {
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING:
      break;
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING:
      INFO(L"");
      INFO(L"Test set:");
      models.get_datasets()->get_dataset(ENV::TESTG)->simulate_trading(model);
      break;
    case ENUM_TYPE_DATASET_MANAGER_STORAGE::
        TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
      INFO(L"");
      INFO(L"Valid set:");
      models.get_datasets()->get_dataset(ENV::VALID)->simulate_trading(model);

      INFO(L"");
      INFO(L"Test set:");
      models.get_datasets()->get_dataset(ENV::TESTG)->simulate_trading(model);
      break;
    default:
      ERR(L"DatasetV1 storage type (%d | %ls) is not managed in the switch.",
          models.get_datasets()->get_storage_type(),
          ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES[models.get_datasets()
                                                      ->get_storage_type()]
              .c_str());
      return false;
  }

  return true;
}
