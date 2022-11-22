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

namespace DL::v1 {
bool Model::Set__Regularization__Weight_Decay(
    real const regularization__weight_decay_received) {
  if (regularization__weight_decay_received < 0_r) {
    ERR(
        L"Weight decay (%f) less than zero.",
        cast(regularization__weight_decay_received));

    return false;
  } else if (regularization__weight_decay_received > 1_r) {
    ERR(
        L"Weight decay (%f) bigger than one.",
        cast(regularization__weight_decay_received));

    return false;
  }

  if (this->weight_decay !=
      regularization__weight_decay_received) {
    bool const tmp_use_regularization(this->Use__Regularization_Parameter());

    this->weight_decay = regularization__weight_decay_received;

    if (tmp_use_regularization == false &&
        regularization__weight_decay_received != 0_r) {
      if (this->Allocate__Parameter__Regularization() == false) {
        ERR(
            L"An error has been triggered from the "
            "\"Allocate__Parameter__Regularization()\" function.",);

        return false;
      }

      if (this->pre_training_level != 0_UZ) {
        this->Indexing_Regularization_Parameters__Pre_training();
      } else {
        this->Indexing_Regularization_Parameters();
      }
    }

    if (this->Use__Regularization_Parameter() == false) {
      this->Deallocate__Parameter__Regularization();
    }

#ifdef COMPILE_CUDA
    if (this->is_cu_initialized) {
      this->cumodel->Set__Regularization__Weight_Decay(
          regularization__weight_decay_received);
    }
#endif
  }

  return true;
}
}
