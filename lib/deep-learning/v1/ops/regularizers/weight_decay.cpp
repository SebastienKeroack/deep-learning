/* Copyright 2016, 2022 Sébastien Kéroack. All Rights Reserved.

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
#include "deep-learning/io/logger.hpp"

namespace DL::v1 {
bool Model::set_weight_decay(real const val) {
  if (val < 0_r) {
    ERR(L"Weight decay (%f) less than zero.", val);
    return false;
  } else if (val > 1_r) {
    ERR(L"Weight decay (%f) bigger than one.", val);
    return false;
  }

  if (this->weight_decay != val) {
    bool const use_regularization(this->Use__Regularization_Parameter());

    this->weight_decay = val;

    if (use_regularization == false && val != 0_r) {
      if (this->Allocate__Parameter__Regularization() == false) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Parameter__Regularization()` function.");
        return false;
      }

      if (this->pre_training_level != 0_UZ)
        this->Indexing_Regularization_Parameters__Pre_training();
      else
        this->Indexing_Regularization_Parameters();
    }

    if (this->Use__Regularization_Parameter() == false)
      this->Deallocate__Parameter__Regularization();

#ifdef COMPILE_CUDA
    if (this->is_cu_initialized) this->cumodel->set_weight_decay(val);
#endif
  }

  return true;
}
}  // namespace DL::v1
