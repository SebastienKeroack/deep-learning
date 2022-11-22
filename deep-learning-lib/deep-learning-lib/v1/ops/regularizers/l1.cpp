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
#include "deep-learning-lib/ops/math.hpp"
#include "deep-learning-lib/data/string.hpp"
#include "deep-learning-lib/data/time.hpp"
#include "deep-learning-lib/io/logger.hpp"

#include <omp.h>

using namespace DL::Math;

namespace DL::v1 {
real Model::Get__Regularization__L1(void) const {
  return (this->regularization__l1);
}

bool Model::Set__Regularization__L1(real const regularization__l1_received) {
  if (regularization__l1_received < 0_r) {
    ERR(L"L1 regularization (%f) less than zero.",
        cast(regularization__l1_received));

    return false;
  } else if (regularization__l1_received > 1_r) {
    ERR(L"L1 regularization (%f) bigger than one.",
        cast(regularization__l1_received));

    return false;
  }

  if (this->regularization__l1 != regularization__l1_received) {
    bool const tmp_use_regularization(this->Use__Regularization_Parameter());

    this->regularization__l1 = regularization__l1_received;

    if (tmp_use_regularization == false && regularization__l1_received != 0_r) {
      if (this->Allocate__Parameter__Regularization() == false) {
        ERR(L"An error has been triggered from the "
            "\"Allocate__Parameter__Regularization()\" function.", );

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
      this->cumodel->Set__Regularization__L1(regularization__l1_received);
    }
#endif
  }

  return true;
}

void Model::Update_Derivative_Weight__Regularization__L1(
    size_t const str, size_t const end, size_t const batch_size) {
  if (this->use_mp && this->is_mp_initialized) {
    this->Update_Derivative_Weight__Regularization__L1__OpenMP(str, end,
                                                               batch_size);
  } else {
    this->Update_Derivative_Weight__Regularization__L1__Loop(str, end,
                                                             batch_size);
  }
}

void Model::Update_Derivative_Weight__Regularization__L1__Loop(
    size_t const str, size_t const end, size_t const batch_size) {
  real *derivative_it(this->ptr_array_derivatives_parameters + str);
  real const *const derivative_last(derivative_it + end),
      *mask_it(this->ptr_array_mask_regularized_parameters + str);

  var const *parameter_it(this->ptr_array_parameters + str);

  for (; derivative_it != derivative_last;
       ++derivative_it, ++parameter_it, ++mask_it) {
    *derivative_it +=
        *mask_it * sign(cast(*parameter_it)) * this->regularization__l1;
  }
}

void Model::Update_Derivative_Weight__Regularization__L1__OpenMP(
    size_t const str, size_t const end, size_t const batch_size) {
  int end_(static_cast<int>(end)), i;

  real const *const mask(this->ptr_array_mask_regularized_parameters);
  real *const derivatives(this->ptr_array_derivatives_parameters);

  var const *const parameters(this->ptr_array_parameters);

#pragma omp parallel for schedule(static)
  for (i = static_cast<int>(str); i < end_; ++i) {
    derivatives[i] +=
        mask[i] * sign(cast(parameters[i])) * this->regularization__l1;
  }
}
}  // namespace DL
