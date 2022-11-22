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

#include "deep-learning-lib/io/logger.hpp"
#include "deep-learning-lib/v1/learner/model.hpp"

#include <omp.h>

namespace DL::v1 {
template <typename T>
void euclidean_norm_st(size_t const str, size_t const end, real const max_norm,
                       T *const parameters) {
  size_t i;

  real norm(0_r), desired_norm, parameter;

  // Euclidean Norm.
  for (i = str; i != end; ++i) {
    parameter = cast(parameters[i]);
    norm += parameter * parameter;
  }

  // Square root.
  norm = sqrt(norm);

  // Threshold.
  if (norm >= max_norm) {
    desired_norm = max_norm / norm;

    for (i = str; i != end; ++i) parameters[i] *= desired_norm;
  }
}

template <typename T>
void euclidean_norm_mp(size_t const str, size_t const end, real const max_norm,
                       T *const parameters) {
  int const str_(static_cast<int>(str)), end_(static_cast<int>(end));
  int i;

  real norm(0_r), desired_norm, parameter;

  // Euclidean Norm.
#pragma omp parallel for reduction(+ : norm)
  for (i = str_; i < end_; ++i) {
    parameter = cast(parameters[i]);
    norm += parameter * parameter;
  }

  // Square root.
  norm = sqrt(norm);

  // Threshold.
  if (norm >= max_norm) {
    desired_norm = max_norm / norm;

#pragma omp parallel for schedule(static)
    for (i = str_; i < end_; ++i) parameters[i] *= desired_norm;
  }
}
}  // namespace DL

// clang-format off
template void DL::v1::euclidean_norm_st<var>(size_t const, size_t const, real const, var *const);
template void DL::v1::euclidean_norm_st<real>(size_t const, size_t const, real const, real *const);
template void DL::v1::euclidean_norm_mp<var>(size_t const, size_t const, real const, var *const);
template void DL::v1::euclidean_norm_mp<real>(size_t const, size_t const, real const, real *const);
// clang-format on