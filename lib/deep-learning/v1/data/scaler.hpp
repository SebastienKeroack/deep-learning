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

#pragma once

#include "deep-learning/data/dtypes.hpp"

namespace DL::v1 {
struct ScalerMinMax {
  ScalerMinMax(void) {}

  ScalerMinMax &operator=(ScalerMinMax const &cls);

  void copy(ScalerMinMax const &cls);

  size_t str_index = 0_UZ;
  size_t end_index = 0_UZ;

  real minval = 0;
  real maxval = 1;
  real minrge = 0;
  real maxrge = 1;
};

struct ScalerZeroCentered {
  ScalerZeroCentered(void) {}

  ScalerZeroCentered &operator=(ScalerZeroCentered const &cls);

  void copy(ScalerZeroCentered const &cls);

  size_t str_index = 0_UZ;
  size_t end_index = 0_UZ;

  real multiplier = 0;
};
}  // namespace DL