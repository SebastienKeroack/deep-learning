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

#pragma once

// Base header:
#include "deep-learning-lib/ops/_math.hpp"

namespace DL::Math {
template <typename T>
T clip(T const val, T const minval, T const maxval) {
  return val < minval ? minval : (val > maxval ? maxval : val);
}

size_t recursive_fused_multiply_add(size_t const *const values,
                                    size_t const depth, size_t const depth_end);

template <typename T>
T reverse_int(T const val);

template <typename T>
T sign(T const val) {
  return static_cast<T>((T(0) < val) - (val < T(0)));
}
}  // namespace DL::Math