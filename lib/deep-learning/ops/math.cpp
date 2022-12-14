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
#include "deep-learning/ops/math.hpp"

// Standard:
#include <stdexcept>

namespace DL::Math {
size_t recursive_fused_multiply_add(size_t const *const values,
                                    size_t const depth,
                                    size_t const depth_end) {
  if (depth == depth_end) return values[depth];

  return values[depth] + values[depth] * recursive_fused_multiply_add(
                                             values, depth + 1, depth_end);
}

template <typename T>
T reverse_int(T const val) {
  if constexpr (std::is_same<T, int>::value || std::is_same<T, long>::value ||
                std::is_same<T, unsigned int>::value ||
                std::is_same<T, unsigned long>::value) {
    T const c1(val & 255), c2((val >> 8) & 255), c3((val >> 16) & 255),
        c4((val >> 24) & 255);

    return ((c1 << 24) + (c2 << 16) + (c3 << 8) + c4);
  }

  if constexpr (std::is_same<T, long long>::value ||
                std::is_same<T, unsigned long long>::value) {
    T const c1(val & 255), c2((val >> 8) & 255), c3((val >> 16) & 255),
        c4((val >> 24) & 255), c5((val >> 32) & 255), c6((val >> 40) & 255),
        c7((val >> 48) & 255), c8((val >> 56) & 255);

    return ((c1 << 56) + (c2 << 48) + (c3 << 40) + (c4 << 32) + (c5 << 24) +
            (c6 << 16) + (c7 << 8) + c8);
  }

  throw std::logic_error("NotImplementedException");
}
}  // namespace DL::Math

// clang-format off
template int DL::Math::reverse_int<int>(int const);
template long DL::Math::reverse_int<long>(long const);
template long long DL::Math::reverse_int<long long>(long long const);
template unsigned int DL::Math::reverse_int<unsigned int>(unsigned int const);
template unsigned long DL::Math::reverse_int<unsigned long>(unsigned long const);
template unsigned long long DL::Math::reverse_int<unsigned long long>(unsigned long long const);
// clang-format on