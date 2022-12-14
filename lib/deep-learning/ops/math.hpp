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

// Framework:
#include "framework.hpp"

// Base header:
#include "deep-learning/ops/_math.hpp"

// Adept:
#if DEEPLEARNING_USE_ADEPT
#include <adept.h>
#endif

// Standard:
#include <cmath>
#include <utility>

namespace DL::Math {
// clang-format off
#if DEEPLEARNING_USE_ADEPT
template <typename... Args> auto op_abs(Args &&...args) -> decltype(adept::abs(std::forward<Args>(args)...)) { return adept::abs(std::forward<Args>(args)...); }
template <typename... Args> auto op_cos(Args &&...args) -> decltype(adept::cos(std::forward<Args>(args)...)) { return adept::cos(std::forward<Args>(args)...); }
template <typename... Args> auto op_exp(Args &&...args) -> decltype(adept::exp(std::forward<Args>(args)...)) { return adept::exp(std::forward<Args>(args)...); }
template <typename... Args> auto op_max(Args &&...args) -> decltype(adept::max(std::forward<Args>(args)...)) { return adept::max(std::forward<Args>(args)...); }
template <typename... Args> auto op_min(Args &&...args) -> decltype(adept::min(std::forward<Args>(args)...)) { return adept::min(std::forward<Args>(args)...); }
template <typename... Args> auto op_pow(Args &&...args) -> decltype(adept::pow(std::forward<Args>(args)...)) { return adept::pow(std::forward<Args>(args)...); }
template <typename... Args> auto op_sin(Args &&...args) -> decltype(adept::sin(std::forward<Args>(args)...)) { return adept::sin(std::forward<Args>(args)...); }
template <typename... Args> auto op_sqrt(Args &&...args) -> decltype(adept::sqrt(std::forward<Args>(args)...)) { return adept::sqrt(std::forward<Args>(args)...); }
#else
template <typename... Args> auto op_abs(Args &&...args) -> decltype(std::abs(std::forward<Args>(args)...)) { return std::abs(std::forward<Args>(args)...); }
template <typename... Args> auto op_cos(Args &&...args) -> decltype(std::cos(std::forward<Args>(args)...)) { return std::cos(std::forward<Args>(args)...); }
template <typename... Args> auto op_exp(Args &&...args) -> decltype(std::exp(std::forward<Args>(args)...)) { return std::exp(std::forward<Args>(args)...); }
template <typename... Args> auto op_max(Args &&...args) -> decltype(std::max(std::forward<Args>(args)...)) { return std::max(std::forward<Args>(args)...); }
template <typename... Args> auto op_min(Args &&...args) -> decltype(std::min(std::forward<Args>(args)...)) { return std::min(std::forward<Args>(args)...); }
template <typename... Args> auto op_pow(Args &&...args) -> decltype(std::pow(std::forward<Args>(args)...)) { return std::pow(std::forward<Args>(args)...); }
template <typename... Args> auto op_sin(Args &&...args) -> decltype(std::sin(std::forward<Args>(args)...)) { return std::sin(std::forward<Args>(args)...); }
template <typename... Args> auto op_sqrt(Args &&...args) -> decltype(std::sqrt(std::forward<Args>(args)...)) { return std::sqrt(std::forward<Args>(args)...); }
#endif
// clang-format on

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