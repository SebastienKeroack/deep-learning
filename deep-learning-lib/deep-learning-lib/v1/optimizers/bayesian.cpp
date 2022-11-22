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

#include "deep-learning-lib/ops/math.hpp"

#include <iostream>
#include <array>

using namespace DL::Math;

namespace DL::v1 {
template <typename T>
T constexpr sqrtNewtonRaphson(T x, T curr, T prev) {
  return curr == prev
              ? curr
              : sqrtNewtonRaphson(x,
                                  T(0.5) * (curr + x / curr), curr);
}

template <typename T>
T constexpr constexpr_sqrt(T x) {
  return x >= T(0) && x < std::numeric_limits<T>::infinity()
              ? sqrtNewtonRaphson(x, x, T(0))
              : std::numeric_limits<T>::quiet_NaN();
}

// Normal probability density function.
template <typename T>
T normal_pdf(T x) {
  T constexpr inv_sqrt_2pi(T(1) / constexpr_sqrt(T(2) * PI<T>));
  return inv_sqrt_2pi * std::exp(x * x * T(-0.5));
}

// Normal cumulative distribution function.
template <typename T>
T normal_cdf(T const x) {
  return erfc(-x / constexpr_sqrt(T(2))) * T(0.5);
}
}
