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

#include "deep-learning/ops/math.hpp"

#include <device_launch_parameters.h>

// Linear
template <typename T>
__device__ inline T Activation_Function_LINEAR_extra_real_t(
    T value_0_received, T r_0, T value_1_received, T r_1, T value_received) {
  return ((((r_1 - r_0) * (value_received - value_0_received)) /
           (value_1_received - value_0_received)) +
          r_0);
}
template <typename T>
__device__ inline T Activation_Function_LINEAR_real_t(
    T const steepness_received) {
  return (steepness_received);
}
template <typename T>
__device__ inline T Activation_Function_LINEAR_derive_t(
    T const steepness_received) {
  return (steepness_received);
}

// Linear piece
template <typename T>
__device__ inline T Activation_Function_LINEAR_PIECE_real_t(
    T const summation_received) {
  return (summation_received < T(0)     ? T(0)
          : (summation_received > T(1)) ? T(1)
                                        : summation_received);
}
template <typename T>
__device__ inline T Activation_Function_LINEAR_PIECE_derive_t(
    T const steepness_received) {
  return (steepness_received);
}

// Linear piece symmetric
template <typename T>
__device__ inline T Activation_Function_LINEAR_PIECE_SYMMETRIC_real_t(
    T const summation_received) {
  return (summation_received < T(-1)    ? T(-1)
          : (summation_received > T(1)) ? T(1)
                                        : summation_received);
}
template <typename T>
__device__ inline T Activation_Function_LINEAR_PIECE_SYMMETRIC_derive_t(
    T const steepness_received) {
  return (steepness_received);
}

// Stepwise
template <typename T>
__device__ inline T Activation_Function_STEPWISE_real_t(
    T value_0_received, T value_1_received, T value_2_received,
    T value_3_received, T value_4_received, T value_5_received, T r_0_received,
    T r_1_received, T r_2_received, T r_3_received, T r_4_received,
    T r_5_received, T min_received, T max_received, T value_received) {
  return (
      value_received < value_4_received
          ? (value_received < value_2_received
                 ? (value_received < value_1_received
                        ? (value_received < value_0_received
                               ? min_received
                               : Activation_Function_LINEAR_extra_real_t(
                                     value_0_received, r_0_received,
                                     value_1_received, r_1_received,
                                     value_received))
                        : Activation_Function_LINEAR_extra_real_t(
                              value_1_received, r_1_received, value_2_received,
                              r_2_received, value_received))
                 : (value_received < value_3_received
                        ? Activation_Function_LINEAR_extra_real_t(
                              value_2_received, r_2_received, value_3_received,
                              r_3_received, value_received)
                        : Activation_Function_LINEAR_extra_real_t(
                              value_3_received, r_3_received, value_4_received,
                              r_4_received, value_received)))
          : (value_received < value_5_received
                 ? Activation_Function_LINEAR_extra_real_t(
                       value_4_received, r_4_received, value_5_received,
                       r_5_received, value_received)
                 : max_received));
}

// Sigmoid
template <typename T>
__device__ inline T Activation_Function_SIGMOID_real_t(
    T const summation_received) {
  return (T(1) / (T(1) + exp(T(-2) * summation_received)));
}
template <typename T>
__device__ inline T Activation_Function_SIGMOID_derive_t(
    T const steepness_received, T const value_received) {
  T tmp_value_clip(DL::Math::clip<var>(value_received, T(0.01), T(0.99)));
  return (T(2) * steepness_received * tmp_value_clip * (T(1) - tmp_value_clip));
}

// Sigmoid symmetric
template <typename T>
__device__ inline T Activation_Function_TANH_real_t(
    T const summation_received) {
  return (T(2) / (T(1) + exp(T(-2) * summation_received)) - T(1));
}
template <typename T>
__device__ inline T Activation_Function_TANH_derive_t(
    T const steepness_received, T const value_received) {
  T tmp_value_clip(DL::Math::clip<var>(value_received, T(-0.98), T(0.98)));
  return (steepness_received * (T(1) - (tmp_value_clip * tmp_value_clip)));
}

// Sigmoid stepwise
template <typename T>
__device__ inline T Activation_Function_SIGMOID_STEPWISE_real_t(
    T const summation_received) {
  return (Activation_Function_STEPWISE_real_t(
      T(-2.64665246009826660156e+00), T(-1.47221946716308593750e+00),
      T(-5.49306154251098632812e-01), T(5.49306154251098632812e-01),
      T(1.47221934795379638672e+00), T(2.64665293693542480469e+00),
      T(4.99999988824129104614e-03), T(5.00000007450580596924e-02),
      T(2.50000000000000000000e-01), T(7.50000000000000000000e-01),
      T(9.49999988079071044922e-01), T(9.95000004768371582031e-01), T(0), T(1),
      summation_received));
}
template <typename T>
__device__ inline T Activation_Function_SIGMOID_STEPWISE_derive_t(
    T const steepness_received, T const value_received) {
  T tmp_value_clip(DL::Math::clip<var>(value_received, T(0.01), T(0.99)));
  return (T(2) * steepness_received * tmp_value_clip * (T(1) - tmp_value_clip));
}

// Sigmoid stepwise symmetric
template <typename T>
__device__ inline T Activation_Function_TANH_STEPWISE_real_t(
    T const summation_received) {
  return (Activation_Function_STEPWISE_real_t(
      T(-2.64665293693542480469e+00), T(-1.47221934795379638672e+00),
      T(-5.49306154251098632812e-01), T(5.49306154251098632812e-01),
      T(1.47221934795379638672e+00), T(2.64665293693542480469e+00),
      T(-9.90000009536743164062e-01), T(-8.99999976158142089844e-01),
      T(-5.00000000000000000000e-01), T(5.00000000000000000000e-01),
      T(8.99999976158142089844e-01), T(9.90000009536743164062e-01), T(-1), T(1),
      summation_received));
}
template <typename T>
__device__ inline T Activation_Function_TANH_STEPWISE_derive_t(
    T const steepness_received, T const value_received) {
  T tmp_value_clip(DL::Math::clip<var>(value_received, T(-0.98), T(0.98)));
  return (steepness_received * (T(1) - (tmp_value_clip * tmp_value_clip)));
}

// Threshold
template <typename T>
__device__ inline T Activation_Function_THRESHOLD_real_t(
    T const summation_received) {
  return (summation_received < T(0) ? T(0) : T(1));
}

// Threshold symmetric
template <typename T>
__device__ inline T Activation_Function_THRESHOLD_SYMMETRIC_real_t(
    T const summation_received) {
  return (summation_received < T(0) ? T(-1) : T(1));
}

// Gaussian
template <typename T>
__device__ inline T Activation_Function_GAUSSIAN_real_t(
    T const summation_received) {
  return (exp(-summation_received * summation_received));
}
template <typename T>
__device__ inline T Activation_Function_GAUSSIAN_derive_t(
    T const steepness_received, T const value_received,
    T const summation_received) {
  return (T(-2) * summation_received * value_received * steepness_received *
          steepness_received);
}

// Gaussian symmetric
template <typename T>
__device__ inline T Activation_Function_GAUSSIAN_SYMMETRIC_real_t(
    T const summation_received) {
  return (exp(-summation_received * summation_received) * T(2) - T(1));
}
template <typename T>
__device__ inline T Activation_Function_GAUSSIAN_SYMMETRIC_derive_t(
    T const steepness_received, T const value_received,
    T const summation_received) {
  return (T(-2) * summation_received * (value_received + T(1)) *
          steepness_received * steepness_received);
}

// Elliot
template <typename T>
__device__ inline T Activation_Function_ELLIOT_real_t(
    T const summation_received) {
  return ((summation_received / T(2)) /
              (T(1) + DL::Math::Absolute<T>(summation_received)) +
          T(0.5));
}
template <typename T>
__device__ inline T Activation_Function_ELLIOT_derive_t(
    T const steepness_received, T const summation_received) {
  return (steepness_received * T(1) /
          (T(2) * (T(1) + DL::Math::Absolute<T>(summation_received)) *
           (T(1) + DL::Math::Absolute<T>(summation_received))));
}

// Elliot symmetric
template <typename T>
__device__ inline T Activation_Function_ELLIOT_SYMMETRIC_real_t(
    T const summation_received) {
  return ((summation_received) /
          (T(1) + DL::Math::Absolute<T>(summation_received)));
}
template <typename T>
__device__ inline T Activation_Function_ELLIOT_SYMMETRIC_derive_t(
    T const steepness_received, T const summation_received) {
  return (steepness_received * T(1) /
          ((T(1) + DL::Math::Absolute<T>(summation_received)) *
           (T(1) + DL::Math::Absolute<T>(steepness_received))));
}

// Sin
template <typename T>
__device__ inline T Activation_Function_SIN_real_t(T const summation_received) {
  return (sin(summation_received) / T(2) + T(0.5));
}
template <typename T>
__device__ inline T Activation_Function_SIN_derive_t(
    T const steepness_received, T const summation_received) {
  return (steepness_received * cos(steepness_received * summation_received) /
          T(2));
}

// Sin symmetric
template <typename T>
__device__ inline T Activation_Function_SIN_SYMMETRIC_real_t(
    T const summation_received) {
  return (sin(summation_received));
}
template <typename T>
__device__ inline T Activation_Function_SIN_SYMMETRIC_derive_t(
    T const steepness_received, T const summation_received) {
  return (steepness_received * cos(steepness_received * summation_received));
}

// Cos
template <typename T>
__device__ inline T Activation_Function_COS_real_t(T const summation_received) {
  return (cos(summation_received) / T(2) + T(0.5));
}
template <typename T>
__device__ inline T Activation_Function_COS_derive_t(
    T const steepness_received, T const summation_received) {
  return (steepness_received * -sin(steepness_received * summation_received) /
          T(2));
}

// Cos symmetric
template <typename T>
__device__ inline T Activation_Function_COS_SYMMETRIC_real_t(
    T const summation_received) {
  return (cos(summation_received));
}
template <typename T>
__device__ inline T Activation_Function_COS_SYMMETRIC_derive_t(
    T const steepness_received, T const summation_received) {
  return (steepness_received * -sin(steepness_received * summation_received));
}

// Rectifier linear unit
template <typename T>
__device__ inline T Activation_Function_RELU_real_t(
    T const summation_received) {
  return (summation_received < T(0) ? T(0) : summation_received);
}
template <typename T>
__device__ inline T Activation_Function_RELU_derive_t(
    T const steepness_received, T const summation_received) {
  return (summation_received < T(0) ? T(0) : steepness_received);
}

// Leaky rectifier linear unit, slope = 0.01
template <typename T>
__device__ inline T Activation_Function_LRELU_real_t(
    T const summation_received) {
  return (summation_received < T(0) ? T(0.01) * summation_received
                                    : summation_received);
}
template <typename T>
__device__ inline T Activation_Function_LRELU_derive_t(
    T const steepness_received, T const summation_received) {
  return (summation_received < T(0) ? T(0.01) * steepness_received
                                    : steepness_received);
}

// Parametric rectifier linear unit
template <typename T>
__device__ inline T Activation_Function_PRELU_real_t(
    T const summation_received) {
  return (summation_received < T(0) ? T(0.01) * summation_received
                                    : summation_received);
}
template <typename T>
__device__ inline T Activation_Function_PRELU_derive_t(
    T const steepness_received, T const summation_received) {
  return (summation_received < T(0) ? T(0.01) * steepness_received
                                    : steepness_received);
}

// Softmax
template <typename T>
__device__ inline T Activation_Function_SOFTMAX_real_t(
    T const summation_received) {
  return (exp(summation_received));
}
// w.r.t Cross-entropy
template <typename T>
__device__ inline T Activation_Function_SOFTMAX_CE_derive_t(
    T const steepness_received) {
  return (steepness_received);
}
// w.r.t MSE
template <typename T>
__device__ inline T Activation_Function_SOFTMAX_MSE_derive_t(
    T const steepness_received, T const value_received) {
  return (steepness_received * value_received * (T(1) - value_received));
}
