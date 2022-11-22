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

#include "deep-learning-lib/v1/data/enum/activation.hpp"

/* Stepwise linear functions used for some of the activation functions */
/* defines used for the stepwise linear functions. */
#define AF_LINEAR_real(v1, r1, v2, r2, x) \
  static_cast<var>((((r2 - r1) * (x - v1)) / (v2 - v1)) + r1)
#define AF_STEPWISE_real(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6,     \
                         minval, maxval, x)                                  \
  (x < v5 ? (x < v3 ? (x < v2 ? (x < v1 ? minval                             \
                                        : AF_LINEAR_real(v1, r1, v2, r2, x)) \
                              : AF_LINEAR_real(v2, r2, v3, r3, x))           \
                    : (x < v4 ? AF_LINEAR_real(v3, r3, v4, r4, x)            \
                              : AF_LINEAR_real(v4, r4, v5, r5, x)))          \
          : (x < v6 ? AF_LINEAR_real(v5, r5, v6, r6, x) : maxval))

// Linear.
#define AF_LINEAR_derive 1_r

// Sigmoid.
#define AF_SIGMOID_real(x) (1_r / (1_r + exp(-x)))
#define AF_SIGMOID_derive(q) (q * (1_r - q))

// Sigmoid symmetric.
#define AF_TANH_real(x) (2_r / (1_r + exp(-2_r * x)) - 1_r)
#define AF_TANH_derive(q) (1_r - q * q)

// Gaussian.
#define AF_GAUSSIAN_real(x) exp(-x* x)
#define AF_GAUSSIAN_derive(x, q) (-2_r * x * q)

// Gaussian symmetric.
#define AF_GAUSSIAN_SYMMETRIC_real(x) (exp(-x * x) * 2_r - 1_r)
#define AF_GAUSSIAN_SYMMETRIC_derive(x, q) (-2_r * x * (q + 1_r))

// Elliot.
#define AF_ELLIOT_real(x) ((x * 0.5_r) / (1_r + abs(x)) + 0.5_r)
#define AF_ELLIOT_derive(x) (1_r / (2_r * (1_r + abs(x)) * (1_r + abs(x))))

// Elliot symmetric.
#define AF_ELLIOT_SYMMETRIC_real(x) (x / (1_r + abs(x)))
#define AF_ELLIOT_SYMMETRIC_derive(x) (1_r / ((1_r + abs(x)) * (1_r + abs(x))))

// Sine.
#define AF_SIN_real(x) (sin(x) * 0.5_r + 0.5_r)
#define AF_SIN_derive(x) (cos(x) * 0.5_r)

// Sine symmetric.
#define AF_SIN_SYMMETRIC_real(x) sin(x)
#define AF_SIN_SYMMETRIC_derive(x) cos(x)

// Cosine.
#define AF_COS_real(x) (cos(x) * 0.5_r + 0.5_r)
#define AF_COS_derive(x) (-sin(x) * 0.5_r)

// Cosine symmetric.
#define AF_COS_SYMMETRIC_real(x) cos(x)
#define AF_COS_SYMMETRIC_derive(x) -sin(x)

//  Inverse square root unit.
#define AF_ISRU_real(x, slope) (x * (1_r / sqrt(1_r + slope * x * x)))
#define AF_ISRU_derive(x, q, slope) (pow(q / x, 3_r))

//  Inverse square root linear unit.
#define AF_ISRLU_real(x, slope) \
  (x < 0_r ? x * (1_r / sqrt(1_r + slope * x * x)) : x)
#define AF_ISRLU_derive(x, q, slope) (x < 0_r ? pow(q / x, 3_r) : 1_r)

// Exponential linear unit.
#define AF_ELU_real(x, slope) (x < 0_r ? slope * (exp(x) - 1_r) : x)
#define AF_ELU_derive(x, q, slope) (x < 0_r ? slope * q : 1_r)

// Scaled exponential linear unit.
#define SELU_Alpha 1.6732632423543772848170429916717_r
#define SELU_Scale 1.0507009873554804934193349852946_r
#define AF_SELU_real(x) \
  (SELU_Scale * (x <= 0_r ? SELU_Alpha * exp(x) - SELU_Alpha : x))
#define AF_SELU_derive(x, q) (SELU_Scale * (x <= 0_r ? (q + SELU_Alpha) : 1_r))

// Rectifier linear unit.
#define AF_RELU_real(x) (x < 0_r ? 0_r : x)
#define AF_RELU_derive(x) (x < 0_r ? 0_r : 1_r)

// Leaky rectifier linear unit.
#define AF_LRELU_real(x, slope) (x < 0_r ? slope * x : x)
#define AF_LRELU_derive(x, slope) (x < 0_r ? slope : 1_r)
#define AF_LRELU_ALPHA 0.01_r

// Parametric rectifier linear unit.
// Not implemented...
#define AF_PRELU_real(x, slope) (x < 0_r ? slope * x : x)
#define AF_PRELU_derive(x, slope) (x < 0_r ? slope : 1_r)
#define AF_PRELU_ALPHA 0.01_r

// Softmax.
#define AF_SOFTMAX_real(x) exp(x)
// w.r.t Cross-entropy.
#define AF_SOFTMAX_CE_derive 1_r
// w.r.t Loss.
#define AF_SOFTMAX_ii_derive(x) (x * (1_r - x))
#define AF_SOFTMAX_ik_derive(x, y_k) (-x * y_k)

#define AF_FIRE(type, x, q)                                                 \
  switch (type) {                                                           \
    case DL::v1::ACTIVATION::LINEAR:                                            \
      q = x;                                                                \
      break;                                                                \
    case DL::v1::ACTIVATION::LINEAR_PIECE:                                      \
      q = x < 0_r ? 0_r : (x > 1_r ? 1_r : x);                              \
      break;                                                                \
    case DL::v1::ACTIVATION::LINEAR_PIECE_SYMMETRIC:                            \
      q = x < -1_r ? -1_r : (x > 1_r ? 1_r : x);                            \
      break;                                                                \
    case DL::v1::ACTIVATION::SIGMOID:                                           \
      q = AF_SIGMOID_real(x);                                               \
      break;                                                                \
    case DL::v1::ACTIVATION::TANH:                                              \
      q = AF_TANH_real(x);                                                  \
      break;                                                                \
    case DL::v1::ACTIVATION::SIGMOID_STEPWISE:                                  \
      q = AF_STEPWISE_real(                                                 \
          -2.64665246009826660156_r, -1.47221946716308593750_r,             \
          -5.49306154251098632812e-01_r, 5.49306154251098632812e-01_r,      \
          1.47221934795379638672_r, 2.64665293693542480469_r,               \
          4.99999988824129104614e-03_r, 5.00000007450580596924e-02_r,       \
          2.50000000000000000000e-01_r, 7.50000000000000000000e-01_r,       \
          9.49999988079071044922e-01_r, 9.95000004768371582031e-01_r, 0_r,  \
          1_r, x);                                                          \
      break;                                                                \
    case DL::v1::ACTIVATION::TANH_STEPWISE:                                     \
      q = AF_STEPWISE_real(                                                 \
          -2.64665293693542480469_r, -1.47221934795379638672_r,             \
          -5.49306154251098632812e-01_r, 5.49306154251098632812e-01_r,      \
          1.47221934795379638672_r, 2.64665293693542480469_r,               \
          -9.90000009536743164062e-01_r, -8.99999976158142089844e-01_r,     \
          -5.00000000000000000000e-01_r, 5.00000000000000000000e-01_r,      \
          8.99999976158142089844e-01_r, 9.90000009536743164062e-01_r, -1_r, \
          1_r, x);                                                          \
      break;                                                                \
    case DL::v1::ACTIVATION::THRESHOLD:                                         \
      q = x < 0_r ? 0_r : 1_r;                                              \
      break;                                                                \
    case DL::v1::ACTIVATION::THRESHOLD_SYMMETRIC:                               \
      q = x < 0_r ? -1_r : 1_r;                                             \
      break;                                                                \
    case DL::v1::ACTIVATION::GAUSSIAN:                                          \
      q = AF_GAUSSIAN_real(x);                                              \
      break;                                                                \
    case DL::v1::ACTIVATION::GAUSSIAN_SYMMETRIC:                                \
      q = AF_GAUSSIAN_SYMMETRIC_real(x);                                    \
      break;                                                                \
    case DL::v1::ACTIVATION::GAUSSIAN_STEPWISE:                                 \
      q = 0;                                                                \
      break;                                                                \
    case DL::v1::ACTIVATION::ELLIOT:                                            \
      q = AF_ELLIOT_real(x);                                                \
      break;                                                                \
    case DL::v1::ACTIVATION::ELLIOT_SYMMETRIC:                                  \
      q = AF_ELLIOT_SYMMETRIC_real(x);                                      \
      break;                                                                \
    case DL::v1::ACTIVATION::SINE:                                              \
      q = AF_SIN_real(x);                                                   \
      break;                                                                \
    case DL::v1::ACTIVATION::SINE_SYMMETRIC:                                    \
      q = AF_SIN_SYMMETRIC_real(x);                                         \
      break;                                                                \
    case DL::v1::ACTIVATION::COSINE:                                            \
      q = AF_COS_real(x);                                                   \
      break;                                                                \
    case DL::v1::ACTIVATION::COSINE_SYMMETRIC:                                  \
      q = AF_COS_SYMMETRIC_real(x);                                         \
      break;                                                                \
    case DL::v1::ACTIVATION::ISRU:                                              \
      q = AF_ISRU_real(x, 1_r);                                             \
      break;                                                                \
    case DL::v1::ACTIVATION::ISRLU:                                             \
      q = AF_ISRLU_real(x, 1_r);                                            \
      break;                                                                \
    case DL::v1::ACTIVATION::ELU:                                               \
      q = AF_ELU_real(x, 1_r);                                              \
      break;                                                                \
    case DL::v1::ACTIVATION::SELU:                                              \
      q = AF_SELU_real(x);                                                  \
      break;                                                                \
    case DL::v1::ACTIVATION::RELU:                                              \
      q = AF_RELU_real(x);                                                  \
      break;                                                                \
    case DL::v1::ACTIVATION::LEAKY_RELU:                                        \
      q = AF_LRELU_real(x, AF_LRELU_ALPHA);                                 \
      break;                                                                \
    case DL::v1::ACTIVATION::PARAMETRIC_RELU:                                   \
      q = AF_PRELU_real(x, AF_PRELU_ALPHA);                                 \
      break;                                                                \
    case DL::v1::ACTIVATION::SOFTMAX:                                           \
      q = AF_SOFTMAX_real(x);                                               \
      break;                                                                \
    default:                                                                \
      break;                                                                \
  }
