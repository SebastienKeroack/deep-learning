/* Copyright 2022 Sébastien Kéroack. All Rights Reserved.

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

// Deep learning lib:
#include "deep-learning-lib/io/logger.hpp"
#include "deep-learning-lib/ops/math.hpp"
#include "deep-learning-lib/v1/ops/activations/functions.hpp"

// Standard:
#include <iostream>

using namespace DL::Math;

namespace DL {
TEST(ADEPT, Dense3x1SigmoidLinearLoss) {
  adept::Stack stack;

  var X[3] = {0.5_r, 0.5_r, 0.5_r};
  var W[3] = {1.3_r, -5.3_r, 4.3_r};
  var B[1] = {0.4_r};
  var H[1] = {0_r};
  var Q[1] = {0_r};

  real constexpr Y[1] = {0.5_r};
  real loss;
  real dQ[1] = {0_r};

  auto summation_fn([](var const *const X, var const *const W,
                       var const *const B, var *const H) -> void {
    H[0] = (X[0] * W[0]) + (X[1] * W[1]) + (X[2] * W[2]) + B[0];
  });

  auto sigmoid_fn([](var const *const H, var *const Q) -> void {
    Q[0] = 1_r / (1_r + exp(-H[0]));
  });

  auto d_sigmoid_fn([](real *const dH, var const *const Q) -> void {
    dH[0] = cast(Q[0]) * (1_r - cast(Q[0]));
  });

  auto linear_loss_fn([](real const *const Y, var const *const Q) -> real {
    return Y[0] - cast(Q[0]);
  });

  auto d_linear_loss_fn([](real *const dQ, real const *const Y,
                           var const *const Q) -> void { dQ[0] = -1_r; });

  stack.new_recording();

  summation_fn(X, W, B, H);

  sigmoid_fn(H, Q);

  loss = linear_loss_fn(Y, Q);

  d_linear_loss_fn(dQ, Y, Q);

  Q[0].set_gradient(dQ[0]);

  stack.reverse();

  real dW[3] = {0_r};
  real dH[1] = {0_r};
  real dB[1] = {0_r};

  d_sigmoid_fn(dH, Q);

  for (int i(0); i != 3; ++i) {
    dW[i] = cast(X[i]) * dH[0] * dQ[0];
    ASSERT_DOUBLE_EQ(dW[i], W[i].get_gradient());
  }
  dB[0] = dH[0] * dQ[0];
  ASSERT_DOUBLE_EQ(dB[0], B[0].get_gradient());
}

TEST(ADEPT, Dense3x1SigmoidMSELoss) {
  adept::Stack stack;

  var X[3] = {0.5_r, 0.5_r, 0.5_r};
  var W[3] = {1.3_r, -5.3_r, 4.3_r};
  var B[1] = {0.4_r};
  var H[1] = {0_r};
  var Q[1] = {0_r};

  real constexpr Y[1] = {0.5_r};
  real loss;
  real dQ[1] = {0_r};

  auto summation_fn([](var const *const X, var const *const W,
                       var const *const B, var *const H) -> void {
    H[0] = (X[0] * W[0]) + (X[1] * W[1]) + (X[2] * W[2]) + B[0];
  });

  auto sigmoid_fn([](var const *const H, var *const Q) -> void {
    Q[0] = 1_r / (1_r + exp(-H[0]));
  });

  auto d_sigmoid_fn([](real *const dH, var const *const Q) -> void {
    dH[0] = cast(Q[0]) * (1_r - cast(Q[0]));
  });

  auto mse_loss_fn([](real const *const Y, var const *const Q) -> real {
    return pow(Y[0] - cast(Q[0]), 2_r);
  });

  auto d_mse_loss_fn(
      [](real *const dQ, real const *const Y, var const *const Q) -> void {
        dQ[0] = 2_r * (cast(Q[0]) - Y[0]);
      });

  stack.new_recording();

  summation_fn(X, W, B, H);

  sigmoid_fn(H, Q);

  loss = mse_loss_fn(Y, Q);

  d_mse_loss_fn(dQ, Y, Q);

  Q[0].set_gradient(dQ[0]);

  stack.reverse();

  real dW[3] = {0_r};
  real dH[1] = {0_r};
  real dB[1] = {0_r};

  d_sigmoid_fn(dH, Q);

  for (int i(0); i != 3; ++i) {
    dW[i] = cast(X[i]) * dH[0] * dQ[0];
    ASSERT_DOUBLE_EQ(dW[i], W[i].get_gradient());
  }
  dB[0] = dH[0] * dQ[0];
  ASSERT_DOUBLE_EQ(dB[0], B[0].get_gradient());
}

TEST(ADEPT, Dense1x3SoftmaxCCELoss) {
  adept::Stack stack;

  var X[1] = {0.5_r};
  var W[3] = {1.3_r, -5.3_r, 4.3_r};
  var B[3] = {0.4_r, 0.4_r, 0.4_r};
  var H[3] = {0_r};
  var Q[3] = {0_r};

  real constexpr Y[3] = {1_r, 0_r, 0_r};
  real loss;
  real dQ[3] = {0_r};

  auto summation_fn([](var const *const X, var const *const W,
                       var const *const B, var *const H) -> void {
    H[0] = (X[0] * W[0]) + B[0];
    H[1] = (X[0] * W[1]) + B[1];
    H[2] = (X[0] * W[2]) + B[2];
  });

  auto softmax_fn([](var const *const H, var *const Q) -> void {
    var sum(0_r), h_max(-std::numeric_limits<real>::max());
    // Get h max:
    for (int i(0); i != 3; ++i) h_max = std::max(H[i], h_max);
    // Activation function with summation:
    for (int i(0); i != 3; ++i) sum += Q[i] = exp(H[i] - h_max);
    // Convert division to multiplication:
    sum = 1_r / sum;
    // Normalize activation function:
    for (int i(0); i != 3; ++i) Q[i] *= sum;
  });

  auto d_softmax_fn([](real *const dE, real const *const dQ, var const *const Q,
                       real const *const Y) -> void {
    for (int i(0); i != 3; ++i)
      for (int j(0); j != 3; ++j)
        dE[i] += (cast(Q[i]) * static_cast<float>(i == j) -
                  cast(Q[j]) * cast(Q[i])) *
                 dQ[j];
  });

  auto cce_loss_fn([](real const *const Y, var const *const Q) -> real {
    real loss(0_r), q;
    for (int i(0); i != 3; ++i) {
      q = clip(cast(Q[i]), ::EPSILON, 1_r - ::EPSILON);
      loss += -(Y[i] * log(q));
    }
    return loss;
  });

  auto cce_derivative_fn(
      [](real *const dQ, real const *const Y, var const *const Q) -> void {
        real q;
        for (int i(0); i != 3; ++i) {
          q = clip(cast(Q[i]), ::EPSILON, 1_r - ::EPSILON);
          dQ[i] = Y[i] == 1_r ? -1_r / q : 0_r;
        }
      });

  stack.new_recording();

  summation_fn(X, W, B, H);

  softmax_fn(H, Q);

  loss = cce_loss_fn(Y, Q);

  cce_derivative_fn(dQ, Y, Q);

  for (int i(0); i != 3; ++i) Q[i].set_gradient(dQ[i]);

  stack.reverse();

  real dW[3] = {0_r};
  real dE[3] = {0_r};
  real dB[3] = {0_r};

  d_softmax_fn(dE, dQ, Q, Y);

  for (int i(0); i != 3; ++i) {
    dW[i] = cast(X[0]) * dE[i];
    ASSERT_DOUBLE_EQ(dW[i], W[i].get_gradient());
  }

  for (int i(0); i != 3; ++i) {
    dB[i] = dE[i];
    ASSERT_DOUBLE_EQ(dB[i], B[i].get_gradient());
  }
}

TEST(ADEPT, Dense1x3SoftmaxMSELoss) {
  adept::Stack stack;

  var X[1] = {0.5_r};
  var W[3] = {1.3_r, -5.3_r, 4.3_r};
  var B[3] = {0.4_r, 0.4_r, 0.4_r};
  var H[3] = {0_r};
  var Q[3] = {0_r};

  real constexpr Y[3] = {1_r, 0_r, 0_r};
  real loss;
  real dQ[3] = {0_r};

  auto summation_fn([](var const *const X, var const *const W,
                       var const *const B, var *const H) -> void {
    for (int i(0); i != 3; ++i) H[i] = (X[0] * W[i]) + B[i];
  });

  auto softmax_fn([](var const *const H, var *const Q) -> void {
    var sum(0_r), h_max(-std::numeric_limits<real>::max());
    // Get h max:
    for (int i(0); i != 3; ++i) h_max = std::max(H[i], h_max);
    // Activation function with summation:
    for (int i(0); i != 3; ++i) sum += Q[i] = exp(H[i] - h_max);
    // Convert division to multiplication:
    sum = 1_r / sum;
    // Normalize activation function:
    for (int i(0); i != 3; ++i) Q[i] *= sum;
  });

  auto d_softmax_fn([](real *const dE, real const *const dQ, var const *const Q,
                       real const *const Y) -> void {
    for (int i(0); i != 3; ++i)
      for (int j(0); j != 3; ++j)
        dE[i] += (cast(Q[i]) * static_cast<float>(i == j) -
                  cast(Q[j]) * cast(Q[i])) *
                 dQ[j];
  });

  auto mse_loss_fn([](real const *const Y, var const *const Q) -> real {
    real loss(0_r);
    for (int i(0); i != 3; ++i) loss += pow(Y[i] - cast(Q[i]), 2_r);
    return loss / 3.0;
  });

  auto d_mse_loss_fn(
      [&](real *const dQ, real const *const Y, var const *const Q) -> void {
        for (int i(0); i != 3; ++i) dQ[i] = 2_r * (cast(Q[i]) - Y[i]) / 3.0;
      });

  stack.new_recording();

  summation_fn(X, W, B, H);

  softmax_fn(H, Q);

  loss = mse_loss_fn(Y, Q);

  d_mse_loss_fn(dQ, Y, Q);

  for (int i(0); i != 3; ++i) Q[i].set_gradient(dQ[i]);

  stack.reverse();

  real dW[3] = {0_r};
  real dE[3] = {0_r};
  real dB[3] = {0_r};

  d_softmax_fn(dE, dQ, Q, Y);

  for (int i(0); i != 3; ++i) {
    dW[i] = cast(X[0]) * dE[i];
    ASSERT_DOUBLE_EQ(dW[i], W[i].get_gradient());
  }

  for (int i(0); i != 3; ++i) {
    dB[i] = dE[i];
    ASSERT_DOUBLE_EQ(dB[i], B[i].get_gradient());
  }
}
}  // namespace DL