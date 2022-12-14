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

#include <string>
#include <map>

namespace DL::v1 {
struct ACTIVATION {
  typedef enum : int {
    NONE = 0,
    COSINE = 1,
    COSINE_SYMMETRIC = 2,
    ELU = 3,
    ELLIOT = 4,
    ELLIOT_SYMMETRIC = 5,
    GAUSSIAN = 6,
    GAUSSIAN_STEPWISE = 7,
    GAUSSIAN_SYMMETRIC = 8,
    ISRU = 9,
    ISRLU = 10,
    LINEAR = 11,
    LINEAR_PIECE = 12,
    LINEAR_PIECE_SYMMETRIC = 13,
    LEAKY_RELU = 14,
    PARAMETRIC_RELU = 15,
    RELU = 16,
    SELU = 17,
    SIGMOID = 18,
    SINE = 19,
    SIGMOID_STEPWISE = 20,
    SINE_SYMMETRIC = 21,
    SOFTMAX = 22,
    TANH = 23,
    TANH_STEPWISE = 24,
    THRESHOLD = 25,
    THRESHOLD_SYMMETRIC = 26,
    LENGTH = 27
  } TYPE;
};

static std::map<ACTIVATION::TYPE, std::wstring> ACTIVATION_NAME = {
    {ACTIVATION::NONE, L"NONE"},
    {ACTIVATION::COSINE, L"Cosine"},
    {ACTIVATION::COSINE_SYMMETRIC, L"Cosine symmetric"},
    {ACTIVATION::ELU, L"Exponential Linear Unit"},
    {ACTIVATION::ELLIOT, L"Elliot"},
    {ACTIVATION::ELLIOT_SYMMETRIC, L"Elliot symmetric"},
    {ACTIVATION::GAUSSIAN, L"Gaussian"},
    {ACTIVATION::GAUSSIAN_STEPWISE, L"Gaussian stepwise"},
    {ACTIVATION::GAUSSIAN_SYMMETRIC, L"Gaussian symmetric"},
    {ACTIVATION::ISRU, L"Inverse Square Root Unit"},
    {ACTIVATION::ISRLU, L"Inverse Square Root Linear Unit"},
    {ACTIVATION::LINEAR, L"Linear"},
    {ACTIVATION::LINEAR_PIECE, L"Linear piece"},
    {ACTIVATION::LINEAR_PIECE_SYMMETRIC, L"Linear piece symmetric"},
    {ACTIVATION::LEAKY_RELU, L"Leaky Rectified Linear Units"},
    {ACTIVATION::PARAMETRIC_RELU, L"[x] Parametric Rectified Linear Units"},
    {ACTIVATION::RELU, L"Rectified Linear Units"},
    {ACTIVATION::SELU, L"Scaled exponential Linear Unit"},
    {ACTIVATION::SIGMOID, L"Sigmoid"},
    {ACTIVATION::SINE, L"Sine"},
    {ACTIVATION::SIGMOID_STEPWISE, L"Sigmoid stepwise"},
    {ACTIVATION::SINE_SYMMETRIC, L"Sine symmetric"},
    {ACTIVATION::SOFTMAX, L"Softmax"},
    {ACTIVATION::TANH, L"Tanh"},
    {ACTIVATION::TANH_STEPWISE, L"Tanh stepwise"},
    {ACTIVATION::THRESHOLD, L"Threshold"},
    {ACTIVATION::THRESHOLD_SYMMETRIC, L"Threshold symmetric"},
    {ACTIVATION::LENGTH, L"LENGTH"}};
}  // namespace DL::v1
