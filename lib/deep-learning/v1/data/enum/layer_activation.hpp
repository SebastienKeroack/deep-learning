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
struct LAYER_ACTIVATION {
  typedef enum : int {
    NONE = 0,
    ASYMMETRIC = 1,
    RECTIFIER = 2,
    SELF_NORMALIZATION = 3,
    SOFTMAX = 4,
    SYMMETRIC = 5,
    LENGTH = 6
  } TYPE;
};

static std::map<LAYER_ACTIVATION::TYPE, std::wstring>
    LAYER_ACTIVATION_NAME = {
        {LAYER_ACTIVATION::NONE, L"NONE"},
        {LAYER_ACTIVATION::ASYMMETRIC, L"Asymmetric"},
        {LAYER_ACTIVATION::RECTIFIER, L"Rectifier"},
        {LAYER_ACTIVATION::SELF_NORMALIZATION, L"Self-normalization"},
        {LAYER_ACTIVATION::SOFTMAX, L"Softmax"},
        {LAYER_ACTIVATION::SYMMETRIC, L"Symmetric"},
        {LAYER_ACTIVATION::LENGTH, L"LENGTH"}};
}  // namespace DL
