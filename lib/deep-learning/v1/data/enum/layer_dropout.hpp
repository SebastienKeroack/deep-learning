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
struct LAYER_DROPOUT {
  typedef enum : int {
    NONE = 0,
    // https://arxiv.org/abs/1706.02515
    ALPHA = 1,
    BERNOULLI = 2,
    BERNOULLI_INVERTED = 3,
    GAUSSIAN = 4,
    // https://arxiv.org/abs/1802.02375
    SHAKEDROP = 5,
    UOUT = 6,
    ZONEOUT = 7,
    LENGTH = 8
  } TYPE;
};

static std::map<LAYER_DROPOUT::TYPE, std::wstring> LAYER_DROPOUT_NAME = {
    {LAYER_DROPOUT::NONE, L"NONE"},
    {LAYER_DROPOUT::ALPHA, L"[x] Alpha"},
    {LAYER_DROPOUT::BERNOULLI, L"Bernoulli"},
    {LAYER_DROPOUT::BERNOULLI_INVERTED, L"Bernoulli inverted"},
    {LAYER_DROPOUT::GAUSSIAN, L"Gaussian"},
    {LAYER_DROPOUT::SHAKEDROP, L"ShakeDrop"},
    {LAYER_DROPOUT::UOUT, L"Uout"},
    {LAYER_DROPOUT::ZONEOUT, L"Zoneout"},
    {LAYER_DROPOUT::LENGTH, L"LENGTH"}};
}  // namespace DL
