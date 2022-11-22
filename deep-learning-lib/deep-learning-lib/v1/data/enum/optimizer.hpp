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
struct OPTIMIZER {
  typedef enum : int {
    NONE = 0,
    // https://openreview.net/forum?id=Bkg3g2R9FX
    ADABOUND = 1,
    ADAM = 2,
    ADAMAX = 3,
    // https://openreview.net/forum?id=Bkg3g2R9FX
    AMSBOUND = 4,
    AMSGRAD = 5,
    GD = 6,
    IRPROP_MINUS = 7,
    IRPROP_PLUS = 8,
    // https://arxiv.org/abs/1805.07557
    NOSADAM = 9,
    QUICKPROP = 10,
    SARPROP = 11,
    LENGTH = 12
  } TYPE;
};

static std::map<OPTIMIZER::TYPE, std::wstring> OPTIMIZER_NAME = {
    {OPTIMIZER::NONE, L"NONE"},
    {OPTIMIZER::ADABOUND, L"AdaBound"},
    {OPTIMIZER::ADAM, L"ADAM"},
    {OPTIMIZER::ADAMAX, L"AdaMax"},
    {OPTIMIZER::AMSBOUND, L"AMSBound"},
    {OPTIMIZER::AMSGRAD, L"AMSGrad"},
    {OPTIMIZER::GD, L"Gradient descent"},
    {OPTIMIZER::IRPROP_MINUS, L"iRPROP-"},
    {OPTIMIZER::IRPROP_PLUS, L"iRPROP+"},
    {OPTIMIZER::NOSADAM, L"Nostalgic Adam - HH"},
    {OPTIMIZER::QUICKPROP, L"[x] QuickProp"},
    {OPTIMIZER::SARPROP, L"[x] SARProp"},
    {OPTIMIZER::LENGTH, L"LENGTH"}};
}  // namespace DL
