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
struct INITIALIZER {
  typedef enum : int {
    NONE = 0,
    // http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    // "Understanding the difficulty of training deep feedforward neural
    // networks".
    GLOROT_GAUSSIAN = 1,
    // http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    // "Understanding the difficulty of training deep feedforward neural
    // networks".
    GLOROT_UNIFORM = 2,
    IDENTITY = 3,
    // https://arxiv.org/abs/1511.06422 "All you need is a good init".
    LSUV = 4,
    // https://arxiv.org/abs/1312.6120 "Exact solutions to the
    // nonlinear dynamics of learning in deep linear neural networks".
    ORTHOGONAL = 5,
    UNIFORM = 6,
    LENGTH = 7
  } TYPE;
};

static std::map<INITIALIZER::TYPE, std::wstring> INITIALIZER_NAME = {
    {INITIALIZER::NONE, L"NONE"},
    {INITIALIZER::GLOROT_GAUSSIAN, L"Glorot gaussian"},
    {INITIALIZER::GLOROT_UNIFORM, L"Glorot uniform"},
    {INITIALIZER::IDENTITY, L"Identity"},
    {INITIALIZER::LSUV, L"[x] LSUV"},
    {INITIALIZER::ORTHOGONAL, L"Orthogonal"},
    {INITIALIZER::UNIFORM, L"Uniform"},
    {INITIALIZER::LENGTH, L"LENGTH"}};

}  // namespace DL
