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
struct LAYER_NORM {
  typedef enum : int {
    NONE = 0,
    BATCH_NORMALIZATION = 1,
    BATCH_RENORMALIZATION = 2,
    GHOST_BATCH_NORMALIZATION = 3,
    STREAMING_NORMALIZATION = 4,
    LENGTH = 5
  } TYPE;
};

static std::map<LAYER_NORM::TYPE, std::wstring> LAYER_NORM_NAME = {
    {LAYER_NORM::NONE, L"NONE"},
    {LAYER_NORM::BATCH_NORMALIZATION, L"Batch normalization"},
    {LAYER_NORM::BATCH_RENORMALIZATION, L"Batch renormalization"},
    {LAYER_NORM::GHOST_BATCH_NORMALIZATION, L"[x] Ghost batch normalization"},
    {LAYER_NORM::STREAMING_NORMALIZATION, L"[x] Streaming normalization"},
    {LAYER_NORM::LENGTH, L"LENGTH"}};

}  // namespace DL
