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
struct HYPEROPT {
  typedef enum : int {
    NONE = 0,
    GAUSSIAN_SEARCH = 1,
    LENGTH = 2
  } TYPE;
};

static std::map<HYPEROPT::TYPE, std::wstring> HYPEROPT_NAME = {
    {HYPEROPT::NONE, L"NONE"},
    {HYPEROPT::GAUSSIAN_SEARCH, L"Gaussian search"},
    {HYPEROPT::LENGTH, L"LENGTH"}};
}  // namespace DL