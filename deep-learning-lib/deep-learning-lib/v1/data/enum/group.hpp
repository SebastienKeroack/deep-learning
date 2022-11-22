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
struct GROUP {
  typedef enum : int { NONE = 0, RESIDUAL = 1, LENGTH = 2 } TYPE;
};

static std::map<GROUP::TYPE, std::wstring> GROUP_NAME = {
    {GROUP::NONE, L"NONE"},
    {GROUP::RESIDUAL, L"Residual"},
    {GROUP::LENGTH, L"LENGTH"}};
}  // namespace DL
