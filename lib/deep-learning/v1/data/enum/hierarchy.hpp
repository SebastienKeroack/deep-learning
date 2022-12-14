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
struct HIERARCHY {
  typedef enum : int {
    TRAINER = 0,
    TRAINED = 1,
    COMPETITOR = 2,
    ALL = 3,
    LENGTH = 4
  } TYPE;
};

static std::map<HIERARCHY::TYPE, std::wstring> HIERARCHY_NAME = {
    {HIERARCHY::TRAINER, L"Trainer"},
    {HIERARCHY::TRAINED, L"Trained"},
    {HIERARCHY::COMPETITOR, L"Competitor"},
    {HIERARCHY::ALL, L"All"},
    {HIERARCHY::LENGTH, L"LENGTH"}};
}  // namespace DL
