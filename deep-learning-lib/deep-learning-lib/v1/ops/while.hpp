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

#include <chrono>

#include "deep-learning-lib/v1/data/enum/while.hpp"

namespace DL::v1 {
struct WhileCond {
  WhileCond(void) {}

  WHILE_MODE::TYPE type = WHILE_MODE::ITERATION;

  union {
    unsigned long long maximum_iterations = 1ull;

    std::chrono::system_clock::time_point expiration;
  };
};
}  // namespace DL
