/* Copyright 2022 Sébastien Kéroack. All Rights Reserved.

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

// Standard:
#include <vector>

namespace DL {
struct Shape {
  Shape(void);
  Shape(std::vector<int> const &dims);

  int num_elements(void) const;
  int &operator[](size_t const idx);
  int const &operator[](size_t const idx) const;

  std::vector<int> dims;
};
}  // namespace DL