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

// PCH:
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/data/shape.hpp"

namespace DL {
Shape::Shape(void) {}

Shape::Shape(std::vector<int> const &dims) : dims(dims) {}

int Shape::num_elements(void) const {
  int n(this->dims[0]);
  for (size_t i(1_UZ); i != this->dims.size(); ++i) {
    if (-1 == this->dims[i]) return -1;
    n *= this->dims[i];
  }
  return n;
}

int &Shape::operator[](size_t const idx) { return this->dims[idx]; }

int const &Shape::operator[](size_t const idx) const { return this->dims[idx]; }
}  // namespace DL
