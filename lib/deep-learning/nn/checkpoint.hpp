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

// Deep learning:
#include "deep-learning/data/enum/env.hpp"
#include "deep-learning/data/enum/hierarchy.hpp"

// Eigen:
// `CRTDBG_NEW` is not compatible with `Eigen`.
#ifdef _CRTDBG_MAP_ALLOC
#undef new
#endif

#include <eigen3/Eigen/Dense>

#ifdef _CRTDBG_MAP_ALLOC
#define new CRTDBG_NEW
#endif

// Standard:
#include <string>

namespace DL {
using Matrix3x2 =
    Eigen::Matrix<double, ENV::LENGTH, HIERARCHY::LENGTH, Eigen::RowMajor>;
using MapVec1x2 = Eigen::Map<Eigen::Vector<double, 2>>;

class Checkpoint {
 public:
  Checkpoint(std::wstring const &workdir, bool const load = false,
             std::wstring const &name = L"checkpoint");

  bool inited = false;
  bool is_top;
  bool load(void);
  bool operator()(void);
  bool save(void);

  int last_update_step = 0;

  void reset(void);
  void update(int const step);

  MapVec1x2 operator[](int const key);

  Matrix3x2 values;

  std::wstring path_name;
};
}  // namespace DL