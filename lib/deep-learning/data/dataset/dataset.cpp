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
#include "pch.hpp"

// File header:
#include "deep-learning/data/dataset.hpp"

// FMT:
#include <fmt/core.h>
#include <fmt/xchar.h>

// Standard:
#include <iostream>
#include <string>

namespace DL {
Dataset::Dataset(std::wstring const &workdir) : workdir(workdir) {}

Dataset::~Dataset(void) {
  delete[] (this->X);
  delete[] (this->Xm);
  delete[] (this->Y);
  delete[] (this->Ym);
}

bool Dataset::load(ENV::TYPE const &env_type) { return false; }

void Dataset::print_sample(size_t const idx) const {
  real const *X(this->Xm[idx]);
  real const *Y(this->Ym[idx]);
  size_t i;

  std::wcout << fmt::format(L"X[{}]:", idx) << std::endl;
  for (i = 0_UZ; i != this->n_inp - 1_UZ; ++i) std::wcout << X[i] << L" ";
  std::wcout << X[i] << std::endl << std::endl;

  std::wcout << fmt::format(L"Y[{}]:", idx) << std::endl;
  for (i = 0_UZ; i != this->n_out - 1_UZ; ++i) std::wcout << Y[i] << L" ";
  std::wcout << Y[i] << std::endl << std::endl;
}
}  // namespace DL