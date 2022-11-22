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

// Deep learning lib:
#include "deep-learning-lib/data/dtypes.hpp"
#include "deep-learning-lib/data/enum/env.hpp"

// Standard:
#include <string>

namespace DL {
class Dataset {
 protected:
  size_t _n_data = 0_UZ;
  size_t _n_inp = 0_UZ;
  size_t _n_out = 0_UZ;
  size_t _seq_w = 1_UZ;

 public:
  Dataset(std::wstring const &workdir);
  ~Dataset(void);

  virtual bool load(ENV::TYPE const &env_type);

  size_t const &n_data = this->_n_data;
  size_t const &n_inp = this->_n_inp;
  size_t const &n_out = this->_n_out;
  size_t const &seq_w = this->_seq_w;

  virtual void print_sample(size_t const idx) const;

  real const **Xm = nullptr;
  real const **Ym = nullptr;
  real *X = nullptr;
  real *Y = nullptr;

  std::wstring workdir;
};
}  // namespace DL