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
#include "deep-learning/data/dataset.hpp"
#include "deep-learning/data/enum/env.hpp"
#include "deep-learning/data/shape.hpp"

namespace DL {
class MNIST : public Dataset {
 public:
  MNIST(std::wstring const &workdir);

  bool load(ENV::TYPE const &env_type);

  void print_sample(size_t const idx) const;

 private:
  bool download(std::wstring const &file_name, std::wstring const &path_name);

  int _n_cols = 0;
  int _n_rows = 0;

  Shape const load_images(std::wstring const &path_name);
  Shape const load_labels(std::wstring const &path_name);
};
}  // namespace DL