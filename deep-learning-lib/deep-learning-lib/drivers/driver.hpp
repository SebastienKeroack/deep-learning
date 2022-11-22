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
#include "deep-learning-lib/data/dataset.hpp"
#include "deep-learning-lib/data/enum/env.hpp"
#include "deep-learning-lib/v1/learner/model.hpp"

namespace DL {
class Driver {
 public:
  Driver(Dataset &dataset, ENV::TYPE const &env_type, v1::Model &model,
         size_t const minibatch_size = 0_UZ);

  double evalt(void);
  double train(void);

  // TODO: Use `minibatch_size`.
  size_t minibatch_size = 0_UZ;

  void evalt_mp(void);
  void evalt_st(void);
  void train_mp(void);
  void train_st(void);

  Dataset &dataset;
  ENV::TYPE env_type;
  v1::Model &model;
};
}  // namespace DL