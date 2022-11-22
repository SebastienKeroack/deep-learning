/* Copyright 2016, 2019 S�bastien K�roack. All Rights Reserved.

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

// Base header:
#include "deep-learning-lib/ops/distributions/distribution.hpp"

namespace DL::Dist {
class Bernoulli : public DistributionBase {
 public:
  Bernoulli(void);
  Bernoulli(real const probability, unsigned int const seed = DEAFULT_SEED);

  Bernoulli &operator=(Bernoulli const &cls);

  bool operator()(void);

  void copy(Bernoulli const &cls);
  void probability(real const probability);
  virtual void clear(void);
  virtual void reset(void);

 private:
  real _probability = 0_r;

  std::bernoulli_distribution _dist;
};
}  // namespace DL::Dist