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

// Base header:
#include "deep-learning/ops/distributions/distribution.hpp"

namespace DL::Dist {
class Gaussian : public DistributionBase {
 public:
  Gaussian(void);
  Gaussian(real const mean, real const std,
           unsigned int const seed = DEAFULT_SEED);

  Gaussian &operator=(Gaussian const &cls);

  void copy(Gaussian const &cls);
  void range(real const mean, real const std);
  virtual void clear(void);
  virtual void reset(void);

  real operator()(void);

 private:
  real _mean = 0;
  real _std = 1;

  std::normal_distribution<real> _dist;
};
}  // namespace DL::Dist
