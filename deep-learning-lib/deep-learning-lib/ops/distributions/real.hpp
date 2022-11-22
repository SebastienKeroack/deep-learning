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
#include "deep-learning-lib/ops/distributions/distribution.hpp"

namespace DL::Dist {
class Real : public DistributionBase {
 public:
  Real(void);
  Real(real const minval, real const maxval,
       unsigned int const seed = DEAFULT_SEED);

  Real &operator=(Real const &cls);

  void copy(Real const &cls);
  /* The generated values follow a uniform distribution in the range
   [minval, maxval). The lower bound minval is included in the range, while
   the upper bound maxval is excluded. */
  void range(real const minval, real const maxval);
  virtual void clear(void);
  virtual void reset(void);

  real operator()(void);

 private:
  real _minval = 0_r;
  real _maxval = 1_r;

  std::uniform_real_distribution<real> _dist;
};
}  // namespace DL::Dist
