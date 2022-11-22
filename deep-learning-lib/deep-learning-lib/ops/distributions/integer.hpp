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
template <typename T>
class Integer : public DistributionBase {
 public:
  Integer(void);
  Integer(T const minval, T const maxval,
          unsigned int const seed = DEAFULT_SEED);

  Integer &operator=(Integer const &cls);

  void copy(Integer const &cls);
  void range(T const minval, T const maxval);
  virtual void clear(void);
  virtual void reset(void);

  T operator()(void);

 private:
  T _minval = 0;
  T _maxval = 1;

  std::uniform_int_distribution<T> _dist;
};
}  // namespace DL::Dist
