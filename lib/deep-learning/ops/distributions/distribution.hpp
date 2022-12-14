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

#include <random>

namespace DL::Dist {
inline constexpr unsigned int DEAFULT_SEED(5489u);

class DistributionBase {
 protected:
  unsigned int p_seed;

  // https://fr.wikipedia.org/wiki/Mersenne_Twister
  std::mt19937 p_generator_mt19937;

 public:
  DistributionBase(unsigned int const seed = DEAFULT_SEED);

  virtual ~DistributionBase(void);

  DistributionBase &operator=(DistributionBase const &cls);

  void copy(DistributionBase const &cls);
  void seed(unsigned int const seed);
  virtual void clear(void);
  virtual void reset(void);
};
}  // namespace DL::Dist
