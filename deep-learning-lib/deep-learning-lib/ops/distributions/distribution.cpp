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

// PCH:
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/ops/distributions/distribution.hpp"

namespace DL::Dist {
DistributionBase::DistributionBase(unsigned int const seed) {
  this->seed(seed);
}

DistributionBase &DistributionBase::operator=(DistributionBase const &cls) {
  if (&cls != this) this->copy(cls);
  return *this;
}

DistributionBase::~DistributionBase(void) {}

void DistributionBase::clear(void) { this->seed(DEAFULT_SEED); }

void DistributionBase::copy(DistributionBase const &cls) {
  this->p_generator_mt19937 = cls.p_generator_mt19937;
  this->p_seed = cls.p_seed;
}

void DistributionBase::reset(void) { this->seed(this->p_seed); }

void DistributionBase::seed(unsigned int const seed) {
  this->p_generator_mt19937.seed(seed);
  this->p_seed = seed;
}
}  // namespace DL::Dist
