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
#include "pch.hpp"

// File header:
#include "deep-learning/ops/distributions/bernoulli.hpp"

// Deep learning:
#include "deep-learning/io/logger.hpp"

namespace DL::Dist {
Bernoulli::Bernoulli(void) : DistributionBase() {}

Bernoulli::Bernoulli(real const probability, unsigned int const seed)
    : DistributionBase(seed) {
  this->probability(probability);
}

Bernoulli &Bernoulli::operator=(Bernoulli const &cls) {
  if (&cls != this) this->copy(cls);
  return *this;
}

bool Bernoulli::operator()(void) {
  return this->_dist(this->p_generator_mt19937);
}

void Bernoulli::clear(void) {
  DistributionBase::clear();
  this->_probability = 0_r;
}

void Bernoulli::copy(Bernoulli const &cls) {
  DistributionBase::copy(cls);

  this->_probability = cls._probability;

  this->_dist = cls._dist;
}

void Bernoulli::probability(real const probability) {
  ASSERT_MSG(probability >= 0_r && probability < 0_r,
             "`probability` need to be in the range of [0, 1).");

  if (this->_probability == probability) return;

  this->_probability = probability;

  decltype(this->_dist.param()) new_probability(probability);
  this->_dist.param(new_probability);
}

void Bernoulli::reset(void) {
  DistributionBase::reset();
  this->_dist.reset();
}
}  // namespace DL::Dist
