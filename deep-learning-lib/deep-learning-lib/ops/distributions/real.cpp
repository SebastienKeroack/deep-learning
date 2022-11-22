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
#include "deep-learning-lib/ops/distributions/real.hpp"

// Deep learning lib:
#include "deep-learning-lib/io/logger.hpp"

namespace DL::Dist {
Real::Real(void) : DistributionBase() {
  this->_dist.param(std::uniform_real_distribution<real>::param_type(
      this->_minval, this->_maxval));
}

Real::Real(real const minval, real const maxval, unsigned int const seed)
    : DistributionBase(seed) {
  this->range(minval, maxval);
}

Real &Real::operator=(Real const &cls) {
  if (&cls != this) this->copy(cls);
  return *this;
}

void Real::clear(void) {
  DistributionBase::clear();
  this->range(0_r, 1_r);
}

void Real::copy(Real const &cls) {
  DistributionBase::copy(cls);

  this->_minval = cls._minval;
  this->_maxval = cls._maxval;

  this->_dist = cls._dist;
}

void Real::range(real const minval, real const maxval) {
  ASSERT_MSG(minval < maxval, "`minval` can not be less than `minval`");

  if (this->_minval == minval && this->_maxval == maxval) return;

  this->_minval = minval;
  this->_maxval = maxval;

  this->_dist.param(
      std::uniform_real_distribution<real>::param_type(minval, maxval));
}

void Real::reset(void) {
  DistributionBase::reset();
  this->_dist.reset();
}

real Real::operator()(void) { return this->_dist(this->p_generator_mt19937); }
}  // namespace DL::Dist
