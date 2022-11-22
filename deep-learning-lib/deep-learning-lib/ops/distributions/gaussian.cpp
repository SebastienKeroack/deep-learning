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
#include "deep-learning-lib/ops/distributions/gaussian.hpp"

namespace DL::Dist {
Gaussian::Gaussian(void) : DistributionBase() {
  this->_dist.param(
      std::normal_distribution<real>::param_type(this->_mean, this->_std));
}

Gaussian::Gaussian(real const mean, real const std, unsigned int const seed)
    : DistributionBase(seed) {
  this->range(mean, std);
}

Gaussian &Gaussian::operator=(Gaussian const &cls) {
  if (&cls != this) this->copy(cls);
  return *this;
}

void Gaussian::clear(void) {
  DistributionBase::clear();
  this->range(0_r, 1_r);
}

void Gaussian::copy(Gaussian const &cls) {
  DistributionBase::copy(cls);

  this->_mean = cls._mean;
  this->_std = cls._std;

  this->_dist = cls._dist;
}

void Gaussian::range(real const mean, real const std) {
  if (this->_mean == mean && this->_std == std) return;

  this->_mean = mean;
  this->_std = std;

  this->_dist.param(std::normal_distribution<real>::param_type(mean, std));
}

void Gaussian::reset(void) {
  DistributionBase::reset();
  this->_dist.reset();
}

real Gaussian::operator()(void) {
  return this->_dist(this->p_generator_mt19937);
}
}  // namespace DL::Dist
