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
#include "deep-learning/ops/distributions/integer.hpp"

// Deep learning:
#include "deep-learning/io/logger.hpp"

namespace DL::Dist {
template <typename T>
Integer<T>::Integer(void) : DistributionBase() {
  decltype(this->_dist.param()) new_range(this->_minval, this->_maxval);
  this->_dist.param(new_range);
}

template <typename T>
Integer<T>::Integer(T const minval, T const maxval, unsigned int const seed)
    : DistributionBase(seed) {
  this->range(minval, maxval);
}

template <typename T>
Integer<T> &Integer<T>::operator=(Integer<T> const &cls) {
  if (&cls != this) this->copy(cls);
  return *this;
}

template <typename T>
void Integer<T>::clear(void) {
  DistributionBase::clear();
  this->range(T(0), T(1));
}

template <typename T>
void Integer<T>::copy(Integer const &cls) {
  DistributionBase::copy(cls);

  this->_minval = cls._minval;
  this->_maxval = cls._maxval;

  this->_dist = cls._dist;
}

template <typename T>
void Integer<T>::range(T const minval, T const maxval) {
  ASSERT_MSG(minval < maxval, "`minval` cannot be less than `maxval`");

  if (this->_minval == minval && this->_maxval == maxval) return;

  this->_minval = minval;
  this->_maxval = maxval;

  decltype(this->_dist.param()) new_range(minval, maxval);
  this->_dist.param(new_range);
}

template <typename T>
void Integer<T>::reset(void) {
  DistributionBase::reset();
  this->_dist.reset();
}

template <typename T>
T Integer<T>::operator()(void) {
  return this->_dist(this->p_generator_mt19937);
}
}  // namespace DL::Dist

// clang-format off
template class DL::Dist::Integer<int>;
template class DL::Dist::Integer<size_t>;
// clang-format on