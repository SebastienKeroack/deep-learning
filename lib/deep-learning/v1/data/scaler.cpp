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

#include "pch.hpp"

#include "deep-learning/v1/data/scaler.hpp"

namespace DL::v1 {
ScalerMinMax &ScalerMinMax::operator=(ScalerMinMax const &cls) {
  if (&cls != this) this->copy(cls);
  return *this;
}

void ScalerMinMax::copy(ScalerMinMax const &cls) {
  this->str_index = cls.str_index;
  this->end_index = cls.end_index;

  this->minval = cls.minval;
  this->maxval = cls.maxval;
  this->minrge = cls.minrge;
  this->maxrge = cls.maxrge;
}

ScalerZeroCentered &ScalerZeroCentered::operator=(
    ScalerZeroCentered const &cls) {
  if (&cls != this)
    this->copy(cls);
  return *this;
}

void ScalerZeroCentered::copy(ScalerZeroCentered const &cls) {
  this->str_index = cls.str_index;
  this->end_index = cls.end_index;

  this->multiplier = cls.multiplier;
}
}  // namespace DL