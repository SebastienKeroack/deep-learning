/* Copyright 2022 Sébastien Kéroack. All Rights Reserved.

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
#include "deep-learning/nn/checkpoint.hpp"

// Deep learning:
#include "deep-learning/data/enum/env.hpp"
#include "deep-learning/data/enum/hierarchy.hpp"
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/logger.hpp"

using namespace DL::File;
using namespace DL::Str;

namespace DL {
Checkpoint::Checkpoint(std::wstring const &workdir, bool const load,
                       std::wstring const &name) {
  this->path_name = workdir + OS_SEP + name;

  if (load) this->load();
  if (this->inited == false) this->reset();
}

bool Checkpoint::load(void) {
  if (path_exist(this->path_name) == false) {
    DEBUG(L"No such file `%ls`.", this->path_name.c_str());
    return false;
  }

  std::wifstream file;
  if (iopen(file, this->path_name, std::wios::in | std::wios::binary) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`File::iopen(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  double *data(this->values.data());
  for (int i(0); i != this->values.size(); ++i)
    if (parse_real(file, data[i]) == false) {
      ERR(L"An error has been triggered from the "
          L"`File::parse_real(data[%d])` function.",
          i);
      return false;
    }

  if (iclose(file, this->path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::iclose(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  this->operator()();

  return true;
}

bool Checkpoint::operator()(void) {
  MapVec1x2 const train(this->values.data() + ENV::TRAIN * 2);
  MapVec1x2 const valid(this->values.data() + ENV::VALID * 2);

  this->is_top = (((train[HIERARCHY::TRAINER] <= train[HIERARCHY::TRAINED] ||
                    train[HIERARCHY::TRAINER] <= valid[HIERARCHY::TRAINED]) &&
                   valid[HIERARCHY::TRAINER] < valid[HIERARCHY::TRAINED]) ||
                  ((valid[HIERARCHY::TRAINER] <= valid[HIERARCHY::TRAINED] ||
                    valid[HIERARCHY::TRAINER] <= train[HIERARCHY::TRAINED]) &&
                   train[HIERARCHY::TRAINER] < train[HIERARCHY::TRAINED])) ||
                 this->inited == false;

  this->inited = true;

  return this->is_top;
}

bool Checkpoint::save(void) {
  std::wofstream file;
  if (wopen(file, this->path_name, std::wios::out | std::wios::binary) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`File::wopen(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  file << std::setprecision(15);
  double const *data(this->values.data());
  for (int i(0); i != this->values.size(); ++i) file << data[i] << L" ";

  if (wclose(file, this->path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::wclose(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  return true;
}

void Checkpoint::reset(void) {
  this->values.setConstant(HUGE_VAL);
  this->inited = false;
  this->is_top = false;
}

void Checkpoint::update(int const step = 0) {
  this->values.col(HIERARCHY::TRAINED) = this->values.col(HIERARCHY::TRAINER);
  this->last_update_step = step;
}

MapVec1x2 Checkpoint::operator[](int const key) {
  return MapVec1x2(this->values.data() + key * 2);
}
}  // namespace DL