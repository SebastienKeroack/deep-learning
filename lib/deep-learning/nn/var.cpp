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
#include "deep-learning/data/string.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/nn/var.hpp"

// Deep learning:
#include "deep-learning/io/file.hpp"

using namespace DL::File;
using namespace DL::Str;

namespace DL {
template <typename T>
Var<T>::Var(std::wstring const &name, T const initial,
            std::wstring const &workdir, bool const load)
    : value(initial) {
  this->path_name = workdir + OS_SEP + name;
  if (load) this->load();
}

template <typename T>
bool Var<T>::save(void) {
  std::wofstream file;
  if (wopen(file, this->path_name, std::wios::out | std::wios::binary) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`File::wopen(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  file << this->value;

  if (wclose(file, this->path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::wclose(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  return true;
}

template <typename T>
bool Var<T>::load(void) {
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

  file >> this->value;

  if (iclose(file, this->path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::iclose(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  return true;
}
}  // namespace DL

// clang-format off
template class DL::Var<int>;
// clang-format on