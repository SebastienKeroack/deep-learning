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

#pragma once

// Standard:
#include <string>
#include <vector>

namespace DL {
class Flags {
 public:
  Flags(int const n_args, char const *const args[]);
  Flags(int const n_args, wchar_t const *const args[]);

  bool const &load = this->_load;
  bool const &save = this->_save;
  bool exist(wchar_t const *const key);
  bool operator[](wchar_t const *const key);

  double get(wchar_t const *const key, double const default_);

 private:
  bool _load = false;
  bool _save = false;

  void collect_default_args(void);
  void register_global_scope(void);

  std::vector<std::wstring> args;
};

inline class Flags *flags = nullptr;

#define ARG_EXIST(key) flags->exist(key)
}  // namespace DL
