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
#include "deep-learning/io/flags.hpp"

// Standard:
#include <codecvt>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

namespace DL {
Flags::Flags(int const n_args, char const *const args[]) {
  this->register_global_scope();

  std::vector<std::wstring> args_(n_args);
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t> > converter;
  for (int i(0); i != n_args; ++i)
    args_[i] = converter.from_bytes(std::string(args[i]));
  this->args = args_;

  this->collect_default_args();
}

Flags::Flags(int const n_args, wchar_t const *const args[]) {
  this->register_global_scope();

  std::vector<std::wstring> args_(n_args);
  for (int i(0); i != n_args; ++i) args_[i] = args[i];
  this->args = args_;

  this->collect_default_args();
}

bool Flags::exist(wchar_t const *const key) {
  for (std::wstring const &arg : this->args)
    if (arg.compare(key) == 0) return true;
  return false;
}

bool Flags::operator[](wchar_t const *const key) { return exist(key); }

void Flags::collect_default_args(void) {
  this->_load = this->exist(L"--load");
  this->_save = this->exist(L"--save");
}

void Flags::register_global_scope(void) {
  if (flags != nullptr)
    throw std::runtime_error(
        "Flags has already been initiated somewhere else.");
  flags = this;
}

double Flags::get(wchar_t const *const key, double const default_) {
  std::wstring const key_(L"--" + std::wstring(key));
  for (size_t i(0_UZ); i != this->args.size(); ++i)
    if (this->args[i].compare(key_) == 0)
      return std::stod(this->args[i + 1_UZ]);
  return default_;
}
}  // namespace DL