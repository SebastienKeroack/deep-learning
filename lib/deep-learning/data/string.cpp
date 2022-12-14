/* Copyright 2016, 2022 Sébastien Kéroack. All Rights Reserved.

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

// Standard:
#include <algorithm>
#include <codecvt>
#include <string>

namespace DL::Str {
#ifdef __linux__
std::string to_string(std::wstring const &val) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.to_bytes(val);
}
#endif

std::wstring to_upper(std::wstring val) {
  std::transform(val.begin(), val.end(), val.begin(), ::toupper);
  return val;
}

std::wstring to_wstring(bool const val) { return val ? L"true" : L"false"; }

std::wstring to_wstring(char const *const val) {
  return to_wstring(std::string(val));
}

std::wstring to_wstring(std::string const &val) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.from_bytes(val);
}
}  // namespace DL::Str
