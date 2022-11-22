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

#pragma once

// Standard:
#include <iomanip>
#include <sstream>
#include <string>

namespace DL::Str {
#define CRLF L"\r\n";

#ifdef COMPILE_WINDOWS
constexpr auto OS_SEP = L'\\';
#elif COMPILE_LINUX
constexpr auto OS_SEP = L'/';

std::string wstring_to_utf8(std::wstring const &val);
#endif

std::wstring to_upper(std::wstring val);

std::wstring to_wstring(bool const val);
std::wstring to_wstring(char const *const val);
std::wstring to_wstring(std::string const &val);

template <typename T>
std::wstring to_wstring(
    T const val, int const precision = 16, bool const sign = false,
    std::ios_base::fmtflags fmt_type = std::ios_base::floatfield) {
  static_assert(std::is_fundamental<T>::value, "T is not a fundamental type.");
  
  std::wostringstream stream;

  if (sign && val >= T(0)) stream << L"+";

  stream << std::setprecision(precision);

  if (std::ios_base::floatfield != fmt_type)
    stream.setf(fmt_type, std::ios_base::floatfield);

  stream << val;

  return stream.str();
}
}  // namespace DL::Str
