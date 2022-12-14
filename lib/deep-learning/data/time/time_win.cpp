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
#include "deep-learning/data/time.hpp"

// Standard:
#include <chrono>
#include <iomanip>
#include <sstream>

namespace DL::Time {
std::wstring now(bool const use_local_time) {
  // Get actual system time.
  std::chrono::time_point const clock_now(std::chrono::system_clock::now());

  // Get seconds since 1970/1/1 00:00:00 UTC.
  std::time_t const time(std::chrono::system_clock::to_time_t(clock_now));

  // Get time_point from `time` (note: no milliseconds).
  std::chrono::time_point const clock_rounded(
      std::chrono::system_clock::from_time_t(time));

  // Get milliseconds (difference between `clock_now` and `clock_rounded`).
  int const ms(static_cast<int>(
      std::chrono::duration<double, std::milli>(clock_now - clock_rounded)
          .count()));

  std::wostringstream output;

  std::wstring const tmp_fmt(L"%Y/%m/%d %T");

  struct tm ptime;

  if (use_local_time)
    localtime_s(&ptime, &time);
  else
    gmtime_s(&ptime, &time);

  output << std::put_time(&ptime, tmp_fmt.c_str());

  // Return ("[Y/MW/D H:M:S.xxx]").
  return L'[' + output.str() + L'.' + std::to_wstring(ms) + L']';
}

std::wstring now_format(wchar_t const *const fmt, bool const use_local_time) {
  std::wostringstream output;

  std::wstring tmp_fmt(fmt);

  time_t time(std::time(nullptr));

  struct tm ptime;

  if (use_local_time)
    localtime_s(&ptime, &time);
  else
    gmtime_s(&ptime, &time);

  output << std::put_time(&ptime, tmp_fmt.c_str());

  return output.str();
}
}  // namespace DL::Time
