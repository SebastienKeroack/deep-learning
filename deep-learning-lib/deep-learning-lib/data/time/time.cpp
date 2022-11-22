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
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/data/time.hpp"

// Deep learning lib:
#include "deep-learning-lib/data/string.hpp"

// FMT:
#include <fmt/core.h>
#include <fmt/xchar.h>

// Standard:
#include <cmath>
#include <thread>

using namespace DL::Str;

namespace DL::Time {
double chrono_cast(TIME_POINT t0) {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 CHRONO_NOW() - t0)
                 .count()) /
         1000.0;
}

double chrono_cast(TIME_POINT t1, TIME_POINT t0) {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                 .count()) /
         1000.0;
}

void sleep(unsigned int const ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

std::wstring date_now(bool const use_local_time) {
  return now_format(L"[%d/%m/%Y]", use_local_time);
}

std::wstring datetime_now(bool const use_local_time) {
  return now_format(L"[%d/%m/%Y %T]", use_local_time);
}

std::wstring time_format(double time_elapsed) {
  if (std::isinf(time_elapsed))
    return L"Inf";
  else if (std::isnan(time_elapsed))
    return L"NaN";

  bool const negate(time_elapsed < 0.0);
  if (negate) time_elapsed = -time_elapsed;

  std::wstring out;
  if (time_elapsed <= 0.000000999) {
    out = fmt::to_wstring(time_elapsed * pow(10, 6));
    out = out.substr(2, 3) + L"ns";
  } else if (time_elapsed <= 0.000999) {
    out = fmt::to_wstring(time_elapsed * pow(10, 3));
    out = out.substr(2, 3) + L"us";
  } else if (time_elapsed <= 0.999) {
    out = fmt::to_wstring(time_elapsed);
    out = out.substr(2, 3) + L"ms";
  } else if (time_elapsed <= 59.0) {
    int const s(static_cast<int>(time_elapsed));

    std::wstring ms(to_wstring(time_elapsed - floor(time_elapsed), 3));
    ms = ms.substr(ms.find(L'.') + 1, 3);

    if (ms.compare(L"0"))
      fmt::format_to(std::back_inserter(out), L"{:02d}.{}s", s, ms);
    else
      fmt::format_to(std::back_inserter(out), L"{:02d}s", s);
  } else if (time_elapsed <= 3599.0) {
    int const m(static_cast<int>(time_elapsed / 60.0));
    int const s(static_cast<int>(time_elapsed) % 60);

    std::wstring ms(to_wstring(time_elapsed - floor(time_elapsed), 3));
    ms = ms.substr(ms.find(L'.') + 1, 3);

    if (ms.compare(L"0"))
      fmt::format_to(std::back_inserter(out), L"{:02d}m:{:02d}.{}s", m, s, ms);
    else
      fmt::format_to(std::back_inserter(out), L"{:02d}m:{:02d}s", m, s);
  } else if (time_elapsed <= 86399.0) {
    int const h(static_cast<int>(time_elapsed / 3600.0));
    int const m(static_cast<int>(time_elapsed / 60.0) % 60);
    int const s(static_cast<int>(time_elapsed) % 60);

    std::wstring ms(to_wstring(time_elapsed - floor(time_elapsed), 3));
    ms = ms.substr(ms.find(L'.') + 1, 3);

    if (ms.compare(L"0"))
      fmt::format_to(std::back_inserter(out), L"{:02d}h:{:02d}m:{:02d}.{}s", h,
                     m, s, ms);
    else
      fmt::format_to(std::back_inserter(out), L"{:02d}h:{:02d}m:{:02d}s", h, m,
                     s);
  } else if (time_elapsed <= 31556952.0) {
    int const j(static_cast<int>(time_elapsed / 86400.0));
    int const h(static_cast<int>(time_elapsed / 3600.0) % 24);
    int const m(static_cast<int>(time_elapsed / 60.0) % 60);
    int const s(static_cast<int>(time_elapsed) % 60);

    std::wstring ms(to_wstring(time_elapsed - floor(time_elapsed), 3));
    ms = ms.substr(ms.find(L'.') + 1, 3);

    bool const include_ms(ms.compare(L"0"));

    if (j == 1) {
      if (include_ms)
        fmt::format_to(std::back_inserter(out),
                       L"1-day {:02d}h:{:02d}m:{:02d}.{}s", h, m, s, ms);
      else
        fmt::format_to(std::back_inserter(out),
                       L"1-day {:02d}h:{:02d}m:{:02d}s", h, m, s);
    } else {
      if (include_ms)
        fmt::format_to(std::back_inserter(out),
                       L"{}-days {:02d}h:{:02d}m:{:02d}.{}s", j, h, m, s, ms);
      else
        fmt::format_to(std::back_inserter(out),
                       L"{}-days {:02d}h:{:02d}m:{:02d}s", j, h, m, s);
    }
  } else {
    int const y(static_cast<int>(time_elapsed / 31556952.0));
    int const j(static_cast<int>(
        static_cast<double>(static_cast<int>(time_elapsed / 86400.0 * 1e+4) %
                            3652425) /
        1e+4));
    int const h(static_cast<int>(time_elapsed / 3600.0) % 24);
    int const m(static_cast<int>(time_elapsed / 60.0) % 60);
    int const s(static_cast<int>(time_elapsed) % 60);

    std::wstring ms(to_wstring(time_elapsed - floor(time_elapsed), 3));
    ms = ms.substr(ms.find(L'.') + 1, 3);

    bool const include_ms(ms.compare(L"0"));

    if (y == 1) {
      if (j == 0) {
        if (include_ms)
          fmt::format_to(std::back_inserter(out),
                         L"1-year {:02d}h:{:02d}m:{:02d}.{}s", h, m, s, ms);
        else
          fmt::format_to(std::back_inserter(out),
                         L"1-year {:02d}h:{:02d}m:{:02d}s", h, m, s);
      } else if (j == 1) {
        if (include_ms)
          fmt::format_to(std::back_inserter(out),
                         L"1-year 1-day {:02d}h:{:02d}m:{:02d}.{}s", h, m, s,
                         ms);
        else
          fmt::format_to(std::back_inserter(out),
                         L"1-year 1-day {:02d}h:{:02d}m:{:02d}s", h, m, s);
      } else {
        if (include_ms)
          fmt::format_to(std::back_inserter(out),
                         L"1-year {:03d}-days {:02d}h:{:02d}m:{:02d}.{}s", j, h,
                         m, s, ms);
        else
          fmt::format_to(std::back_inserter(out),
                         L"1-year {:03d}-days {:02d}h:{:02d}m:{:02d}s", j, h, m,
                         s);
      }
    } else {
      if (j == 0) {
        if (include_ms)
          fmt::format_to(std::back_inserter(out),
                         L"{:04d}-years {:02d}h:{:02d}m:{:02d}.{}s", y, h, m, s,
                         ms);
        else
          fmt::format_to(std::back_inserter(out),
                         L"{:04d}-years {:02d}h:{:02d}m:{:02d}s", y, h, m, s);
      } else if (j == 1) {
        if (include_ms)
          fmt::format_to(std::back_inserter(out),
                         L"{:04d}-years 1-day {:02d}h:{:02d}m:{:02d}.{}s", y, h,
                         m, s, ms);
        else
          fmt::format_to(std::back_inserter(out),
                         L"{:04d}-years 1-day {:02d}h:{:02d}m:{:02d}s", y, h, m,
                         s);
      } else {
        if (include_ms)
          fmt::format_to(std::back_inserter(out),
                         L"{:04d}-years {:03d}-days {:02d}h:{:02d}m:{:02d}.{}s",
                         y, j, h, m, s, ms);
        else
          fmt::format_to(std::back_inserter(out),
                         L"{:04d}-years {:03d}-days {:02d}h:{:02d}m:{:02d}s", y,
                         j, h, m, s);
      }
    }
  }

  if (negate)
    return L'-' + out;
  else
    return out;
}

}  // namespace DL::Time
