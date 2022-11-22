/* Copyright 2016, 2022 S�bastien K�roack. All Rights Reserved.

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
#include <chrono>
#include <string>

namespace DL::Time {
#ifdef _DEBUG
#define USE_LOCAL_TIME true
#else
#define USE_LOCAL_TIME false
#endif

#ifdef COMPILE_WINDOWS
#define TIME_POINT std::chrono::steady_clock::time_point
#elif COMPILE_LINUX
#define TIME_POINT std::chrono::_V2::system_clock::time_point
#endif

#define CHRONO_NOW() std::chrono::high_resolution_clock::now()

double chrono_cast(TIME_POINT t0);

double chrono_cast(TIME_POINT t1, TIME_POINT t0);

void sleep(unsigned int const ms);

std::wstring date_now(bool const use_local_time = USE_LOCAL_TIME);

std::wstring datetime_now(bool const use_local_time = USE_LOCAL_TIME);

std::wstring now(bool const use_local_time = USE_LOCAL_TIME);

std::wstring now_format(wchar_t const *const fmt,
                        bool const use_local_time = USE_LOCAL_TIME);

std::wstring time_format(double time_elapse);
}  // namespace DL::Time