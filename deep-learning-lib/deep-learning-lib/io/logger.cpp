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
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/io/logger.hpp"

// Deep learning lib:
#include "deep-learning-lib/data/string.hpp"
#include "deep-learning-lib/data/time.hpp"

// Standard:
#include <stdarg.h>

#include <iostream>
#include <stdexcept>
#include <thread>

using namespace DL::Str;
using namespace DL::Time;

namespace DL {
Logger::Logger(LOGLEVEL::TYPE const level) : level(level) {
  this->register_global_scope();
}

void Logger::operator()(LOGLEVEL::TYPE const &level,
                        std::wstring const &path_name, long int const &line_num,
                        wchar_t const *fmt...) {
  if (this->level > level) return;

  std::thread::id this_id = std::this_thread::get_id();

  std::wcout << L'[' << LOGLEVEL_NAME[level] << this_id << L"] ("
             << now_format(L"%d/%m/%Y %T ", USE_LOCAL_TIME)
             << path_name.substr(path_name.find_last_of(OS_SEP) + 1) << L':'
             << line_num << L"): ";

  va_list args;
  va_start(args, fmt);
  vwprintf(fmt, args);
  va_end(args);

  if (this->new_line) std::wcout << std::endl;
}

void Logger::register_global_scope(void) {
  if (logger != nullptr)
    throw std::runtime_error(
        "ShutdownBlock has already been initiated somewhere else.");
  logger = this;
}
}  // namespace DL