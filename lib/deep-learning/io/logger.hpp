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

// Deep learning:
#include "deep-learning/data/enum/loglevel.hpp"
#include "deep-learning/data/string.hpp"

// Standard:
#include <string>

namespace DL {
class Logger {
 public:
  Logger(LOGLEVEL::TYPE const level);

  bool new_line = true;

  void operator()(LOGLEVEL::TYPE const &level, std::wstring const &path_name,
                  long int const &line_num, wchar_t const *fmt...);

  LOGLEVEL::TYPE level;

 private:
  void register_global_scope(void);
};

inline class Logger *logger = nullptr;

#define TOKENIZE_L_(expr) L_(expr)

#define DEBUG(fmt, ...)                                                \
  (*DL::logger)(LOGLEVEL::DEBUG, TOKENIZE_L_(__FILE__), __LINE__, fmt, \
                ##__VA_ARGS__)
#define INFO(fmt, ...)                                                \
  (*DL::logger)(LOGLEVEL::INFO, TOKENIZE_L_(__FILE__), __LINE__, fmt, \
                ##__VA_ARGS__)
#define WARN(fmt, ...)                                                \
  (*DL::logger)(LOGLEVEL::WARN, TOKENIZE_L_(__FILE__), __LINE__, fmt, \
                ##__VA_ARGS__)
#define ERR(fmt, ...)                                                \
  (*DL::logger)(LOGLEVEL::ERR, TOKENIZE_L_(__FILE__), __LINE__, fmt, \
                ##__VA_ARGS__)
#define FATAL(fmt, ...)                                                \
  (*DL::logger)(LOGLEVEL::FATAL, TOKENIZE_L_(__FILE__), __LINE__, fmt, \
                ##__VA_ARGS__)

#ifdef _WIN32
#define L_GIZING(expr) (L#expr)
#elif __linux__
#define L_GIZING(expr) (L###expr)
#endif

#define ASSERT_MSG(expr, fmt, ...) \
  (void)((!!(expr)) || (FATAL(L_GIZING((expr) && (fmt)), ##__VA_ARGS__), 0))
}  // namespace DL
