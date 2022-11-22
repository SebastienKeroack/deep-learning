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

// Deep learning lib:
#include "deep-learning-lib/device/system/shutdown_block.hpp"
#include "deep-learning-lib/io/flags.hpp"
#include "deep-learning-lib/io/logger.hpp"

// Standard:
#include <string>

namespace DL {
class Session {
 public:
  Session(int const n_args, char const *const args[]);
  Session(int const n_args, wchar_t const *const args[]);
  ~Session(void);

  bool preparing_for_shutdown(void) const;

  Flags &flags(void);

  Logger &logger(void);

  Sys::ShutdownBlock &shutdownblock(void);

#ifdef COMPILE_ADEPT
  adept::Stack &stack(void);
#endif

  std::wstring const &workdir = this->_workdir;

 private:
  void enable_utf8(void);
  void initialize(void);
  void initialize_workspace(void);
  void register_global_scope(void);
  void register_global_variables(void);

  Flags *_flags = nullptr;

  Logger *_logger = nullptr;

  Sys::ShutdownBlock *_shutdownblock = nullptr;

#ifdef COMPILE_ADEPT
  adept::Stack *_stack = nullptr;
#endif

  std::wstring _workdir = L"";
};

inline class Session *session = nullptr;
}  // namespace DL