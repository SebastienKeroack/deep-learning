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

// PCH:
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/session.hpp"

// Deep learning lib:
#include "deep-learning-lib/data/string.hpp"
#include "deep-learning-lib/io/file.hpp"
#include "deep-learning-lib/io/flags.hpp"
#include "deep-learning-lib/io/logger.hpp"

// Standard:
#ifdef COMPILE_WINDOWS
#include <fcntl.h>
#include <io.h>
#elif COMPILE_LINUX
#include <locale>
#endif

#include <stdexcept>

using namespace DL::File;
using namespace DL::Str;
using namespace DL::Sys;

namespace DL {
Session::Session(int const n_args, char const *const args[]) {
  this->_flags = new Flags(n_args, args);
  this->initialize();
}

Session::Session(int const n_args, wchar_t const *const args[]) {
  this->_flags = new Flags(n_args, args);
  this->initialize();
}

Session::~Session(void) {
  if (this->_shutdownblock->unblock() == false)
    ERR(L"An error has been triggered from the `ShutdownBlock::unblock()` "
        L"function.");

  delete (this->_flags);
  delete (this->_logger);
  delete (this->_shutdownblock);

#ifdef COMPILE_ADEPT
  delete (this->_stack);
#endif
}

bool Session::preparing_for_shutdown(void) const {
  return this->_shutdownblock->preparing_for_shutdown();
}

Flags &Session::flags(void) { return *this->_flags; }

Logger &Session::logger(void) { return *this->_logger; }

ShutdownBlock &Session::shutdownblock(void) { return *this->_shutdownblock; }

#ifdef COMPILE_ADEPT
adept::Stack &Session::stack(void) { return *this->_stack; }
#endif

void Session::enable_utf8(void) {
#ifdef COMPILE_WINDOWS
  if (_setmode(_fileno(stdout), _O_U8TEXT) == FAIL)
    ERR(L"Couldn't set UTF-8 mode.");
#elif COMPILE_LINUX
  std::setlocale(LC_ALL, "");
#endif
}

void Session::initialize(void) {
  this->enable_utf8();

  this->register_global_scope();

  this->register_global_variables();

  this->initialize_workspace();
}

void Session::initialize_workspace(void) {
  this->_workdir = home_directory() + OS_SEP + L"deep-learning";

  if (create_directories(this->_workdir) == false)
    ERR(L"An error has been triggered from the `create_directories(%ls)` "
        L"function.",
        this->_workdir.c_str());
}

void Session::register_global_variables(void) {
  this->_logger = new Logger(LOGLEVEL::INFO);

#ifdef COMPILE_WINDOWS
  this->_shutdownblock = new ShutdownBlock(L"Deep Learning", L"Deep_Learning");
#elif COMPILE_LINUX
  this->_shutdownblock = new ShutdownBlock(L"Deep_Learning");
#endif

  if (this->_shutdownblock->block(true) == false)
    ERR(L"An error has been triggered from the `ShutdownBlock::block()` "
        L"function.");

#ifdef COMPILE_ADEPT
  this->_stack = new adept::Stack;
#endif
}

void Session::register_global_scope(void) {
  if (session != nullptr)
    throw std::runtime_error(
        "Session has already been initiated somewhere else.");
  session = this;
}
}  // namespace DL